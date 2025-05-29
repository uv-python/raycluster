#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Author: Ugo Varetto

# TODO: Consider using --block, instead of os.kill, for workers other than the one waiting
#       for vLLM sto start.
""" This script launches Ray and optionally vLLM from container.

It can be started on head and worker nodes in any order.

Because vLLM requires a running Ray instance to perform distributes inference,
Ray must be started on each node before vLLM is started on the head node.
And the Ray head node process must be started before the worker processes.
The script first ensures that all the scripts have started on each node and
then waits for all the Ray processes to be started before executing vLLM
on the head node.
Workers use a multicast group to communicate their IP address and wait for
an ACK from the head node which returns the port to use.
The head node starts Ray then sends the IP address to use to the workers
which then start Ray passing the IP address and port received from the head
process on the command line to the ray executable.

Different parts of the scripts are executed only by the head or worker processes,
according to the following sequence table.

                                   Head process                                |           Worker process
-------------------------------------------------------------------------------------------------------------------------
|                                                                              |
|1. parse paramerters                                                          |   parse parameters
|2. wait on the multicast group to receive the address of each worker process  |   broadcast ip address    
|3. receive IP address of each worker                                          |                                           
|4. send ACK to workers                                                        |
|                                                                              |   receive ACK from head node
|                                              All scripts started             |                        
|-------------------------------------------------------------------------------------------------------------------------
|5. launch Ray process                                                         |   wait to receive IP address of head node
|6. broadcast address of head node                                             |
|7. wait to receive IP address from workers                                    |   receive IP address of head node
|8.                                                                            |   launch Ray worker process
|9.*                                                                           |   broadcast IP address*
|10. receive IP address of each worker*                                        |   pause execution (SIGSTOP)
|                                                                              |   
|                               ALL Ray processes started (pause here if vLLM not to be started)
|-------------------------------------------------------------------------------------------------------------------------
|                                                                              |
|11. launch vLLM                                                               |
|                                                                              |
V (time)                                                                       | 

Two multicast groups are used:
    1) for sending data from worker head
    2) for sending data from head to worker

(3) uses the same port but is generated from (1) by adding '1' to the last byte
of the address.


(*): Resusing same code to sync, IP address is not used.

It is possible to specify both port and multicast group; the default multicast group is
224.0.0.100 making it non-routable and the default port is 5001.
All the command line parameters are documented, just run the script with  --help to see a list.
The 'slurm-all-to-all-udp-test.py' script allows to check connectivity among all nodes
in a cluster.

Additional parameters to the container runner can be passed through the --container-parameters
argument.

When running inside SLURM the --slurm argument allows to automatically select head an workers nodes.

It is possible to have the script automatically configure vLLM using information from
the SLURM environment by specifying the --auto parameter together with --slurm; note
however that the configuration depends on the structure of the model e.g. the number
of parallel tensor layers must be divisible by the number of attention heads in the model.
The 'num-gpu-from-attention-heads.py' can be used to compute the number of GPUs and number of parallel
tensors for the model.


Examples:

slurm: autodetect head and workers
```
srun ./start-cluster.py singularity ./vllm_latest.sif --slurm \
     --num-gpus 8  --mode lQwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

slurm + auto:
```
srun ./start-cluster.py singularity ./vllm_latest.sif --slurm --auto --model Qwen/Qwen3-30B-A3B
```

Run on head node, two nodes, one worker, one head, 8 GPUs per node:
```
./start-cluster.py singularity ./vllm_latest.sif --head --num-gpus 8 \
   --num-workers 1 --model Qwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2
```
Run on worker node:
```
./start-cluster.py singularity ./vllm_latest.sif --num-gpus 8

SLURM job:
```
#!/usr/bin/env bash
#SBATCH --job-name=vLLM
#SBATCH --account=<account name>
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --partition=<partition name>
module load singularity # ensure a container runner is available
srun ./start-cluster.py singularity ./vllm_late --slurm --auto --model Qwen/Qwer3-30B-A3B
```
"""

import os
import subprocess as sub
import argparse
import re
import socket
import struct
import sys
import ipaddress
import signal
from dataclasses import dataclass
import random
import selectors


@dataclass
class VLLMConfig:
    tensor_parallel: int = 0
    pipeline_parallel: int = 0
    num_gpus: int = 0


# keep process active in stopped state, useful
# to avoid SLURM to end job when current process is
# done spawning backgound processes
def pause():
    os.kill(os.getpid(), signal.SIGSTOP)


def get_host_name(addr: str) -> str:
    h: str = socket.gethostbyaddr(addr)[0]
    return h.split(".")[0]


# A worker node is randomly selecte to wait until the vLLM service
# is available and then either invokes a user-provided script or
# stores the service URL into a local file.
def notify_loop(head, port, slurm, script="") -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while s.connect_ex(
            (head, port)
        ):  # returns a value other than zero when it fails
            pass
    if script:
        try:
            sub.call([script, "http://" + head + ":" + str(port)])
        except Exception as e:
            abort("Error invoking notification script\n" + str(e))
    else:
        try:
            jid = os.environ["SLURM_JOB_ID"]
            with open(os.environ["HOME"] + f"/{jid}-vllm-url", "w") as f:
                f.write(f"http://{head}:{port}")
        except Exception as e:
            abort(f"Cannot write to file - {str(e)}")

    if slurm:
        pause()
    else:
        sys.exit(0)


# Return configuration from SLURM Job info
def vllm_config_from_slurm_job() -> VLLMConfig:
    # --tensor-parallel = num gpus per node
    # --pipeline-parallel = num nodes
    if "SLURM_JOB_GPUS" not in os.environ:
        abort("SLURM_JOB_GPUS env var not set")
    num_gpus: int = len(os.environ["SLURM_JOB_GPUS"].split(","))
    if "SLURM_JOB_NUM_NODES" not in os.environ:
        abort("SLURM_JOB_NUM_NODES env var not set")

    num_nodes: int = int(os.environ["SLURM_JOB_NUM_NODES"])
    return VLLMConfig(num_gpus, num_nodes, num_gpus)


# Randomly select the node that will check on the availability of the vLLM service.
def select_notifier_node(nodes, head_node) -> str:
    nn: str = head_node
    while nn == head_node:
        nn = "nid" + random.choice(nodes)
    return nn


# Abort execution printing an error end exiting.
def abort(msg: str, exit_code: int = 1) -> None:
    print(msg, sys.stderr)
    sys.exit(exit_code)


# Check if IP address is valid.
def valid_ip_address(addr: str) -> bool:
    try:
        ipaddress.ip_address(addr)
    except:
        return False
    return True


# Check that multicast group address is correct.
def valid_mcast_address(addr: str) -> bool:
    b = addr.split(".")
    if len(b) != 4:
        return False
    try:
        if int(b[0]) < 224 or int(b[0]) > 239:
            return False
        if int(b[1]) < 0 or int(b[1]) > 255:
            return False
        if int(b[2]) < 0 or int(b[2]) > 255:
            return False
        if int(b[3]) < 0 or int(b[3]) > 255:
            return False
        return True
    except:
        return False


# Check if TCP/UPD port is valid.
def valid_port(p: int) -> bool:
    return 1024 < p < 65536


# Send Ray port number to workers, called by head process
def notify_client(client: str, port: int, ray_port: int) -> None:
    try:
        sock: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        sock.sendto(bytes(str(ray_port), "utf-8"), (client, port))
    except:
        abort("Error creating socket")


# Synchronously receive IP addressess from al workers on
# the multicast group and send back port of ray process to each worker
def sync_with_workers(
    mcast_group: str, port: int, num_workers: int, ray_port: int
) -> set[bytes]:
    try:
        sock: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((mcast_group, port))
        # add socket to multicast group by passing the multicast group
        # address to setsockopt as 32 bit long integer ('l') packed
        # into four bytes ('4s') converting from string (inet_aton)
        mreq: bytes = struct.pack(
            "4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        w: set[bytes] = set()
        # at each iteration a message from a worker is received; because the same
        # worker might be sending more than one message a 'set' is used to record
        # which workers have already notified the head node ggto ensure one unique
        # entry is stored
        while len(w) != num_workers:
            client = sock.recv(64)
            notify_client(client.decode("utf-8"), port + 1, ray_port)
            w.add(client)
        return w
    except:
        abort("Error synchronising with workers")
        return set()  # LSP tool does not understand that abort terminates the program
    finally:
        sock.close()


# Broadcast node IP address to multicast group and wait to receive Ray port from
# head node as ACK
def sync_with_head(
    mcast_group: str,
    port: int,
    ray_ip_address,
    ttl: int = 4,
    select_timeout: float = 2,
) -> int:
    try:
        sel: selectors.DefaultSelector = selectors.DefaultSelector()
        sock: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        # Sync happens twice once before launching Ray, when ray_ip_address in empty
        # and once after Ray is started when ray_ip_address contains the IP extracted from
        # Ray's output.
        msg: str = (
            socket.gethostbyname(socket.getfqdn())
            if not ray_ip_address
            else ray_ip_address
        )
        sock2: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock2.bind(("0.0.0.0", port + 1))
        sel.register(sock2, selectors.EVENT_READ, data=None)
        rayport: bytes = bytes()
        # At each iteration a new broadcast message is sent and an attempt
        # at receiving a response from the head node is performed.
        # If the recvfrom call fails an exception is thrown, hence the need
        # for a try/except block
        while True:
            sock.sendto(msg.encode(), (mcast_group, port))
            events = sel.select(timeout=select_timeout)
            if not events:
                continue
            rayport, _ = sock2.recvfrom(128)
            return int(rayport.decode("utf-8"))

    except Exception as e:
        abort("Error synchronising with head node\n" + str(e))
        return 0  # lsp tool does not understand that abort terminates the program
    finally:
        sock.close()
        sel.unregister(sock2)
        sock2.close()


# Receive message broadcast to multicast group.
# Invoked from worker process to receive IP address of head node.
def ip_address_receive(mcast_group: str, port: int) -> str:
    try:
        sock: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((mcast_group, port))
        mreq: bytes = struct.pack(
            "4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        return sock.recv(128).decode("utf-8")
    except:
        abort("Error receiving messages from workers")
        return ""  # lsp tool does not understand that abort terminates the program
    finally:
        sock.close()


# Broadcast head node IP address. Called by head node after all worker process are started
# and waiting for message.
def broadcast_ip_address(ip: str, mcast_group: str, port: int, ttl: int = 3) -> None:
    try:
        sock: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        sock.sendto(ip.encode(), (mcast_group, port))
    except:
        abort("Error broadcasting messages")
    finally:
        sock.close()


# Remove ANSI escape characters from text.
def remove_ansi_escape_chars(buffer: str) -> str:
    ansi_escape: re.Pattern = re.compile(r"\x1b[^m]+m")
    t: str = ""
    try:
        t = ansi_escape.sub("", buffer)
        return t
    except:
        return ""


# Extract current IP address from Ray output, note that it is not possible to
# retrieve the address through other means beause it is not known which
# NIC/IP address will be used.
# The stdout output returned by subprocess functions is an array of bytes
# containing ANSI escape sequences which need to be removed to simplify parsing.
def extract_ip_address(buffer: bytes) -> str:
    """Parse Ray output text and extract IP address.
    `gethostbyname` won't work when there are multiple IP addresses
    bound to the same node.
    """
    # Doesn't work when more than one IP address in the same subnet
    # return socket.gethostbyname(socket.getfqdn());
    t: list[str] = []
    t = remove_ansi_escape_chars(buffer.decode("utf-8")).split()
    if not t:
        return ""
    try:
        ip = t[t.index("IP:") + 1]
        return ip
    except:
        return ""


# Return the list of nodes running the SLURM job.
def slurm_nodelist() -> tuple[bool, list[str]]:  # return <OK | NOT OK, value>
    OK: bool = True
    if "SLURM_JOB_NODELIST" not in os.environ:
        return (not OK, [])
    r = re.compile(r"nid\[([^\]]+)\]")
    result = r.search(os.environ["SLURM_JOB_NODELIST"])
    if not result:
        return (not OK, [])
    try:
        n = result.group(1)
        return (OK, sorted(n.split(",")))
    except:
        return (not OK, [])


# Check if current node is the head node. The head node is the first
# node in the sorted sequence of SLURM nodes.
def is_head_from_slurm_nodelist(nodelist: list[str]) -> bool:
    return ("nid" + nodelist[0]) == socket.gethostname()


# Return the number of worker nodes.
def num_workers_from_slurm_nodelist(nodelist: list[str]) -> int:
    return len(nodelist) - 1  # one is the head process


#
# -------------------------------------------------------------------------------
#
def main() -> None:
    # ----------------------------------------------------------------------------
    # 1. Configure environment
    # check if ray alreay active:
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")

    hostname = socket.gethostname()

    # ----------------------------------------------------------------------------
    # 2. Parse arguments
    parser = argparse.ArgumentParser(
        prog="start-cluster",
        description="Run Ray and vllM containers; workers and head processes"
        "can be started independently in any order and the port needs only be "
        "specified for the head node. The same container is used to start both "
        "Ray and vLLM; if no extra command line parameters beyond the ones"
        "required for ray are specified, vLLM is not run.\n"
        "The command line parameters are passed to 'vllm serve'"
        "'--distributed-executor-backend ray' is added to the 'vllm serve' command line",
    )
    parser.add_argument(
        "container_runner", help="Container runner, Singularity, Apptainer, Podmason..."
    )
    parser.add_argument(
        "container_image", help="Path to container must contain Ray and if additinal "
    )
    parser.add_argument(
        "--container-args",
        help="string containing a space separated list of command line "
        "argument to pass to the 'container runner'",
    )
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs")
    parser.add_argument("--head", const=True, type=bool, nargs="?", help="Head node")
    parser.add_argument(
        "--model", help="Huggingface model path e.g. Qwen/Qwer3-30B-A3B"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Ray TCP port, for head only, workers will receive port from head",
    )
    parser.add_argument(
        "--mcast-address",
        help="Multicast address, default is 224.0.0.100, "
        "two multicast addresses are used, the second one is "
        "obtained by incrementing the las byte by one",
    )
    parser.add_argument("--mcast-port", help="Multicast port, default is 5001")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Used by head node to wait until all workers are active",
    )
    parser.add_argument("--app-dir", help="Local directory mapping /app")
    parser.add_argument(
        "--hf-dir",
        help="Local mapping of Huggingface /root/.cache/hugingface directory",
    )
    parser.add_argument(
        "--dashboard",
        const=True,
        type=bool,
        nargs="?",
        help="Enable Ray dashboard by installing the proper version of Ray and additional packages through pip",
    )
    parser.add_argument(
        "--slurm",
        const=True,
        type=bool,
        nargs="?",
        help="Automatically select head node and workers",
    )
    parser.add_argument(
        "--auto",
        const=True,
        type=bool,
        nargs="?",
        help="When --slurm enabled it generates a configuration for vllm "
        "using the SLURM job information",
    )
    parser.add_argument(
        "--notification-script",
        help="call the specified script passing <node name> <node ip> <port> on the command line",
    )
    parser.add_argument(
        "--vllm-port", type=int, help="vLLM port"
    )  # need to know to run notification loop

    # TTL: NOT USED YET
    parser.add_argument(
        "--ttl", type=int, help="TTL (number of hops) for multicast packets"
    )

    parser.add_argument("--ip-address", help="Ray's --node-ip-address parameter")

    app_args, vllm_args = parser.parse_known_args()  # known, unknown

    if hasattr(app_args, "vllm_port") and "--port" in vllm_args:
        abort("Only set port with --vllm-port, do not use vllm --port parameter")

    if not app_args.model:
        print(
            "WARNING: No model selected only Ray will be started not vLLM, to launch vLLM specify model through --model argument"
        )

    # 'vllm_args' are the parameters after `vllm serve'`
    if not app_args.slurm:
        try:
            _ = sub.check_output(
                [
                    app_args.container_runner,
                    "exec",
                    app_args.container,
                    "ray",
                    "status",
                ],
                stderr=sub.STDOUT,
            )
            print("Ray already running, exiting...")
            sys.exit(1)
        except:
            pass

    port: int = app_args.port or 6379
    if not valid_port(port):
        abort("Invalid port")

    head: bool = False
    worker: bool = False
    num_workers: int = 0
    vllm_config: VLLMConfig = VLLMConfig()
    if app_args.slurm:
        print("Autodetecting head and workers...")
        ok, nodes = slurm_nodelist()
        if not ok:
            abort("Cannot retrieve SLURM nodelist")
        head = is_head_from_slurm_nodelist(nodes)
        worker = not head
        num_workers = num_workers_from_slurm_nodelist(nodes)
        if app_args.auto:
            vllm_config = vllm_config_from_slurm_job()
        if head:
            print(f"Head is: {hostname}")
            print(f"{num_workers} workers")
        else:
            print(f"Worker: {hostname}")

    else:
        head = app_args.head or False
        worker = not head
        if head and not app_args.num_workers:
            abort(
                "When --head specified --num-workers is required because the head node needs to know "
                "how many workers it needs to wait for"
            )
        num_workers = app_args.num_workers

    num_gpus: int = app_args.num_gpus or (
        vllm_config.num_gpus if vllm_config.num_gpus > 0 else 1
    )
    # used by the workers to send their IP address to the head node
    worker_to_head_mcast: str = app_args.mcast_address or "224.0.0.100"  # non routable
    if not valid_mcast_address(worker_to_head_mcast):
        abort("Invalid multicast ip address")
    ip: list[str] = worker_to_head_mcast.split(".")
    ip[3] = str(int(ip[-1]) + 1)
    # used by the head ndoe to sennd its IP address to the worker nodes
    head_to_workers_mcast: str = ".".join(ip)
    mcast_port: int = app_args.mcast_port or 5001

    if not valid_port(mcast_port):
        abort("Invalid multicast port")

    if app_args.auto and not app_args.slurm:
        print(
            "Warning 'auto' is only available when 'slurm' selected, won't be generating vllm configuration"
        )

    # 2.1 Optionally install dashboard
    if head and app_args.dashboard:
        sub.call(
            [
                app_args.container_runner,
                "exec",
                app_args.container_image,
                "pip",
                "install",
                "ray[default]",
                "py-spy",
                "memray",
            ]
        )

    # ----------------------------------------------------------------------------
    # 3. Sync workers with head

    # sync head with workers, head and workers can start in any order
    # workers keep sending a broadcast message and wait for a response from the head node
    # the head node replies to each worker with the ray port to use
    workers: set[bytes] = set()
    if head:
        workers = sync_with_workers(worker_to_head_mcast, mcast_port, num_workers, port)
    else:
        port = sync_with_head(worker_to_head_mcast, mcast_port, app_args.ip_address)

    # --------------------------------------------------------------------------
    # SYNC PONT 1: ALL SCRIPTS STARTED ON ALL NODES
    # --------------------------------------------------------------------------

    head_address: str = ""

    # WORKERS: BLOCK and wait for IP address from head node
    if worker:
        head_address = ip_address_receive(head_to_workers_mcast, mcast_port)

    # HEAD: continue execution

    # execution continues only on head node until the IP address has been extracted
    # from Ray's output and sent to workers which receive it in the line above,
    # note that because there are normally multiple IP addresses on the node, it it not
    # safe to retreive the IP address through other means

    # ----------------------------------------------------------------------------
    # 4. Run Ray

    # execute ray with the container
    execute_ray: list[str] = [
        app_args.container_runner,
        "exec",
        app_args.container_image,
        "ray",
    ]
    cmd_line: list[str] = []
    if head:  # head node reaches this point before workers
        cmd_line = execute_ray + [
            "start",
            "--head",
            "--port",
            str(port),
            "--num-gpus",
            str(num_gpus),
        ]
        if app_args.dashboard:
            cmd_line += ["--dashboard-host", "0.0.0.0"]
        else:
            cmd_line += ["--include-dashboard=False"]
    else:  # workers reach this point after head
        cmd_line = execute_ray + [
            "start",
            "--num-gpus",
            str(num_gpus),
            "--address",
            str(head_address) + ":" + str(port),
        ]

    if app_args.ip_address:
        cmd_line += ["--node-ip-address", app_args.ip_address]
    # print(' '.join(cmd_line))
    out: bytes = bytes()
    try:
        out = sub.check_output(cmd_line)
        print(remove_ansi_escape_chars(out.decode("utf-8")))
    except Exception as e:
        abort(str(e))

    # 4.1 Extract IP address

    local_ip: str = extract_ip_address(out)
    print(f"IP Address: {local_ip}")

    # --------------------------------------------------------------------------
    # 5. Broadcast IP address of head process
    # send head address to workers waiting at SYNC POINT 1
    if head:
        broadcast_ip_address(local_ip, head_to_workers_mcast, mcast_port)

    # --------------------------------------------------------------------------
    # 6. Re-sync

    # At this point, we know that all the worker processes have started but we do not know if
    # each process has started the Ray process and therefore we need to sync again:
    # we need to exit the program only after we are sure that all Ray's worker and head processes
    # have started.
    # This time the worker and head processeswill send a messagesto after Ray has been started.
    # Instead of write additional code we can reuse what implemented above.
    if head:
        workers = sync_with_workers(worker_to_head_mcast, mcast_port, num_workers, port)
    else:
        _ = sync_with_head(worker_to_head_mcast, mcast_port, local_ip)

    # --------------------------------------------------------------------------
    # SYNC POINT 2: RAY STARTED ON ALL NODES
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # 7. Print IP addresses
    if worker:
        print(f"Head node address: {head_address}")
    else:
        print("Workers: ")
        print("=" * 10)
        for w in workers:
            print(w.decode("utf-8"))
    if head:
        sub.call(
            [
                app_args.container_runner,
                "exec",
                app_args.container_image,
                "ray",
                "status",
            ]
        )

    # --------------------------------------------------------------------------
    # 8. Launch notification loop on selected worker node

    # keep trying to connect to http://<head node>:<port> until
    # connection is established
    if worker and app_args.model:
        ok, nodes = slurm_nodelist()
        if not ok:
            abort("Failed to retrieve job node list")
        notifier = select_notifier_node(nodes, get_host_name(head_address))
        if not notifier:
            abort("Failed to select notifier process")
        vllm_port = app_args.vllm_port if app_args.vllm_port else 8000
        if socket.gethostname() == notifier:
            notify_loop(
                get_host_name(head_address),
                vllm_port,
                app_args.slurm,
                app_args.notification_script,
            )

    # --------------------------------------------------------------------------
    # 9. Launch vllm if '--model' specified

    # if no model specified do not launch vllm and pause execution
    # when run from withing SLURM or exit
    if not app_args.model or worker:
        if not app_args.slurm:
            sys.exit(0)
        else:
            pause()

    # 9.1 Launch vllm

    # launch vllm on the head node
    os.environ["VLLM_HOST_IP"] = local_ip
    vllm_cmdline: list[str] = []
    vllm_auto_args = []
    vllm_args += ["--port", str(app_args.vllm_port if app_args.vllm_port else 8000)]
    vllm_args += ["--distributed-executor-backend", "ray"]
    if vllm_config.num_gpus > 0:
        vllm_auto_args = [
            "--tensor-parallel-size",
            str(vllm_config.tensor_parallel),
            "--pipeline-parallel-size",
            str(vllm_config.pipeline_parallel),
        ]

    if app_args.container_args:
        cargs = app_args.container_args.split()
        vllm_cmdline = (
            [app_args.container_runner, "exec"]
            + cargs
            + [app_args.container_image, "vllm", "serve", app_args.model]
            + vllm_args
            + vllm_auto_args
        )
    else:
        vllm_cmdline = (
            [
                app_args.container_runner,
                "exec",
                app_args.container_image,
                "vllm",
                "serve",
                app_args.model,
            ]
            + vllm_args
            + vllm_auto_args
        )

    print(" ".join(vllm_cmdline))
    try:
        sub.call(vllm_cmdline)
        print("Started!")
    except Exception as e:
        print("Error running vLLM")
        print(e)
        sys.exit(1)


# Entry point
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
