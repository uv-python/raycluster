#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Author: Ugo Varetto
# TODO: consider adding timeouts for sync operations, 
#       when running in SLURM:
#           * Add notifier process which either execute scripts or saves file with name
#             <job id>--<head node name>--<head node ip address>--<vllm port>
#           * Add option to specify script to run on the command line
        
"""
This script allows launching containers for Ray and optionally vLLM.
Ideally you should run the same version of Ray contained in the vLLM container
to avoid problems.
For AMD GPUs: make sure the amdgpu python package is NOT visible when running this script.
It is possible to launch workers and head processes in any order:
    1. the worker processes start then keep broadcas messages with their own IP address
       until they receive a response form the head node containing the TCP port to connect to;
    2. the worker process receives a messag from the head process and then waits until it receives
       the IP address of the node to connect to;
    3. the head process starts and waits until it has finished receiving messages from all
       the workers to which it replies with the port to connect to;
    4. the head process starts the Ray process and then sends a multicast message containing
       the IP address to connect to;
    5. after all the process are started another synchronisation step is performend after the
       Ray processes are started by both the workers and the head node to ensure the Ray
       cluster is up and running before running anything else;
    6. the head node runs vLLM if 'vllm serve' command line parameters are specified on 
       the command line;
    7. the worker nodes stop by sending a SIGSTOP signal to themselves after launching the
       Ray process.

It is possible to specify both port and multicast group; the default multicast group is
224.0.0.100 making it non-routable.
All the command line parameters are documented, just run the script with  --help to see a list.

When running inside SLURM and specifying the --slurm switch head and workers are automatically
detected and run:

```
srun ./start-cluster.py singularity ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif --slurm \
     --num-gpus 8  Qwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

The script has been so far used only with Singularity and Apptainer on HPE/Cray systems but
nothing is specific to the environment so it should work on any Linux cluster.

Example:

Run on head node, two nodes, one worker, one head, 8 GPUs per node:
```
./start-cluster.py singularity ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif --head --num-gpus 8 \
   --broadcast --num-workers 1 Qwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2
```
Run on worker node:
```
./start-cluster.py singularity ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif --num-gpus 8
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

# Workers' IP address
workers : set[bytes] = set()

# IP address rad from Ray outputj
ray_ip_address : str = ""

def abort(msg: str, exit_code: int =1) -> None:
    print(msg, sys.stderr)
    sys.exit(exit_code)

def valid_ip_address(addr: str) -> bool:
    try:
        ipaddress.ip_address(addr)
    except:
        return False
    return True

def valid_port(p: int) -> bool:
    return 1024 < p < 65536

def notify_client(client: str, port: int, ray_port: int) -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.sendto(bytes(str(ray_port), 'utf-8'), (client, port))
    except:
        abort("Error creating socket")


def sync_with_workers(mcast_group: str, port: int, num_workers: int, ray_port: int) -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((mcast_group, port))
        mreq = struct.pack("4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        w = set()
        # at each iteration a message from a worker is received; because the same
        # worker might be sending more than one message a 'set' is used to record
        # which workers have already notified the head node to ensure one unique
        # entry is stored
        while len(w) != num_workers:
            client = sock.recv(64)
            notify_client(client.decode('utf-8'), port+1, ray_port);
            w.add(client)
        global workers
        workers = w
    except:
        abort("Error synchronising with workers")
    finally:
        sock.close()

def sync_with_head(mcast_group: str, port: int, ttl: int = 3 ) -> int:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        # Sync happens twice once before launching Ray, when ray_ip_address in empty
        # and once after Ray is started when ray_ip_address contains the IP extracted from
        # Ray's output.
        msg= socket.gethostbyname(socket.getfqdn()) if not ray_ip_address else ray_ip_address

        sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock2.bind(("0.0.0.0", port+1))
        sock2.setblocking(False);
        done = False
        rayport : bytes = bytes()
        # at each iteration a new broadcast message is sent and an attempt
        # at receiving a response from the head node is performed.
        # if the recvfrom call fails an exception is thrown, hence the need
        # for a try/except block
        while not done:
            sock.sendto(msg.encode(), (mcast_group, port))
            try:
                if (x := sock2.recvfrom(128)):
                    rayport = x[0] 
                    done = True
            except:
                pass
        return int(rayport.decode('utf-8'))
    except Exception as e:
        abort("Error synchronising with head node\n" + str(e))
        return 0 # lsp tool does not understand that abort terminates the program
    finally:
        sock.close()
        sock.close()

def mcast_address_receive(mcast_group: str, port: int) -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((mcast_group, port))
        mreq = struct.pack("4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        return sock.recv(128).decode('utf-8')
    except:
        abort("Error receiving messages from workers")
        return "" # lsp tool does not understand that abort terminates the program
    finally:
        sock.close()

def broadcast_ip_address(ip: str, mcast_group: str, port: int, ttl: int = 3) -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        sock.sendto(ip.encode(), (mcast_group, port))
    except:
        abort("Error broadcasting messages")
    finally:
        sock.close()

def remove_ansi_escape_chars(buffer) -> str:

    ansi_escape = re.compile(r'\x1b[^m]+m')
    t : str = "" 
    try:
        t = ansi_escape.sub('', buffer.decode('utf-8'))
        return t
    except:
        return ""

def extract_ip_address(buffer: bytes) -> str:
    """ Parse Ray output text and extract IP address.
    `gethostbyname` won't work when there are multiple IP addresses
    bound to the same node.
    """
    #Doesn't work when more than one IP address in the same subnet
    #return socket.gethostbyname(socket.getfqdn());
    ansi_escape = re.compile(r'\x1b[^m]+m')
    t : list[str] = []
    try:
        t = ansi_escape.sub('', buffer.decode('utf-8')).split()
    except:
        return ""
    return t[t.index('IP:') + 1]


def slurm_nodelist() -> tuple[bool, list[str]]: # return <OK | NOT OK, value>
    OK : bool = True
    if not "SLURM_JOB_NODELIST" in os.environ:
        return (not OK, False)
    r = re.compile(r"nid\[([^\]]+)\]")
    result = r.search(os.environ["SLURM_JOB_NODELIST"])
    try:
        n = result.group(1)
        return (OK, sorted(n.split(',')))
    except:
        return (not OK, [])

def is_head_from_slurm_nodelist(nodelist: list[str]) -> bool:
    return ('nid' + nodelist[0]) == socket.gethostname()

def num_workers_from_slurm_nodelist(nodelist: list[str]) -> int:
    return len(nodelist) - 1 # one is the head process

#
#-------------------------------------------------------------------------------
#

def main():

# 1. Configure environment
    #check if ray alreay active:
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    try:
        _ = sub.check_output(['ray', 'status'], stderr=sub.STDOUT)
        print("Ray already running, exiting...")
        sys.exit(1)
    except:
        pass
    
    hostname = socket.gethostname()

# 2. Parse arguments
    parser = argparse.ArgumentParser(prog="start-cluster", description="Run Ray and vllM containers; workers and head processes" \
                                                                  "can be started independently in any order and the port needs only be " \
                                                                  "specified for the head node. The same container is used to start both " \
                                                                  "Ray and vLLM; if no extra command line parameters beyond the ones" \
                                                                  "required for ray are specified, vLLM is not run.\n" \
                                                                  "The command line parameters are passed to 'vllm serve'" \
                                                                  "'--distributed-executor-backend ray' is added to the 'vllm serve' command line")
    parser.add_argument("container_runner", help="Container runner, Singularity, Apptainer, Podmason...")
    parser.add_argument("container_image", help="Path to container must contain Ray and if additinal ")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs")
    parser.add_argument("--head", const=True, nargs='?', help="Head node")
    parser.add_argument("--port", type=int, help="Ray TCP port, for head only, workers will receive port from head")
    parser.add_argument("--mcast-address", help="Multicast address, default is 224.0.0.100")
    parser.add_argument("--mcast-port", help="Multicast port, default is 5001")
    parser.add_argument("--num-workers", type=int, help="Used by head node to wait until all workers are active")
    parser.add_argument("--app-dir", help="Local directory mapping /app")
    parser.add_argument("--hf-dir", help="Local mapping of Huggingface /root/.cache/hugingface directory")
    parser.add_argument("--dashboard", const=True, nargs='?', 
                        help="Enable Ray dashboard by installing the proper version of Ray and additional packages through pip")
    parser.add_argument("--slurm", const=True, nargs='?', help="Automatically select head node and workers")
                    
    ray_args, vllm_args = parser.parse_known_args() # known, unknown
    #'unknown' are the parameters after `vllm serve'`
    
    port : int = ray_args.port or 6379
    if not valid_port(port):
        abort("Invalid port")

    head : bool = False
    worker : bool  = False
    num_workers : int = 0
    if ray_args.slurm:
        print("Autodetecting head and workers...")
        ok, nodes = slurm_nodelist() 
        if not ok:
            abort("Cannot retrieve SLURM nodelist")
        head = is_head_from_slurm_nodelist(nodes)
        worker = not head
        num_workers = num_workers_from_slurm_nodelist(nodes)
        if head:
            print(f"Head is: {hostname}")
            print(f"{num_workers} workers")

    else:
        head : bool = ray_args.head or False
        worker : bool = not head 
        if head and not ray_args.num_workers:
            abort("When --head specified --num-workers is required because the head node needs to know " \
                  "how many workers it needs to wait for")
        num_workers = ray_args.num_workers 

    num_gpus : int = ray_args.num_gpus or 0
    mcast_address : str = ray_args.mcast_address or "224.0.0.100" # non routable
    mcast_port : int = ray_args.mcast_port or 5001
    if not valid_ip_address(mcast_address):
        abort("Invalid multicast ip address")
        
    if not valid_port(mcast_port):
        abort("Invalid multicast port")

# 2.1 Optionally install dashboard
    if head and ray_args.dashboard:
        sub.call([ray_args.container_runner, 'exec', ray_args.container_image, 'pip', 'install', 'ray[default]', 'py-spy', 'memray'])
# 
# 3. Sync workers with head

    # sync head with workers, head and workers can start in any order
    # workers keep sending a broadcast message and wait for a response from the head node
    # the head node replies to each worker with the ray port to use
    if head:
        sync_with_workers(mcast_address, mcast_port, num_workers, port)
    else:
        port = sync_with_head(mcast_address, mcast_port)
    
    head_address : str = ""
    if worker:
        head_address = mcast_address_receive(mcast_address, mcast_port)

# 4. Run Ray

    # execute ray with the container
    execute_ray : list[str] = [ray_args.container_runner, "exec",  ray_args.container_image, "ray"]

    cmd_line : list[str] = []
    if head:
        cmd_line = execute_ray + ['start', '--head', '--port', str(port) , '--num-gpus', str(num_gpus)]
        if ray_args.dashboard:
            cmd_line += ["--dashboard-host", "0.0.0.0"]
        else:
            cmd_line += ["--include-dashboard=False"]
    else:
        cmd_line = execute_ray + ['start', '--num-gpus', str(num_gpus), '--address', str(head_address)+ ":" + str(port)]

    #print(' '.join(cmd_line))
    out : bytes = bytes()
    try:
        out = sub.check_output(cmd_line)
        print(remove_ansi_escape_chars(out))
        
    except Exception as e:
        abort(str(e))

# 4.1 Extract IP address

    local_ip : str = extract_ip_address(out)
    print(f"IP Address: {local_ip}")
    global ray_ip_address
    ray_ip_address = local_ip

# 5. Broadcast IP address of head process
    
    if head:
        broadcast_ip_address(local_ip, mcast_address, mcast_port)

# 6. Re-sync
    
    # At this point, we know that all the worker processes have started but we do not know if
    # each process has started the Ray process and therefore we need to sync again:
    # we need to exit the program only after we are sure that all Ray's worker and head processes
    # have started.
    # This time the worker and head processeswill send a messagesto after Ray has been started.
    # Instead of write additional code we can reuse what implemented above.

    if head:
        sync_with_workers(mcast_address, mcast_port, num_workers, port)
    else:
        _ = sync_with_head(mcast_address, mcast_port)

# 7. Print IP addresses
    if worker:
        print(f"Head node address: {head_address}")
    else:
        print("Workers: ")
        print("=" * 10)
        for w in workers:
            print(w.decode('utf-8'))
 
# 8. Launch vllm if parameters specified
        
    # if no arguments for vllm or worker do not launch vllm and exit
    if not vllm_args or worker:
        print(f"Worker @{hostname} stopping")
        os.kill(os.getpid(), signal.SIGSTOP)
    sub.call([ray_args.container_runner, 'exec', ray_args.container_image, 'ray', 'status'])
    #Launch vllm on the head node 
    os.environ["VLLM_HOST_IP"] = local_ip
    cwd : str = os.environ["PWD"]
    vllm_cmdline : list[str] = [ray_args.container_runner, "exec",
                                ray_args.container_image, "vllm",
                                "serve", "--distributed-executor-backend", "ray"] + vllm_args
    
    # optionally mount paths
    if ray_args.container_runner in ("singularity", "apptainer"):
        if ray_args.app_dir:
            vllm_cmdline += ['--bind', ray_args.app + ":" + "/app"]
        if ray_args.hf_dir:
            vllm_cmdline += ['--bind', ray_args.hf_dir + ":" + "/root/.cache/huggingface"]
    elif ray_args.container_runner in ('podman', 'docker'):
        if ray_args.app_dir:
            vllm_cmdline += ['-v', ray_args.app + ":" + "/app"]
        if ray_args.hf_dir:
            vllm_cmdline += ['-v', ray_args.hf_dir + ":" + "/root/.cache/huggingface"]
    else:
        print(f"Could not map /app and /huggingface path, unknown container type '{ray_args.container_runner}'")
    
    if head:
        print(' '.join(vllm_cmdline))
        try:
            sub.call(vllm_cmdline)
            print("Started!")
        except Exception as e:
            print("Error running vLLM")
            print(e)
            sys.exit(1)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray
