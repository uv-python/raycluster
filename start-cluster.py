#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Author: Ugo Varetto
# TODO: add error checking and async receive with timeout, 
#       do not throw exceptions, add mapping of /app and /huggingfgace folders

import os
import subprocess as sub
import argparse
import re
import socket
import struct
import sys
import ipaddress
import signal


output_log : list[str] = []
workers : set[bytes] = set()

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

def sync_with_head(mcast_group: str, port: int, ttl: int = 3 ) -> int:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        msg=socket.gethostbyname(socket.getfqdn());

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

def broadcast_ip_address(ip: str, mcast_group: str, port: int, ttl: int = 3):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        sock.sendto(ip.encode(), (mcast_group, port))
    except:
        abort("Error broadcasting messages")

def remove_ansi_escape_chars(buffer) -> str:
    ansi_escape = re.compile(r'\x1b[^m]+m')
    t : str = "" 
    try:
        t = ansi_escape.sub('', buffer.decode('utf-8'))
        return t
    except:
        return ""

def extract_ip_address(buffer: bytes) -> str:
    ansi_escape = re.compile(r'\x1b[^m]+m')
    t : list[str] = []
    try:
        t = ansi_escape.sub('', buffer.decode('utf-8')).split()
    except:
        return ""
    return t[t.index('IP:') + 1]
#
#-------------------------------------------------------------------------------
#
def signal_handler(_, __) -> None:
    print(output_log)
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    #check if ray alreay active:
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("ROCR_VISIBLE_DEVICES")
    try:
        _ = sub.check_output(['ray', 'status'], stderr=sub.STDOUT)
        print("Ray already running, exiting...")
        sys.exit(1)
    except:
        pass
    # Parse arguments
    parser = argparse.ArgumentParser(prog="start-cluster", description="Run Ray and vllM container, workers and head nodes " \
                                                                  "can be started independently and the port needs only be " \
                                                                  "specified for the head node. The container is used to start both " \
                                                                  "Ray and vLLM; if no extrea command line parameters beyond the ones" \
                                                                  "required for ray are specified, vLLM is not run.\n" \
                                                                  "The command line parameters are passed to 'vllm serve'")
    parser.add_argument("container_runner", help="Container runner, Singularity, Apptainer, Podman...")
    parser.add_argument("container_image", help="Path to container must contain Ray and if additinal ")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs")
    parser.add_argument("--head", const=True, nargs='?', help="Head node")
    parser.add_argument("--worker", const=True, nargs='?', help="Worker node")
    parser.add_argument("--port", type=int, help="Ray TCP port, for head only, workers will receive port from head")
    parser.add_argument("--head-address", help="IP address of the head node for workers" \
                                          "if not specified UDP multicast will be used ")
    parser.add_argument("--mcast-address", help="Multicast address, default is 224.0.0.100")
    parser.add_argument("--mcast-port", help="Multicast port, default is 5001")
    parser.add_argument("--broadcast", const=True, nargs='?', help="Have head node broadcast IP address through IP multicast")
    parser.add_argument("--num-workers", type=int, help="Used by head node to wait until all workers are active")
    parser.add_argument("--app-dir", help="Local directory mapping /app")
    parser.add_argument("--hf-dir", help="Local mapping of Huggingface /root/.cache/hugingface directory")
                    
    ray_args, vllm_args = parser.parse_known_args() # known, unknown
    #'unknown' are the parameters after `vllm serve'`
    
    port : int = ray_args.port or 6379
    if not valid_port(port):
        print("Invalid port")
        sys.exit(1)

    head : bool = ray_args.head or False
    worker : bool = ray_args.worker or False
    num_gpus : int = ray_args.num_gpus or 0
    if head and not ray_args.num_workers:
        abort("When --head specified --num-workers is required because the head node needs to know " \
              "how many workers it needs to wait for")
    if not head and not worker:
        abort("Either --head or --worker needs to be specified")
    if head and worker:
        abort("Only one of --head or --worked must be specified")
    mcast_address : str = ray_args.mcast_address or "224.0.0.100" # non routable
    mcast_port : int = ray_args.mcast_port or 5001
    if not valid_ip_address(mcast_address):
        abort("Invalid multicast ip address")
        
    if not valid_port(mcast_port):
        abort("Invalid multicast port")

    # sync head with nodes, head and workers can start in any order
    # workers keep sending a broadcast message and wait for a response from the head node
    # the head node replies to each worker with the ray port to use
    if head:
        sync_with_workers(mcast_address, mcast_port, ray_args.num_workers, port)
    else:
        port = sync_with_head(mcast_address, mcast_port)
    
    head_address : str = ""
    if worker and not ray_args.head_address:
        head_address = mcast_address_receive(mcast_address, mcast_port)
  
    # execute ray with the container
    execute_ray : list[str] = [ray_args.container_runner, "exec",  ray_args.container_image, "ray"]

    cmd_line : list[str] = []
    if head:
        cmd_line = execute_ray + ['start', '--head', '--port', str(port) , '--num-gpus', str(num_gpus), "--dashboard-host", "0.0.0.0"]
    else:
        cmd_line = execute_ray + ['start', '--num-gpus', str(num_gpus), '--address', str(head_address)+ ":" + str(port)]

    #print(' '.join(cmd_line))
    out : bytes = bytes()
    try:
        out = sub.run(cmd_line, capture_output=True).stdout#sub.check_output(cmd_line)
        output_log.append(remove_ansi_escape_chars(out))
    except Exception as e:
        abort(str(e))

    local_ip : str = extract_ip_address(out)
    print(f"IP Address: {local_ip}")
    if head and ray_args.broadcast:
        broadcast_ip_address(local_ip, mcast_address, mcast_port)

    if worker:
        print(f"Head node address: {head_address}")
    else:
        print("Workers: ")
        print("=" * 10)
        for w in workers:
            print(w.decode('utf-8'))
    
    # if not arguments for vllm, do not launch vllm and exit
    if not vllm_args:
        sys.exit(0)
    #Launch vllm on the head node 
    os.environ["VLLM_HOST_IP"] = local_ip
    cwd : str = os.environ["PWD"]
    # vllm_cmdline : list[str] = [ray_args.container_runner, "exec", "-H", cwd, # is -H needed?
    #                             ray_args.container_image, "vllm", "serve"] + vllm_args
    vllm_cmdline : list[str] = [ray_args.container_runner, "exec",
                                ray_args.container_image, "vllm", "serve"] + vllm_args
    if ray_args.container_runner in ["singularity", "apptainer"]:
        if ray_args.app:
            vllm_cmdline += ['--bind', ray_args.app + ":" + "/app"]
        if ray_args.hf_dir:
            vllm_cmdline += ['--bind', ray_args.hf_dir + ":" + "/root/.cache/huggingface"]
    else:
        print("WARNING /app and /huggingface mapping only supported for Singularity and Apptainer")
    
    if head:
        print(' '.join(vllm_cmdline))
        try:
            out = sub.check_output(vllm_cmdline)
            print("Started!")
        except Exception as e:
            print("Error running vLLM")
            print(e)
            sys.exit(1)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray
