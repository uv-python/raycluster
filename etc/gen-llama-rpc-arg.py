#!/usr/bin/env python3

import sys


# Generate the value for llama-server's '--rpc argument for connecting
# to a remote node over four different IP addresses (or to four different nodes with two GPUs each)
def main():
    if len(sys.argv) != 6:
        print(f"{sys.argv[0]} <port> <ip 1> <ip 2> <ip 3> <ip 4>")
        sys.exit(0)
    base_port: int = int(sys.argv[1])
    ip1: str = sys.argv[2]
    ip2: str = sys.argv[3]
    ip3: str = sys.argv[4]
    ip4: str = sys.argv[5]

    rpc_arg: str = (
        f"{ip1}:{base_port},{ip1}:{base_port + 1},"
        + f"{ip2}:{base_port},{ip2}:{base_port + 1},"
        + f"{ip3}:{base_port},{ip3}:{base_port + 1},"
        + f"{ip4}:{base_port}"
    )
    # rpc_arg: str = (
    #     f"{ip1}:{base_port},{ip1}:{base_port + 1},"
    #     + f"{ip2}:{base_port},{ip2}:{base_port + 1},"
    #     + f"{ip3}:{base_port},{ip3}:{base_port + 1},"
    #     + f"{ip4}:{base_port},{ip4}:{base_port + 1}"
    # )

    print(rpc_arg)


# Invoke as 'gen-llama-rpc-arg.py 6666 10.45.67.4 10.76.34.3 10.46.7.65 10.45.65.3'
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
