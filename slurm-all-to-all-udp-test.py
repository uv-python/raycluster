#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Author: Ugo Varetto
# Test all to all connectivity of nodes in a SLURM cluster.
import os
import socket
import selectors
import re
import struct
import time
import sys


MCAST_GROUP = "224.0.0.100"  # non routable, local subnet only
PORT = 6379
SELECT_TIMEOUT = 0.5  # seconds, select
TTL = 2  # max number of hops
TIMEOUT = 60.0  # seconds


# retrieve the list of slurm nodes
def slurm_node_list() -> list[str]:
    r = re.compile(r"nid\[([^\]]+)\]")
    nodes = r.search(os.environ["SLURM_JOB_NODELIST"])
    if nodes is None:
        return []
    return list(map(lambda n: "nid" + n, nodes.group(1).split(",")))


# Broadcast host name and receive host name from all the other
# nodes, writing to file the list of received nodes.
# The received node names are added to a set and removed each time
# one is received until the set is emotyu.
def main():
    ttl: int = TTL
    port: int = PORT
    mcast_group = MCAST_GROUP
    nodes = slurm_node_list()
    hostname = socket.gethostname()
    nodeset = set(
        map(lambda x: bytes(x, "utf-8"), filter(lambda n: n != hostname, nodes))
    )
    sock_mcast: socket.socket = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
    )
    sock_mcast.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    sock_mcast.bind((mcast_group, port))
    mreq: bytes = struct.pack("4sl", socket.inet_aton(mcast_group), socket.INADDR_ANY)
    sock_mcast.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sel: selectors.DefaultSelector = selectors.DefaultSelector()
    sel.register(sock_mcast, selectors.EVENT_READ)
    start: float = time.perf_counter()
    with open(hostname, "w") as f:
        while len(nodeset):
            sock_mcast.sendto(hostname.encode(), (mcast_group, port))
            events = sel.select(timeout=SELECT_TIMEOUT)
            if not events:
                elapsed = time.perf_counter() - start
                if elapsed > TIMEOUT:
                    f.write(f"Timed out after {elapsed} seconds")
                    sys.exit(0)
                continue
            n, _ = sock_mcast.recvfrom(128)
            if n in nodeset:
                f.write(n.decode("utf-8") + "\n")
                nodeset.remove(n)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

# Multicast groups                                             Routable
# 224.0.0.0   to 224.0.0.255        Local subnetwork]             No
# 224.0.1.0   to 224.0.1.255        Internetwork control         Yes
# 224.0.2.0   to 224.0.255.255      AD-HOC block 1               Yes
# 224.1.0.0   to 224.1.255.255      Reserved
# 224.2.0.0   to 224.2.255.255      SDP/SAP block                Yes
# 224.3.0.0   to 224.4.255.255      AD-HOC block 2               Yes
# 224.5.0.0   to 224.255.255.255    Reserved
# 225.0.0.0   to 231.255.255.255    Reserved
# 232.0.0.0   to 232.255.255.255    Source-specific multicast    Yes
# 233.0.0.0   to 233.251.255.255    GLOP addressing              Yes
# 233.252.0.0 to 233.255.255.255    AD-HOC block 3               Yes
# 234.0.0.0   to 234.255.255.255    Unicast-prefix-based         Yes
# 235.0.0.0   to 238.255.255.255    Reserved
# 239.0.0.0   to 239.255.255.255    Administratively scoped      Yes
