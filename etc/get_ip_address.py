#!/usr/bin/env python3

"""Print list of NICs when invoked with no argument
or the IP address of a specific NIC when passing the
NIC's name on the command line.
"""

import socket
import fcntl
import struct
import sys
import os


SIOCGIFADDR: int = 0x8915


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            SIOCGIFADDR,
            struct.pack("256s", ifname[:15].encode()),
        )[20:24]
    )


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print(
            "Print list of NICs when invoked with no argument "
            "or the IP address of the NIC specified on the command line"
        )
        sys.exit(0)
    if len(sys.argv) == 1:
        print(os.listdir("/sys/class/net/"))
    else:
        try:
            print(get_ip_address(sys.argv[1]))
        except OSError as e:
            print(e, file=sys.stderr)
