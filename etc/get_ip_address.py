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
import re


SIOCGIFADDR: int = 0x8915


def get_ip_address(ifname: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        return socket.inet_ntoa(
            fcntl.ioctl(
                s.fileno(),
                SIOCGIFADDR,
                struct.pack("256s", ifname[:15].encode()),
            )[20:24]
        )
    except:
        return ""


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print(
            "Print list of NICs when invoked with no argument "
            "or the IP address of the NIC specified on the command line.\n"
            "It possible to return the IP address of all adapters matching "
            "a specific pattern by specifying --re <pattern e.g. 'hsn'>.\n"
            "Use '--re *' to select all adapters."
        )
        sys.exit(0)
    if len(sys.argv) == 1:
        print(os.listdir("/sys/class/net/"))
    else:
        try:
            if sys.argv[1] == "--re":
                r = re.compile(sys.argv[2]) if sys.argv[2] != "*" else None
                for i in os.listdir("/sys/class/net/"):
                    if sys.argv[2] == "*" or r.match(i):
                        ip = get_ip_address(i)
                        if ip:
                            print(f"{i}:\t{ip}")
            else:
                print(get_ip_address(sys.argv[1]))
        except OSError as e:
            print(e, file=sys.stderr)
