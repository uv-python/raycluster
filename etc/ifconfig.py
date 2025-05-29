import subprocess as sub
import re
import sys


def iconfig():
    out = sub.check_output(["ifconfig"])
    nic = re.compile(r"([\w\d]+)\s*Link")
    return re.findall(nic, out.decode("utf-8"))
