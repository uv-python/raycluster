#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Author: Ugo Varetto

# Read node list from each nid* file containing the list of nodes they connect
# to and check if all the nodes are connected, printing the list of missing
# links if they are not.


# list nid* files containing the list of nodes the node connects to
# sort nids file list
# create map nid file -> connected node list
# for each nid file:
# read node list
# sort node list
# add list to current nid key in map

# for each nid in map check that it's connected to all the others,
# if not print the missing node names
import glob
import sys


def check(cm: dict[str, set[str]], node_set: set[str]) -> dict[str, list[str]]:
    m: dict[str, list[str]] = dict()
    for k, v in cm.items():
        if len(v) != (len(node_set) - 1):
            diff = node_set.difference(v)
            diff.discard(k)
            m.update({k: sorted(list(diff))})
    return m


def test_check():
    nn: dict[str, set[str]] = {
        "nid002220": {"nid002221", "nid002222", "nid002223"},
        "nid002221": {"nid002220"},
        "nid002222": {"nid002221", "nid002223"},
        "nid002223": {"nid002220", "nid002221"},
    }
    nodes: set[str] = set(list(nn.keys()))
    c = check(nn, nodes)
    assert c["nid002221"] == ["nid002222", "nid002223"]
    assert c["nid002222"] == ["nid002220"]
    assert c["nid002223"] == ["nid002222"]


def main() -> int:
    if sys.argv[1] == "test":
        test_check()
        return 0

    nids: set[str] = set(glob.glob("nid*"))
    conn_map: dict[str, set[str]] = dict()
    for i in nids:
        with open(i, "r") as f:
            rnids = set(f.read().split())
            conn_map.update({i: rnids})

    diff: dict[str, list[str]] = check(conn_map, nids)
    if diff:
        print("Missing links:")
        for k, v in diff.items():
            print(f"{k}: {v}")
    else:
        print("All nodes connected")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        raise e
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
