# list nid* files
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

def check(cm: dict[str, list[str]], node_set: set[str]) -> None:
    for k, v in cm:
        if len(v) != len(node_set):
            
        #use disjoint sets

def main():
    nids: set[str] = set(glob.glob("nid*"))
    num_nids : int = len(nids)
    conn_map: dict[str, set[str]] = dict()
    for i in nids:
        with open(i, "r") as f:
            rnids = set(f.read().split())
            conn_map.update({i: rnids})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
