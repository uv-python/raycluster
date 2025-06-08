#!/usr/bin/env python3

import requests
import sys


def main():
    if len(sys.argv) == 1:
        print(f"{sys.argv[0]} <Huggingface model name> [number of gpus on node]")
        sys.exit(0)

    url: str = "https://huggingface.co/" + sys.argv[1] + "/resolve/main/config.json"
    attention_heads: int = 0
    try:
        resp = requests.get(url)
        data = resp.json()  # Check the JSON Response Content documentation below
        attention_heads = int(data["num_attention_heads"])
        print(f"Number of attention heads: {attention_heads}")
        print(
            "The number of attention heads must be divisible by the "
            "number of parallel tensors"
        )
    except Exception as e:
        print(f"Cannot download file {url}", file=sys.stderr)
        print(e)
        sys.exit(1)

    if len(sys.argv) == 3:
        try:
            num_gpus = int(sys.argv[2])
        except ValueError:
            print("Invalid number of gpus", sys.stderr)
            sys.exit(1)

        while attention_heads % num_gpus:
            num_gpus -= 1
        print(f"Set number of GPUs to {num_gpus} == number of parallel tensors")


if __name__ == "__main__":
    main()
