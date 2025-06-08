#!/usr/bin/env python3

import requests
import sys


def main():
    if len(sys.argv) == 1:
        print(
            f"{sys.argv[0]} <Huggingface model name>\nReturns the numberof hidden layers."
        )
        sys.exit(0)

    url: str = "https://huggingface.co/" + sys.argv[1] + "/resolve/main/config.json"
    try:
        resp = requests.get(url)
        data = resp.json()  # Check the JSON Response Content documentation below
        num_hidden_layers = int(data["num_hidden_layers"])
        print(f"Number of hidden layers: {num_hidden_layers}")
        print(
            "Use a number equal or greater to this one for the -ngl argument of llama-cli"
        )
    except Exception as e:
        print(f"Cannot download file {url}", file=sys.stderr)
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
