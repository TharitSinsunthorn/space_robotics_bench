#!/root/isaac-sim/python.sh

import argparse
import json

from robomimic.config import config_factory


def main(args: argparse.Namespace):
    cfg = config_factory(args.algo)
    with open(f"{args.algo}.json", "w") as f:
        json.dump(cfg, f, indent=4)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="",
        help="Name of the algorithm",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    main(args=args)
