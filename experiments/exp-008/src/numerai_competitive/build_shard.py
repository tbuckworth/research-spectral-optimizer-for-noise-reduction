from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_train_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--feature-set", choices=["small", "medium", "all"], default="all")
    args = parser.parse_args()
    print(build_train_shard(args.source, args.destination, args.feature_set))


if __name__ == "__main__":
    main()
