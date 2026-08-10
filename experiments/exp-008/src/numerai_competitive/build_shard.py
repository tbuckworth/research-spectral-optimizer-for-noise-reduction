from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_train_shard, build_validation_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--feature-set", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--freeze-manifest", type=Path)
    args = parser.parse_args()
    if args.split == "train":
        if args.freeze_manifest is not None:
            parser.error("--freeze-manifest is only valid for validation")
        result = build_train_shard(args.source, args.destination, args.feature_set)
    else:
        if args.freeze_manifest is None:
            parser.error("validation requires --freeze-manifest")
        result = build_validation_shard(
            args.source, args.destination, args.freeze_manifest, args.feature_set
        )
    print(result)


if __name__ == "__main__":
    main()
