#!/usr/bin/env python3
"""Merge validated_confounded shard files into a single jsonl."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.phase1.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        required=True,
        help="Directory containing validated_confounded.shard_*_of_N.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <intermediate-dir>/validated_confounded.jsonl)",
    )
    args = parser.parse_args()

    shard_re = re.compile(r"validated_confounded\.shard_(\d+)_of_(\d+)\.jsonl$")
    shards: list[tuple[int, int, Path]] = []
    for path in args.intermediate_dir.glob("validated_confounded.shard_*_of_*.jsonl"):
        match = shard_re.match(path.name)
        if not match:
            continue
        shards.append((int(match.group(1)), int(match.group(2)), path))

    if not shards:
        raise SystemExit(f"No shard files found in {args.intermediate_dir}")

    shard_count = shards[0][1]
    if len({s[1] for s in shards}) != 1:
        raise SystemExit("Shard files disagree on shard count")
    if len(shards) != shard_count:
        raise SystemExit(f"Expected {shard_count} shards, found {len(shards)}")

    shards.sort(key=lambda item: item[0])
    merged = []
    for _, _, path in shards:
        merged.extend(read_jsonl(path))

    output = args.output or (args.intermediate_dir / "validated_confounded.jsonl")
    write_jsonl(output, merged)
    print(f"Merged {len(shards)} shards -> {len(merged)} rows -> {output}")


if __name__ == "__main__":
    main()
