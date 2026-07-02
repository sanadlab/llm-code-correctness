"""Build paired / randomly_sampled / weighted mutants_only folds from legacy S1/S2 sources.

Each strategy folder has fit/validate from the S2 (BCB-aligned) limited tree.
Only ``weighted/`` carries an on-disk test split; paired and randomly_sampled
use ``mutants_only/weighted`` test at cache/score time.

Fast path (--merge-from): copy JSONL from existing S*-M-* subfolders.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset.benchmark_config import BENCHMARK_TASK_PREFIX

SPLIT_TO_JSONL = {
    "fit": "fit.jsonl",
    "validate": "validate.jsonl",
    "test": "test.jsonl",
}

STRATEGIES = ("paired", "randomly_sampled", "weighted")
MUTANTS_TEST_STRATEGY = "weighted"

# Legacy folder names used as merge sources.
LEGACY_SOURCES: dict[str, dict[str, str]] = {
    "paired": {
        "fit_val": "S2-M-pair",
    },
    "randomly_sampled": {
        "fit_val": "S2-M-sample",
    },
    "weighted": {
        "fit_val": "S2-M-weighted",
        "test": "S1-M-weighted",
    },
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    fit_val_source: str
    test_source: str | None = None


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def strategy_specs(*, strategies: tuple[str, ...] | None = None) -> list[StrategySpec]:
    chosen = strategies or STRATEGIES
    return [
        StrategySpec(
            name=name,
            fit_val_source=LEGACY_SOURCES[name]["fit_val"],
            test_source=LEGACY_SOURCES[name].get("test"),
        )
        for name in chosen
    ]


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _copy_split(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return _count_jsonl_lines(dst)


def build_strategy(
    spec: StrategySpec,
    *,
    source_root: Path,
    output_root: Path,
    num_folds: int,
    dry_run: bool,
    benchmark: str,
) -> dict[str, Any]:
    fit_val_dir = source_root / spec.fit_val_source
    test_dir = source_root / spec.test_source if spec.test_source else None
    output_dir = output_root / spec.name

    if not fit_val_dir.is_dir():
        raise FileNotFoundError(f"missing source subfolder: {fit_val_dir}")
    if test_dir is not None and not test_dir.is_dir():
        raise FileNotFoundError(f"missing source subfolder: {test_dir}")

    task_counts: dict[str, list[int]] = {"fit": [], "validate": []}
    if test_dir is not None:
        task_counts["test"] = []

    for fold in range(num_folds):
        fold_out = output_dir / f"fold_{fold}"
        if not dry_run:
            fold_out.mkdir(parents=True, exist_ok=True)

        for split in ("fit", "validate"):
            src = fit_val_dir / f"fold_{fold}" / SPLIT_TO_JSONL[split]
            dst = fold_out / SPLIT_TO_JSONL[split]
            if not src.is_file():
                raise FileNotFoundError(f"missing source split: {src}")
            if dry_run:
                task_counts[split].append(_count_jsonl_lines(src))
            else:
                task_counts[split].append(_copy_split(src, dst))

        if test_dir is not None:
            src_test = test_dir / f"fold_{fold}" / SPLIT_TO_JSONL["test"]
            dst_test = fold_out / SPLIT_TO_JSONL["test"]
            if not src_test.is_file():
                raise FileNotFoundError(f"missing source split: {src_test}")
            if dry_run:
                task_counts["test"].append(_count_jsonl_lines(src_test))
            else:
                task_counts["test"].append(_copy_split(src_test, dst_test))

    manifest = {
        "benchmark": benchmark,
        "variant": "mutants_only",
        "strategy": spec.name,
        "fit_val_source": str(fit_val_dir),
        "test_source": str(test_dir) if test_dir is not None else None,
        "test_reference": (
            None if test_dir is not None else f"mutants_only/{MUTANTS_TEST_STRATEGY}"
        ),
        "implementation_limits": {
            "fit": {"canonical": 1, "mutant": 1},
            "validate": {"canonical": 1, "mutant": 1},
            **(
                {"test": {"canonical": 1, "mutant": 3}}
                if test_dir is not None
                else {}
            ),
        },
        "num_folds": num_folds,
        "task_counts": task_counts,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }

    if not dry_run:
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mutants_only paired / randomly_sampled / weighted from legacy S1/S2 folders"
    )
    parser.add_argument(
        "--merge-from",
        type=Path,
        default=Path("dataset/mutants_only"),
        help="Root containing legacy S1-M-* / S2-M-* subfolders (default: dataset/mutants_only)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/mutants_only"),
        help="Destination root (writes strategy subfolders here)",
    )
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=STRATEGIES,
        default=None,
        help="Subset of strategies to build (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_TASK_PREFIX),
        default="bigcodebench",
        help="Benchmark name for manifests",
    )
    args = parser.parse_args()

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    strategies = tuple(args.strategies) if args.strategies else STRATEGIES
    for spec in strategy_specs(strategies=strategies):
        print(f"\n=== {spec.name} ===")
        print(f"  fit/val <= {spec.fit_val_source}")
        if spec.test_source:
            print(f"  test   <= {spec.test_source}")
        else:
            print(f"  test   <= mutants_only/{MUTANTS_TEST_STRATEGY} (shared at score time)")
        manifest = build_strategy(
            spec,
            source_root=args.merge_from,
            output_root=args.output_root,
            num_folds=args.num_folds,
            dry_run=args.dry_run,
            benchmark=args.benchmark,
        )
        for split, counts in manifest["task_counts"].items():
            print(f"  {split}: min={min(counts)} max={max(counts)} counts={counts}")

    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nWrote strategy folders under {args.output_root}")


if __name__ == "__main__":
    main()
