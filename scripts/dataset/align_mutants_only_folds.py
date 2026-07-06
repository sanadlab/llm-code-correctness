"""Align mutants_only fold splits to icse26 benchmark partitions.

S1-M: fit/validate/test all use mutant records aligned to icse26/bigcodebench.

S2-M: fit/validate use mutant records aligned to icse26/bigcodebench only.
Test is not materialized here; icse26 bigcodebench/humaneval test is used at
cache/score time (same as OOD).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset.benchmark_config import BENCHMARK_EXPECTED_COUNTS, BENCHMARK_TASK_PREFIX
from src.phase1.io_utils import write_jsonl

SPLITS = ("fit", "validate", "test")
INDEX_SPLITS = SPLITS
SPLIT_TO_JSONL = {
    "fit": "fit.jsonl",
    "validate": "validate.jsonl",
    "test": "test.jsonl",
}
SOURCE_VALIDATE_NAMES = ("validate.jsonl", "val.jsonl")
DEFAULT_SUBFOLDERS = ("S1-M", "S2-M")

# On-disk full-mutant tree may still use legacy S2-M-BCB name.
SOURCE_SUBFOLDER_ALIAS: dict[str, str] = {
    "S2-M": "S2-M-BCB",
}


@dataclass(frozen=True)
class SubfolderConfig:
    mutant_splits: tuple[str, ...]


SUBFOLDER_CONFIG: dict[str, SubfolderConfig] = {
    "S1-M": SubfolderConfig(
        mutant_splits=("fit", "validate", "test"),
    ),
    "S2-M": SubfolderConfig(
        mutant_splits=("fit", "validate"),
    ),
}


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


def load_reference_partition(reference_root: Path, num_folds: int) -> dict[int, dict[str, list[str]]]:
    """Return fold -> split -> ordered task_id list from reference benchmark."""
    partition: dict[int, dict[str, list[str]]] = {}
    for fold in range(num_folds):
        partition[fold] = {}
        for split in SPLITS:
            path = reference_root / f"fold_{fold}" / SPLIT_TO_JSONL[split]
            if not path.exists():
                raise FileNotFoundError(f"missing reference split: {path}")
            task_ids: list[str] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        task_ids.append(json.loads(line)["task_id"])
            partition[fold][split] = task_ids
    return partition


def load_reference_records(reference_root: Path, num_folds: int, split: str) -> dict[int, list[dict[str, Any]]]:
    """Return fold -> ordered records for one split from a reference benchmark."""
    records_by_fold: dict[int, list[dict[str, Any]]] = {}
    for fold in range(num_folds):
        path = reference_root / f"fold_{fold}" / SPLIT_TO_JSONL[split]
        if not path.exists():
            raise FileNotFoundError(f"missing reference split: {path}")
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records_by_fold[fold] = records
    return records_by_fold


def _source_validate_path(fold_dir: Path) -> Path | None:
    for name in SOURCE_VALIDATE_NAMES:
        path = fold_dir / name
        if path.exists():
            return path
    return None


def _iter_source_files(
    subfolder_root: Path,
    num_folds: int,
    splits: tuple[str, ...],
) -> list[tuple[int, str, Path]]:
    files: list[tuple[int, str, Path]] = []
    for fold in range(num_folds):
        fold_dir = subfolder_root / f"fold_{fold}"
        for split in splits:
            if split == "validate":
                path = _source_validate_path(fold_dir)
                if path is None:
                    continue
            else:
                path = fold_dir / SPLIT_TO_JSONL[split]
            if path.exists():
                files.append((fold, split, path))
    return files


def build_mutant_index(
    subfolder_root: Path,
    num_folds: int,
    splits: tuple[str, ...],
    *,
    task_prefix: str,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Index mutant records by task_id and collect seen task ids."""
    index: dict[str, dict[str, Any]] = {}
    seen_task_ids: set[str] = set()

    for _fold, _split, path in _iter_source_files(subfolder_root, num_folds, splits):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task_id = record["task_id"]
                seen_task_ids.add(task_id)
                if not task_id.startswith(task_prefix):
                    continue
                if task_id in index:
                    existing = index[task_id]
                    existing_impls = {
                        impl["implementation_id"] for impl in existing.get("implementations", [])
                    }
                    new_impls = {
                        impl["implementation_id"] for impl in record.get("implementations", [])
                    }
                    if existing_impls != new_impls:
                        raise ValueError(
                            f"inconsistent mutant implementations for {task_id} in {subfolder_root}"
                        )
                    continue
                index[task_id] = record

    return index, seen_task_ids


def align_subfolder(
    *,
    subfolder: str,
    source_root: Path,
    output_root: Path,
    reference_root: Path,
    partition: dict[int, dict[str, list[str]]],
    num_folds: int,
    dry_run: bool,
    benchmark: str,
    mutant_subfolder: str | None,
    task_prefix: str,
) -> dict[str, Any]:
    config = SUBFOLDER_CONFIG.get(subfolder)
    if config is None:
        raise ValueError(f"unknown subfolder: {subfolder}")

    if mutant_subfolder is not None:
        source_subfolder = mutant_subfolder
    else:
        source_subfolder = SOURCE_SUBFOLDER_ALIAS.get(subfolder, subfolder)
    source_dir = source_root / source_subfolder
    output_dir = output_root / subfolder
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing subfolder: {source_dir}")

    mutant_index, seen_task_ids = build_mutant_index(
        source_dir,
        num_folds,
        INDEX_SPLITS,
        task_prefix=task_prefix,
    )

    mutant_reference_ids: set[str] = set()
    for fold in partition:
        for split in config.mutant_splits:
            mutant_reference_ids.update(partition[fold][split])

    missing_mutant_tasks = sorted(tid for tid in mutant_reference_ids if tid not in seen_task_ids)
    available_mutant_ids = mutant_reference_ids - set(missing_mutant_tasks)

    task_counts: dict[str, list[int]] = {split: [] for split in SPLITS}
    missing_by_fold_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    for fold in range(num_folds):
        fold_out = output_dir / f"fold_{fold}"
        if not dry_run:
            fold_out.mkdir(parents=True, exist_ok=True)

        for split in config.mutant_splits:
            ref_task_ids = partition[fold][split]
            records: list[dict[str, Any]] = []
            missing_here: list[str] = []
            for task_id in ref_task_ids:
                record = mutant_index.get(task_id)
                if record is None:
                    missing_here.append(task_id)
                    continue
                records.append(record)

            task_counts[split].append(len(records))
            if missing_here:
                missing_by_fold_split[split].append(
                    {"fold": fold, "task_ids": missing_here}
                )

            out_path = fold_out / SPLIT_TO_JSONL[split]
            if not dry_run:
                write_jsonl(out_path, records)

        if not dry_run:
            val_path = fold_out / "val.jsonl"
            if val_path.exists():
                val_path.unlink()

    expected = BENCHMARK_EXPECTED_COUNTS[benchmark]
    expected_test_counts = [expected["test"]] * num_folds if subfolder == "S1-M" else None

    manifest = {
        "benchmark": benchmark,
        "variant": "mutants_only",
        "subfolder": subfolder,
        "mutant_source": str(source_dir),
        "mutant_reference": str(reference_root),
        "test_reference": "runtime" if subfolder == "S2-M" else None,
        "num_folds": num_folds,
        "missing_mutant_tasks": missing_mutant_tasks,
        "task_counts": task_counts,
        "expected_task_counts": {
            "fit": [expected["fit"]] * num_folds,
            "validate": [expected["validate"]] * num_folds,
            **(
                {"test": expected_test_counts}
                if expected_test_counts is not None
                else {}
            ),
        },
        "missing_by_fold_split": missing_by_fold_split,
        "aligned_mutant_tasks": len(available_mutant_ids),
        "mutant_reference_unique_tasks": len(mutant_reference_ids),
        "restructured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }

    if not dry_run:
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align mutants_only folds to icse26 benchmark partitions"
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("dataset/icse26/bigcodebench"),
        help="Reference for mutant fit/validate task partitions",
    )
    parser.add_argument(
        "--mutants-root",
        type=Path,
        default=Path("dataset/mutants_only"),
        help="Source mutants_only root (read from here when output-root omitted)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root (default: overwrite --mutants-root in place)",
    )
    parser.add_argument(
        "--subfolders",
        nargs="+",
        default=list(DEFAULT_SUBFOLDERS),
        help="Subfolders to align (default: S1-M S2-M)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="Number of folds (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print summary without writing files",
    )
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_TASK_PREFIX),
        default="bigcodebench",
        help="Benchmark name for manifests and task-id filtering",
    )
    parser.add_argument(
        "--mutant-subfolder",
        type=str,
        default=None,
        help="On-disk mutant tree name (e.g. S1-M-HE); both S1-M and S2-M read from here",
    )
    args = parser.parse_args()

    output_root = args.output_root if args.output_root is not None else args.mutants_root
    source_root = args.mutants_root
    task_prefix = BENCHMARK_TASK_PREFIX[args.benchmark]

    if output_root.resolve() != source_root.resolve() and not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    partition = load_reference_partition(args.reference_root, args.num_folds)

    for subfolder in args.subfolders:
        print(f"\n=== {subfolder} ===")
        manifest = align_subfolder(
            subfolder=subfolder,
            source_root=source_root,
            output_root=output_root,
            reference_root=args.reference_root,
            partition=partition,
            num_folds=args.num_folds,
            dry_run=args.dry_run,
            benchmark=args.benchmark,
            mutant_subfolder=args.mutant_subfolder,
            task_prefix=task_prefix,
        )
        print(f"  test_reference: {manifest['test_reference']}")
        print(f"  missing_mutant_tasks: {len(manifest['missing_mutant_tasks'])}")
        if manifest["missing_mutant_tasks"]:
            print(f"    {manifest['missing_mutant_tasks']}")
        print(
            "  aligned_mutant_tasks: "
            f"{manifest['aligned_mutant_tasks']} / {manifest['mutant_reference_unique_tasks']}"
        )
        for split in SPLITS:
            if split not in manifest["task_counts"] or not manifest["task_counts"][split]:
                continue
            counts = manifest["task_counts"][split]
            print(f"  {split}: min={min(counts)} max={max(counts)} per-fold counts={counts}")

    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nWrote aligned folds under {output_root}")


if __name__ == "__main__":
    main()
