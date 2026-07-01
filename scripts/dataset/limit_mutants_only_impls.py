"""Limit mutants_only implementation counts to match icse26-style density.

DEPRECATED for ablation: use scripts/dataset/build_mutants_only.py to produce
paired/ and randomly_sampled/ from legacy S1/S2 limited folders.

Reads full-mutant data from mutants_only_full and writes legacy subfolders
(S1-M-*, S2-M-*) using sample, pair, or weighted strategies for fit/validate.
S1 test is limited to 3 mutants; S2 has no on-disk test (icse26 test at cache/score time).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset.benchmark_config import BENCHMARK_TASK_PREFIX
from src.dataset.mutant_impl_normalize import normalize_task_record
from src.dataset.mutant_sampling import (
    make_record as _make_record,
    split_impls as _split_impls,
    subsample_record_uniform,
    subsample_record_weighted,
)
from src.dataset.operator_weights import (
    count_operators,
    operator_frequencies,
)
from src.phase1.io_utils import write_jsonl

SPLITS = ("fit", "validate", "test")
SPLIT_TO_JSONL = {
    "fit": "fit.jsonl",
    "validate": "validate.jsonl",
    "test": "test.jsonl",
}

BASE_SUBFOLDERS = ("S1-M", "S2-M")
STRATEGIES = ("sample", "pair", "weighted")

# Legacy full-mutant tree uses S2-M-BCB for BCB-aligned fit/validate (bigcodebench only).
SOURCE_BASE_ALIAS_BY_BENCHMARK: dict[str, dict[str, str]] = {
    "bigcodebench": {"S2-M": "S2-M-BCB"},
    "humaneval": {},
}

FIT_VAL_MUTANTS = 1
S1_TEST_MUTANTS = 3


@dataclass(frozen=True)
class OutputSpec:
    name: str
    base: str
    strategy: str
    is_s1: bool


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


def output_specs(*, strategies: tuple[str, ...] | None = None) -> list[OutputSpec]:
    chosen = strategies or STRATEGIES
    specs: list[OutputSpec] = []
    for base in BASE_SUBFOLDERS:
        for strategy in chosen:
            if strategy not in STRATEGIES:
                raise ValueError(f"unknown strategy {strategy!r}")
            name = f"{base}-{strategy}"
            specs.append(
                OutputSpec(
                    name=name,
                    base=base,
                    strategy=strategy,
                    is_s1=base == "S1-M",
                )
            )
    return specs


def _source_base_dir(source_root: Path, base: str, benchmark: str) -> Path:
    alias = SOURCE_BASE_ALIAS_BY_BENCHMARK.get(benchmark, {}).get(base, base)
    return source_root / alias


def _rng_for_key(seed: int, *parts: str) -> int:
    from src.dataset.mutant_sampling import rng_for_key

    return rng_for_key(seed, *parts)


def subsample_record(
    record: dict[str, Any],
    n_mutants: int,
    seed: int,
    key_parts: tuple[str, ...],
) -> dict[str, Any] | None:
    return subsample_record_uniform(record, n_mutants, seed, key_parts)


def _op_freq_for_records(records: list[dict[str, Any]]) -> dict[str, float]:
    return operator_frequencies(count_operators(records))


def expand_pairs(record: dict[str, Any]) -> list[dict[str, Any]]:
    canonical, mutants = _split_impls(record)
    if not mutants:
        return []
    mutants_sorted = sorted(mutants, key=lambda m: m.get("implementation_id", ""))
    return [_make_record(record, canonical, [mutant]) for mutant in mutants_sorted]


def process_mutant_records(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    split: str,
    is_s1: bool,
    output_name: str,
    fold: int,
    seed: int,
    op_freq: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    skipped: list[str] = []

    for record in records:
        task_id = record["task_id"]
        key_parts = (output_name, str(fold), split, task_id)
        if split in ("fit", "validate"):
            if strategy == "sample":
                transformed = subsample_record(
                    record,
                    FIT_VAL_MUTANTS,
                    seed,
                    key_parts,
                )
                if transformed is None:
                    skipped.append(task_id)
                    warnings.warn(
                        f"[{output_name}] fold_{fold} {split}: skipped {task_id} "
                        f"(need {FIT_VAL_MUTANTS} mutant(s))",
                        stacklevel=2,
                    )
                    continue
                out.append(transformed)
            elif strategy == "weighted":
                if op_freq is None:
                    raise ValueError("op_freq required for weighted strategy")
                transformed = subsample_record_weighted(
                    record,
                    FIT_VAL_MUTANTS,
                    op_freq,
                    seed,
                    key_parts,
                )
                if transformed is None:
                    skipped.append(task_id)
                    warnings.warn(
                        f"[{output_name}] fold_{fold} {split}: skipped {task_id} "
                        f"(need {FIT_VAL_MUTANTS} weighted mutant(s))",
                        stacklevel=2,
                    )
                    continue
                out.append(transformed)
            else:
                pairs = expand_pairs(record)
                if not pairs:
                    skipped.append(task_id)
                    warnings.warn(
                        f"[{output_name}] fold_{fold} {split}: skipped {task_id} (no mutants)",
                        stacklevel=2,
                    )
                    continue
                out.extend(pairs)
        elif is_s1 and split == "test":
            if strategy == "weighted":
                if op_freq is None:
                    raise ValueError("op_freq required for weighted strategy")
                transformed = subsample_record_weighted(
                    record,
                    S1_TEST_MUTANTS,
                    op_freq,
                    seed,
                    key_parts,
                )
            else:
                transformed = subsample_record(
                    record,
                    S1_TEST_MUTANTS,
                    seed,
                    key_parts,
                )
            if transformed is None:
                skipped.append(task_id)
                warnings.warn(
                    f"[{output_name}] fold_{fold} {split}: skipped {task_id} "
                    f"(need {S1_TEST_MUTANTS} mutant(s))",
                    stacklevel=2,
                )
                continue
            out.append(transformed)
        else:
            raise ValueError(f"unexpected mutant record for split={split!r} is_s1={is_s1}")

    return out, skipped


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(normalize_task_record(json.loads(line)))
    return records


def process_subfolder(
    spec: OutputSpec,
    *,
    source_root: Path,
    output_root: Path,
    num_folds: int,
    seed: int,
    dry_run: bool,
    benchmark: str,
) -> dict[str, Any]:
    source_dir = _source_base_dir(source_root, spec.base, benchmark)
    output_dir = output_root / spec.name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing source subfolder: {source_dir}")

    task_counts: dict[str, list[int]] = {"fit": [], "validate": []}
    if spec.is_s1:
        task_counts["test"] = []
    skipped_by_fold_split: dict[str, list[dict[str, Any]]] = {
        "fit": [],
        "validate": [],
    }
    if spec.is_s1:
        skipped_by_fold_split["test"] = []

    operator_weights_by_fold_split: dict[int, dict[str, dict[str, float]]] = {}

    for fold in range(num_folds):
        fold_out = output_dir / f"fold_{fold}"
        if not dry_run:
            fold_out.mkdir(parents=True, exist_ok=True)

        for split in ("fit", "validate"):
            src_path = source_dir / f"fold_{fold}" / SPLIT_TO_JSONL[split]
            records = load_jsonl_records(src_path)
            op_freq = None
            if spec.strategy == "weighted":
                op_freq = _op_freq_for_records(records)
                operator_weights_by_fold_split.setdefault(fold, {})[split] = op_freq
            transformed, skipped = process_mutant_records(
                records,
                strategy=spec.strategy,
                split=split,
                is_s1=spec.is_s1,
                output_name=spec.name,
                fold=fold,
                seed=seed,
                op_freq=op_freq,
            )
            task_counts[split].append(len(transformed))
            if skipped:
                skipped_by_fold_split[split].append({"fold": fold, "task_ids": skipped})
            if not dry_run:
                write_jsonl(fold_out / SPLIT_TO_JSONL[split], transformed)

        if spec.is_s1:
            src_path = source_dir / f"fold_{fold}" / SPLIT_TO_JSONL["test"]
            records = load_jsonl_records(src_path)
            op_freq = None
            if spec.strategy == "weighted":
                op_freq = _op_freq_for_records(records)
                operator_weights_by_fold_split.setdefault(fold, {})["test"] = op_freq
            transformed, skipped = process_mutant_records(
                records,
                strategy=spec.strategy,
                split="test",
                is_s1=True,
                output_name=spec.name,
                fold=fold,
                seed=seed,
                op_freq=op_freq,
            )
            task_counts["test"].append(len(transformed))
            if skipped:
                skipped_by_fold_split["test"].append({"fold": fold, "task_ids": skipped})
            if not dry_run:
                write_jsonl(fold_out / SPLIT_TO_JSONL["test"], transformed)

    manifest = {
        "benchmark": benchmark,
        "variant": "mutants_only",
        "subfolder": spec.name,
        "base_subfolder": spec.base,
        "strategy": spec.strategy,
        "source": str(source_dir),
        "seed": seed,
        "implementation_limits": {
            "fit": {"canonical": 1, "mutant": FIT_VAL_MUTANTS},
            "validate": {"canonical": 1, "mutant": FIT_VAL_MUTANTS},
            "test": (
                {"canonical": 1, "mutant": S1_TEST_MUTANTS}
                if spec.is_s1
                else "runtime_icse26"
            ),
        },
        "test_reference": None if spec.is_s1 else "runtime",
        "num_folds": num_folds,
        "task_counts": task_counts,
        "skipped_by_fold_split": skipped_by_fold_split,
        "restructured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    if spec.strategy == "weighted":
        manifest["operator_weights_by_fold_split"] = {
            str(fold): {
                split: {code: round(freq, 6) for code, freq in freqs.items()}
                for split, freqs in splits.items()
            }
            for fold, splits in operator_weights_by_fold_split.items()
        }

    if not dry_run:
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limit mutants_only implementation counts (2/2/4)"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset/mutants_only_full"),
        help="Full-mutant source root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/mutants_only"),
        help="Destination root for limited subfolders",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
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
    for spec in output_specs(strategies=strategies):
        print(f"\n=== {spec.name} ({spec.strategy}) ===")
        manifest = process_subfolder(
            spec,
            source_root=args.source_root,
            output_root=args.output_root,
            num_folds=args.num_folds,
            seed=args.seed,
            dry_run=args.dry_run,
            benchmark=args.benchmark,
        )
        for split, counts in manifest["task_counts"].items():
            print(f"  {split}: min={min(counts)} max={max(counts)} counts={counts}")
        total_skipped = sum(
            len(entry["task_ids"])
            for entries in manifest["skipped_by_fold_split"].values()
            for entry in entries
        )
        print(f"  skipped_tasks: {total_skipped}")

    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nWrote limited subfolders under {args.output_root}")


if __name__ == "__main__":
    main()
