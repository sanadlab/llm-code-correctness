"""Build mutants_confounds paired / weighted folds from mutants_only + S1-MC confounds.

For tasks present in ``mutants_only`` that have a confound in ``phase2_v3/S1-MC``:

- **paired** (fit/validate): copy all canonical+mutant lines, then append one
  canonical+confound line per qualifying task.
- **weighted** (fit/validate): copy each canonical+mutant line, then append one
  canonical+confound line per qualifying task (two lines per task).
- **weighted** (test): copy each line; if the task has a confound, randomly
  replace one mutant with that confound (deterministic per fold/task).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset.benchmark_config import BENCHMARK_TASK_PREFIX
from src.dataset.mutant_sampling import rng_for_key, split_impls
from src.phase1.io_utils import write_jsonl

SPLIT_TO_JSONL = {
    "fit": "fit.jsonl",
    "validate": "validate.jsonl",
    "test": "test.jsonl",
}

PAIRED_SPLITS = ("fit", "validate")
WEIGHTED_SPLITS = ("fit", "validate", "test")
DEFAULT_SEED = 0


def _normalize_label(label: Any) -> int:
    """Map S1-MC string labels to icse26-style ints expected by load_split."""
    if isinstance(label, bool):
        raise ValueError(f"unsupported label: {label!r}")
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        if label == "correct":
            return 1
        if label == "incorrect":
            return 0
    raise ValueError(f"unsupported label: {label!r}")


def _normalize_impl(impl: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(impl)
    out["label"] = _normalize_label(out["label"])
    return out


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _extract_canonical_confound(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    canonical = None
    confounds: list[dict[str, Any]] = []
    for impl in record.get("implementations", []):
        kind = impl.get("kind")
        if kind == "canonical" and canonical is None:
            canonical = impl
        elif kind == "confound":
            confounds.append(impl)
    if canonical is None or not confounds:
        return None
    if len(confounds) > 1:
        confounds.sort(key=lambda c: c.get("implementation_id", ""))
    return canonical, confounds[0]


def load_confound_index(mc_root: Path, fold: int) -> dict[str, dict[str, Any]]:
    """Map task_id -> S1-MC record for tasks with a confound (any split in fold)."""
    index: dict[str, dict[str, Any]] = {}
    fold_dir = mc_root / f"fold_{fold}"
    for split in ("fit", "validate", "test"):
        path = fold_dir / SPLIT_TO_JSONL[split]
        if not path.is_file():
            continue
        for record in _read_jsonl(path):
            if _extract_canonical_confound(record) is not None:
                index[record["task_id"]] = record
    return index


def make_confound_pair_record(
    template: dict[str, Any],
    mc_record: dict[str, Any],
) -> dict[str, Any]:
    pair = _extract_canonical_confound(mc_record)
    if pair is None:
        raise ValueError(f"record {mc_record.get('task_id')} has no canonical/confound pair")
    canonical, confound = pair
    out = {k: v for k, v in template.items() if k != "implementations"}
    out["implementations"] = [_normalize_impl(canonical), _normalize_impl(confound)]
    return out


def build_paired_split(
    source_records: list[dict[str, Any]],
    confound_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = [copy.deepcopy(r) for r in source_records]
    seen_tasks: set[str] = set()
    for record in source_records:
        task_id = record["task_id"]
        if task_id in seen_tasks:
            continue
        seen_tasks.add(task_id)
        mc_record = confound_index.get(task_id)
        if mc_record is None:
            continue
        out.append(make_confound_pair_record(record, mc_record))
    return out


def build_weighted_fit_val_split(
    source_records: list[dict[str, Any]],
    confound_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = [copy.deepcopy(r) for r in source_records]
    for record in source_records:
        mc_record = confound_index.get(record["task_id"])
        if mc_record is None:
            continue
        out.append(make_confound_pair_record(record, mc_record))
    return out


def replace_random_mutant_with_confound(
    record: dict[str, Any],
    mc_record: dict[str, Any],
    *,
    seed: int,
    key_parts: tuple[str, ...],
) -> dict[str, Any]:
    pair = _extract_canonical_confound(mc_record)
    if pair is None:
        return copy.deepcopy(record)
    _canonical, confound = pair
    canonical, mutants = split_impls(record)
    if not mutants:
        return copy.deepcopy(record)
    idx = rng_for_key(seed, *key_parts) % len(mutants)
    new_impls = [copy.deepcopy(canonical)]
    for i, mutant in enumerate(mutants):
        if i == idx:
            new_impls.append(_normalize_impl(confound))
        else:
            new_impls.append(copy.deepcopy(mutant))
    out = {k: v for k, v in record.items() if k != "implementations"}
    out["implementations"] = new_impls
    return out


def build_weighted_test_split(
    source_records: list[dict[str, Any]],
    confound_index: dict[str, dict[str, Any]],
    *,
    fold: int,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in source_records:
        task_id = record["task_id"]
        mc_record = confound_index.get(task_id)
        if mc_record is None:
            out.append(copy.deepcopy(record))
            continue
        out.append(
            replace_random_mutant_with_confound(
                record,
                mc_record,
                seed=seed,
                key_parts=("mutants_confounds", "weighted", "test", str(fold), task_id),
            )
        )
    return out


def build_strategy(
    strategy: str,
    *,
    mutants_only_root: Path,
    mc_root: Path,
    output_root: Path,
    num_folds: int,
    seed: int,
    dry_run: bool,
    benchmark: str,
) -> dict[str, Any]:
    source_dir = mutants_only_root / strategy
    output_dir = output_root / strategy
    splits = PAIRED_SPLITS if strategy == "paired" else WEIGHTED_SPLITS

    task_counts: dict[str, list[int]] = {split: [] for split in splits}
    confound_tasks_added: dict[str, list[int]] = {split: [] for split in splits}
    test_replacements: list[int] = []

    for fold in range(num_folds):
        confound_index = load_confound_index(mc_root, fold)
        fold_out = output_dir / f"fold_{fold}"
        if not dry_run:
            fold_out.mkdir(parents=True, exist_ok=True)

        for split in splits:
            src = source_dir / f"fold_{fold}" / SPLIT_TO_JSONL[split]
            if not src.is_file():
                raise FileNotFoundError(f"missing source split: {src}")
            source_records = _read_jsonl(src)

            if strategy == "paired":
                built = build_paired_split(source_records, confound_index)
                added = len(built) - len(source_records)
            elif split == "test":
                built = build_weighted_test_split(
                    source_records, confound_index, fold=fold, seed=seed
                )
                added = sum(
                    1
                    for rec in source_records
                    if rec["task_id"] in confound_index
                )
                test_replacements.append(added)
            else:
                built = build_weighted_fit_val_split(source_records, confound_index)
                added = len(built) - len(source_records)

            task_counts[split].append(len(built))
            confound_tasks_added[split].append(added)

            dst = fold_out / SPLIT_TO_JSONL[split]
            if not dry_run:
                write_jsonl(dst, built)

    manifest: dict[str, Any] = {
        "benchmark": benchmark,
        "variant": "mutants_confounds",
        "strategy": strategy,
        "mutants_only_source": str(source_dir),
        "confound_source": str(mc_root),
        "confound_lookup": "S1-MC task_id index across fit/validate/test per fold",
        "transforms": {
            "paired_fit_validate": "append one canonical+confound line per task with confound",
            "weighted_fit_validate": "append one canonical+confound line per task with confound",
            "weighted_test": "replace one randomly chosen mutant with confound in-place",
        },
        "seed": seed,
        "num_folds": num_folds,
        "task_counts": task_counts,
        "confound_lines_added": confound_tasks_added,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    if strategy == "weighted":
        manifest["test_mutant_replacements"] = test_replacements

    if strategy == "paired":
        manifest["test_reference"] = "mutants_confounds/weighted"
        manifest["implementation_limits"] = {
            "fit": {"canonical": 1, "mutant_or_confound": 1},
            "validate": {"canonical": 1, "mutant_or_confound": 1},
        }
    else:
        manifest["implementation_limits"] = {
            "fit": {"canonical": 1, "mutant_or_confound": 1},
            "validate": {"canonical": 1, "mutant_or_confound": 1},
            "test": {"canonical": 1, "mutant": 2, "confound": 1},
        }

    if not dry_run:
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mutants_confounds paired/weighted from mutants_only + S1-MC confounds"
    )
    parser.add_argument(
        "--mutants-only-root",
        type=Path,
        default=Path("dataset/mutants_only"),
    )
    parser.add_argument(
        "--mc-root",
        type=Path,
        default=Path("phase2_v3/S1-MC"),
        help="S1-MC confound source (default: phase2_v3/S1-MC)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/mutants_confounds"),
    )
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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

    for strategy in ("paired", "weighted"):
        print(f"\n=== {strategy} ===")
        manifest = build_strategy(
            strategy,
            mutants_only_root=args.mutants_only_root,
            mc_root=args.mc_root,
            output_root=args.output_root,
            num_folds=args.num_folds,
            seed=args.seed,
            dry_run=args.dry_run,
            benchmark=args.benchmark,
        )
        for split, counts in manifest["task_counts"].items():
            added = manifest["confound_lines_added"][split]
            print(
                f"  {split}: lines min={min(counts)} max={max(counts)} "
                f"confound_added min={min(added)} max={max(added)}"
            )
        if strategy == "weighted":
            reps = manifest["test_mutant_replacements"]
            print(f"  test replacements: min={min(reps)} max={max(reps)}")

    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nWrote strategy folders under {args.output_root}")


if __name__ == "__main__":
    main()
