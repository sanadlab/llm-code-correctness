"""Aggregate `fit_and_score_from_cache.py` results into a mean/std summary CSV.

Walks a results directory for `results.json` files (one per fold, or per
outer/inner cell for OOD fit sources) and groups them by every ablation axis
(model, fit_source, test_dataset, method, representation, fit_framing,
val_framing, test_framing), computing the mean and population std of
`test_one_of_4` and `test_pairwise` within each group.

Each cell is expected to have sibling `metadata.json` (new format) or to
embed grouping fields directly in `results.json` (legacy minimal format).

Usage:
    python scripts/analysis/aggregate_ablation_results.py \\
        --results-dir results --output analysis/ablation/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GROUP_FIELDS = (
    "model", "fit_source", "test_dataset", "method",
    "representation", "fit_framing", "val_framing", "test_framing",
)

FIT_SCORE_SCRIPT = "scripts/experiments/fit_and_score_from_cache.py"


def _fit_source_from_metadata(meta: dict) -> str:
    if meta.get("split_family") == "icse26":
        return "in-domain"
    split_family = meta.get("split_family", "")
    if split_family.startswith("mutants_confounds_"):
        benchmark, strategy = split_family[len("mutants_confounds_"):].split("/", 1)
        suffix = "_he" if benchmark == "he_based" else ""
        return f"{strategy}_confounds{suffix}"
    if split_family.startswith("mutants_only_"):
        benchmark, strategy = split_family[len("mutants_only_"):].split("/", 1)
        suffix = "_he" if benchmark == "he_based" else ""
        return f"{strategy}{suffix}"
    if "ood_source" in meta:
        return meta["ood_source"]
    return meta.get("fit_source", "unknown")


def _test_dataset_from_metadata(meta: dict) -> str:
    return meta.get("test_dataset") or meta["dataset"]


def _test_one_of_4(results: dict) -> float:
    if "test_at_selected_layer" in results:
        return results["test_at_selected_layer"]["one_of_4"]
    return results["test_one_of_4"]


def _test_pairwise(results: dict) -> float:
    if "test_at_selected_layer" in results:
        return results["test_at_selected_layer"]["pairwise_acc"]
    return results["test_pairwise"]


def _normalize_cell(results: dict, meta: Optional[dict]) -> Optional[dict]:
    if meta is not None:
        if meta.get("experiment_script") != FIT_SCORE_SCRIPT:
            return None
        cell = {
            "model": meta["model"],
            "fit_source": _fit_source_from_metadata(meta),
            "test_dataset": _test_dataset_from_metadata(meta),
            "method": meta["method"],
            "representation": meta["representation"],
            "fit_framing": meta["fit_framing"],
            "val_framing": meta["val_framing"],
            "test_framing": meta["test_framing"],
            "test_one_of_4": _test_one_of_4(results),
            "test_pairwise": _test_pairwise(results),
        }
        return cell

    if not all(field in results for field in GROUP_FIELDS):
        return None
    return {
        **{field: results[field] for field in GROUP_FIELDS},
        "test_one_of_4": _test_one_of_4(results),
        "test_pairwise": _test_pairwise(results),
    }


def load_cells(results_dir: Path) -> List[dict]:
    cells = []
    for path in results_dir.rglob("results.json"):
        with open(path) as f:
            results = json.load(f)
        meta_path = path.parent / "metadata.json"
        meta = None
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
        cell = _normalize_cell(results, meta)
        if cell is not None:
            cells.append(cell)
    return cells


def aggregate(cells: List[dict]) -> Dict[Tuple, dict]:
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for cell in cells:
        key = tuple(cell[field] for field in GROUP_FIELDS)
        groups[key].append(cell)

    summary: Dict[Tuple, dict] = {}
    for key, group_cells in groups.items():
        one_of_4 = [c["test_one_of_4"] for c in group_cells]
        pairwise = [c["test_pairwise"] for c in group_cells]
        summary[key] = {
            "n_cells": len(group_cells),
            "mean_one_of_4": statistics.mean(one_of_4),
            "std_one_of_4": statistics.pstdev(one_of_4) if len(one_of_4) > 1 else 0.0,
            "mean_pairwise": statistics.mean(pairwise),
            "std_pairwise": statistics.pstdev(pairwise) if len(pairwise) > 1 else 0.0,
        }
    return summary


def write_csv(summary: Dict[Tuple, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GROUP_FIELDS) + [
        "n_cells", "mean_one_of_4", "std_one_of_4", "mean_pairwise", "std_pairwise",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(summary):
            row = dict(zip(GROUP_FIELDS, key))
            row.update(summary[key])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="analysis/ablation/summary.csv")
    args = parser.parse_args()

    cells = load_cells(Path(args.results_dir))
    print(f"[aggregate] loaded {len(cells)} result cells from {args.results_dir}")

    summary = aggregate(cells)
    write_csv(summary, Path(args.output))
    print(f"[aggregate] wrote {len(summary)} combo rows to {args.output}")


if __name__ == "__main__":
    main()
