"""Aggregate `fit_and_score_from_cache.py` results into a mean/std summary CSV.

Walks a results directory for `results.json` files (one per fold, or per
outer/inner cell for OOD fit sources) and groups them by every ablation axis
(model, fit_source, test_dataset, method, representation, fit_framing,
val_framing, test_framing), computing the mean and population std of `test_one_of_4` and
`test_pairwise` within each group.

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
from typing import Dict, List, Tuple

GROUP_FIELDS = (
    "model", "fit_source", "test_dataset", "method",
    "representation", "fit_framing", "val_framing", "test_framing",
)


def load_cells(results_dir: Path) -> List[dict]:
    cells = []
    for path in results_dir.rglob("results.json"):
        with open(path) as f:
            cells.append(json.load(f))
    return cells


def aggregate(cells: List[dict]) -> Dict[Tuple, dict]:
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for cell in cells:
        if not all(field in cell for field in GROUP_FIELDS):
            continue  # not a fit_and_score_from_cache.py result; skip.
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
