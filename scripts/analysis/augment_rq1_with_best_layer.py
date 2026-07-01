"""Add oracle-best test 1-of-4 columns to an existing RQ1 CSV.

Reads an existing RQ1 matrix CSV (for example `analysis/ablation/rq1_results.csv`)
and scans per-cell `results.json` files under `--results-dir`. For each matrix
row, it computes:

- avg (best, 1of4): mean across cells of the best test-layer one_of_4
- std-dev (best, 1of4): population std-dev across those cell-level best values

This keeps all original CSV columns and row ordering, and appends the two new
columns at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.analysis.rq1_matrix_spec import cell_match_key, row_match_key

BEST_AVG_COL = "avg (best, 1of4)"
BEST_STD_COL = "std-dev (best, 1of4)"


def _safe_best_one_of_4(cell: dict) -> Optional[float]:
    """Return per-cell oracle best test one_of_4, or None when unavailable."""
    test_per_layer = cell.get("test_per_layer")
    if isinstance(test_per_layer, dict) and test_per_layer:
        best = None
        for metrics in test_per_layer.values():
            if not isinstance(metrics, dict):
                continue
            val = metrics.get("one_of_4")
            if val is None:
                continue
            try:
                score = float(val)
            except (TypeError, ValueError):
                continue
            if best is None or score > best:
                best = score
        if best is not None:
            return best

    # Backward-compatible fallback for old slim result files.
    val = cell.get("test_one_of_4")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_best_scores_by_key(results_dir: Path) -> Dict[Tuple, List[float]]:
    groups: Dict[Tuple, List[float]] = {}
    if not results_dir.is_dir():
        return groups

    for path in results_dir.rglob("results.json"):
        with open(path) as f:
            cell = json.load(f)

        required = {
            "model",
            "fit_source",
            "test_dataset",
            "method",
            "representation",
            "fit_framing",
            "val_framing",
            "test_framing",
        }
        if not required.issubset(cell):
            continue

        best = _safe_best_one_of_4(cell)
        if best is None:
            continue

        key = cell_match_key(cell)
        groups.setdefault(key, []).append(best)
    return groups


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def augment_rows(rows: List[dict], best_groups: Dict[Tuple, List[float]]) -> List[dict]:
    out_rows = []
    for row in rows:
        out = dict(row)
        vals = best_groups.get(row_match_key(row), [])
        if vals:
            out[BEST_AVG_COL] = _fmt(statistics.mean(vals))
            out[BEST_STD_COL] = _fmt(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        else:
            out[BEST_AVG_COL] = ""
            out[BEST_STD_COL] = ""
        out_rows.append(out)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="analysis/ablation/rq1_results.csv",
        help="Existing RQ1 CSV to copy and augment.",
    )
    parser.add_argument(
        "--results-dir",
        default="rq1_results",
        help="Directory containing per-cell results.json files.",
    )
    parser.add_argument(
        "--output-csv",
        default="analysis/ablation/rq1_results_with_best_layer.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    results_dir = Path(args.results_dir)

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    best_groups = _load_best_scores_by_key(results_dir)
    augmented_rows = augment_rows(rows, best_groups)

    if BEST_AVG_COL not in fieldnames:
        fieldnames.append(BEST_AVG_COL)
    if BEST_STD_COL not in fieldnames:
        fieldnames.append(BEST_STD_COL)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(augmented_rows)

    print(
        f"[best-layer] wrote {len(augmented_rows)} rows to {output_csv} "
        f"from {len(best_groups)} matched matrix groups"
    )


if __name__ == "__main__":
    main()

