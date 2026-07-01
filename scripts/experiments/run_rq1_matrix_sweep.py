"""Run the full RQ1 matrix via fit_and_score_from_cache.py and write ordered CSV.

Executes every row from generate_rq1_matrix.py (384 configs), optionally in
parallel, aggregates per-row test metrics, and writes a CSV in the same row
order as the matrix template.

Representations are read from inside each cached ``bundle.pt`` (and listed in
``metadata.json``); they are not encoded in the directory path.

Usage:
    python scripts/experiments/run_rq1_matrix_sweep.py \\
        --hidden-states-root /scratch/re2230/hidden_states_sohaila \\
        --results-dir rq1_results \\
        --output analysis/ablation/rq1_results.csv \\
        --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.rq1_matrix_spec import (  # noqa: E402
    FIT_SOURCE_TO_RUNNER,
    MODEL_TO_RUNNER,
    OUTPUT_FIELDNAMES,
    REPRESENTATION_TO_RUNNER,
    bundle_has_representation,
    cell_match_key,
    iter_rq1_rows,
    representations_in_bundle_dir,
    row_match_key,
    row_required_bundle_dirs,
)


def _aggregate_cells(cells: List[dict]) -> dict:
    one_of_4 = [c["test_one_of_4"] for c in cells]
    pairwise = [c["test_pairwise"] for c in cells]
    n = len(cells)
    return {
        "n_cells": n,
        "avg_one_of_4": statistics.mean(one_of_4),
        "stddev_one_of_4": statistics.pstdev(one_of_4) if n > 1 else 0.0,
        "avg_pairwise": statistics.mean(pairwise),
        "stddev_pairwise": statistics.pstdev(pairwise) if n > 1 else 0.0,
    }


def _load_result_cells(results_dir: Path) -> Dict[Tuple, List[dict]]:
    groups: Dict[Tuple, List[dict]] = {}
    if not results_dir.is_dir():
        return groups
    for path in results_dir.rglob("results.json"):
        with open(path) as f:
            cell = json.load(f)
        required = {
            "model", "fit_source", "test_dataset", "method", "representation",
            "fit_framing", "val_framing", "test_framing",
            "test_one_of_4", "test_pairwise",
        }
        if not required.issubset(cell):
            continue
        key = cell_match_key(cell)
        groups.setdefault(key, []).append(cell)
    return groups


def _build_fit_score_cmd(
    row: dict,
    hidden_states_root: Path,
    data_root: Path,
    results_dir: Path,
) -> List[str]:
    return [
        sys.executable, "-u",
        str(REPO_ROOT / "scripts/experiments/fit_and_score_from_cache.py"),
        "--model", MODEL_TO_RUNNER[row["Model"]],
        "--representation", REPRESENTATION_TO_RUNNER[row["Representation"]],
        "--method", row["Method"],
        "--fit-framing", row["FramingFit"],
        "--val-framing", row["FramingVal"],
        "--test-framing", row["FramingTest"],
        "--fit-source", FIT_SOURCE_TO_RUNNER[row["FitSource"]],
        "--test-dataset", row["TestDataset"],
        "--hidden-states-root", str(hidden_states_root),
        "--data-root", str(data_root),
        "--results-dir", str(results_dir),
    ]


def _preflight_row(
    row: dict,
    hidden_states_root: Path,
) -> Optional[str]:
    """Return an error if any required bundle lacks the requested representation."""
    for data_split, bundle_dir in row_required_bundle_dirs(row, hidden_states_root):
        if not (bundle_dir / "bundle.pt").is_file():
            return f"missing cached bundle for {data_split}: {bundle_dir / 'bundle.pt'}"
        reps = representations_in_bundle_dir(bundle_dir)
        if not bundle_has_representation(reps, row["Representation"]):
            return (
                f"representation {row['Representation']!r} not in "
                f"{data_split} bundle {bundle_dir / 'bundle.pt'} "
                f"(cached representations: {reps})"
            )
    return None


def _run_one_row(
    row: dict,
    hidden_states_root: str,
    data_root: str,
    results_dir: str,
) -> Tuple[dict, str, str]:
    err = _preflight_row(row, Path(hidden_states_root))
    if err:
        return row, "failed", err

    cmd = _build_fit_score_cmd(
        row,
        Path(hidden_states_root),
        Path(data_root),
        Path(results_dir),
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
    except Exception as exc:
        return row, "failed", str(exc)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        return row, "failed", detail
    return row, "completed", ""


def _fill_metrics(row: dict, groups: Dict[Tuple, List[dict]]) -> dict:
    out = dict(row)
    key = row_match_key(row)
    cells = groups.get(key, [])
    if not cells:
        if out["Status"] != "failed":
            out["Status"] = "failed"
            out["Notes"] = "no result cells found after run"
        return out

    stats = _aggregate_cells(cells)
    out["avg"] = f"{stats['avg_one_of_4']:.6f}"
    out["std-dev"] = f"{stats['stddev_one_of_4']:.6f}"
    out["avg_one_of_4"] = f"{stats['avg_one_of_4']:.6f}"
    out["stddev_one_of_4"] = f"{stats['stddev_one_of_4']:.6f}"
    out["avg_pairwise"] = f"{stats['avg_pairwise']:.6f}"
    out["stddev_pairwise"] = f"{stats['stddev_pairwise']:.6f}"
    if out["Status"] != "failed":
        out["Status"] = "completed"
        out["Notes"] = f"n_cells={stats['n_cells']}"
    return out


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full RQ1 matrix sweep.")
    parser.add_argument(
        "--hidden-states-root",
        default="/scratch/re2230/hidden_states_sohaila",
    )
    parser.add_argument("--data-root", default="dataset/icse26")
    parser.add_argument("--results-dir", default="rq1_results")
    parser.add_argument("--output", default="analysis/ablation/rq1_results.csv")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel worker count (default: 1).")
    parser.add_argument(
        "--skip-prefilled",
        action="store_true",
        help="Skip rows marked pre-filled in the matrix template.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip runs; aggregate existing results into the output CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N matrix rows (for smoke tests).",
    )
    args = parser.parse_args()

    rows = list(iter_rq1_rows())
    if args.limit is not None:
        rows = rows[: args.limit]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    to_run = [
        row for row in rows
        if not (args.skip_prefilled and row["Status"] == "pre-filled")
    ]

    if not args.aggregate_only:
        print(f"[rq1] running {len(to_run)} configs with jobs={args.jobs}")
        if args.jobs <= 1:
            for i, row in enumerate(to_run, 1):
                print(f"[rq1] [{i}/{len(to_run)}] {row['ConfigID']} "
                      f"{row['FitSource']} {row['TestDataset']} {row['Model']}")
                _, status, note = _run_one_row(
                    row,
                    args.hidden_states_root,
                    args.data_root,
                    str(results_dir),
                )
                if status == "failed":
                    row["Status"] = "failed"
                    row["Notes"] = note
        else:
            completed = 0
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {
                    pool.submit(
                        _run_one_row,
                        row,
                        args.hidden_states_root,
                        args.data_root,
                        str(results_dir),
                    ): row
                    for row in to_run
                }
                for future in as_completed(futures):
                    row = futures[future]
                    completed += 1
                    try:
                        _, status, note = future.result()
                    except Exception as exc:
                        status, note = "failed", str(exc)
                    if status == "failed":
                        row["Status"] = "failed"
                        row["Notes"] = note
                    print(f"[rq1] [{completed}/{len(to_run)}] "
                          f"{row['ConfigID']} {row['FitSource']} "
                          f"{row['TestDataset']} {row['Model']} -> {status}")

    groups = _load_result_cells(results_dir)
    filled_rows = [_fill_metrics(row, groups) for row in rows]
    write_csv(filled_rows, Path(args.output))

    n_completed = sum(r["Status"] == "completed" for r in filled_rows)
    n_failed = sum(r["Status"] == "failed" for r in filled_rows)
    n_pending = sum(r["Status"] == "pending" for r in filled_rows)
    n_prefilled = sum(r["Status"] == "pre-filled" for r in filled_rows)
    print(
        f"[rq1] wrote {len(filled_rows)} rows to {args.output} "
        f"(completed={n_completed}, failed={n_failed}, "
        f"pending={n_pending}, pre-filled={n_prefilled})"
    )


if __name__ == "__main__":
    main()
