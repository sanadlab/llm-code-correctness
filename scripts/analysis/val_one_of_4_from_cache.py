#!/usr/bin/env python3
"""Compute validation 1-of-4 accuracy from cached hidden-state bundles.

This is the analysis entry point for the ``icse26_val_1_4`` validation split,
where each task has four candidate implementations (one correct, three
incorrect). It reuses existing fit directions from ``vectors/`` and projects
the cached validation bundles without re-running the model.

Results are written under ``results_icse26_val_1_4/`` by default — a separate
tree from the main ``results/`` directory used by the original experiments.

Usage:
    python scripts/analysis/val_one_of_4_from_cache.py \\
        --model codellama/CodeLlama-7b-Instruct-hf \\
        --benchmark bigcodebench

    # or via the bash wrapper:
    bash scripts/analysis/Bash_Scripts/run_val_one_of_4_from_cache.sh
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments._common import REPRESENTATIONS
from scripts.experiments.evaluate_from_cache import evaluate_fold

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = _REPO_ROOT / "results_icse26_val_1_4"
DEFAULT_HIDDEN_STATES_ROOT = _REPO_ROOT.parent / "hidden_states_re2230"
DEFAULT_VECTORS_ROOT = _REPO_ROOT / "vectors"
DEFAULT_SPLIT_FAMILY = "icse26_val_1_4"
DEFAULT_VECTORS_SPLIT_FAMILY = "icse26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score validation 1-of-4 from cached hidden states."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument(
        "--split-family",
        default=DEFAULT_SPLIT_FAMILY,
        help=f"Split family for cached validation bundles (default: {DEFAULT_SPLIT_FAMILY}).",
    )
    parser.add_argument(
        "--vectors-split-family",
        default=DEFAULT_VECTORS_SPLIT_FAMILY,
        help=(
            "Split family for fit directions in vectors/ "
            f"(default: {DEFAULT_VECTORS_SPLIT_FAMILY})."
        ),
    )
    parser.add_argument("--distribution", default="id")
    parser.add_argument(
        "--hidden-states-root",
        type=Path,
        default=DEFAULT_HIDDEN_STATES_ROOT,
        help=f"Root directory with cached bundle.pt files (default: {DEFAULT_HIDDEN_STATES_ROOT}).",
    )
    parser.add_argument(
        "--vectors-root",
        type=Path,
        default=DEFAULT_VECTORS_ROOT,
        help=f"Root directory with direction.pt files (default: {DEFAULT_VECTORS_ROOT}).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Output root for results.json files (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument("--representation", default="response_avg", choices=list(REPRESENTATIONS))
    parser.add_argument("--method", default="mean_diff", choices=["mean_diff", "pca"])
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--layer-selection-metric",
        default="one_of_4",
        choices=["one_of_4", "pairwise_acc"],
        help="Metric used to pick the monitoring layer on validation.",
    )
    parser.add_argument("--folds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional per-fold summary CSV path (default: <results-root>/summary.csv).",
    )
    return parser.parse_args()


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "benchmark",
        "split_family",
        "fold",
        "selected_layer",
        "val_one_of_4",
        "val_pairwise_acc",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()
    representation = REPRESENTATIONS[args.representation]
    summary_csv = args.summary_csv or (args.results_root / "summary.csv")

    logger.info("Hidden states root: %s", args.hidden_states_root.resolve())
    logger.info("Vectors root: %s", args.vectors_root.resolve())
    logger.info("Results root: %s", args.results_root.resolve())
    logger.info("Folds: %s", args.folds)

    per_fold: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for fold in args.folds:
        fold_result = evaluate_fold(
            fold=fold,
            representation=representation,
            hidden_states_root=args.hidden_states_root,
            vectors_root=args.vectors_root,
            results_dir=args.results_root,
            model=args.model,
            split_family=args.split_family,
            vectors_split_family=args.vectors_split_family,
            benchmark=args.benchmark,
            distribution=args.distribution,
            method=args.method,
            fit_framing=args.fit_framing,
            val_framing=args.val_framing,
            test_framing=args.test_framing,
            layer_selection_metric=args.layer_selection_metric,
            score_test=False,
            experiment_script="scripts/analysis/val_one_of_4_from_cache.py",
        )
        per_fold.append(fold_result)
        summary_rows.append(
            {
                "model": args.model,
                "benchmark": args.benchmark,
                "split_family": args.split_family,
                "fold": fold,
                "selected_layer": fold_result["selected_layer"],
                "val_one_of_4": fold_result["val_one_of_4"],
                "val_pairwise_acc": fold_result["val_pairwise_acc"],
            }
        )

    write_summary_csv(summary_csv, summary_rows)

    mean_val_1of4 = sum(float(x["val_one_of_4"]) for x in per_fold) / len(per_fold)
    mean_val_pair = sum(float(x["val_pairwise_acc"]) for x in per_fold) / len(per_fold)
    print(
        f"\n[summary] mean val 1-of-4 = {mean_val_1of4:.3f}  "
        f"mean val pairwise = {mean_val_pair:.3f}  (n_folds={len(per_fold)})"
    )
    print(f"[summary] wrote per-fold table to {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
