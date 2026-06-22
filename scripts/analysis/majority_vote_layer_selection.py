#!/usr/bin/env python3
"""Cross-fold majority-vote layer selection from cosine similarity and accuracy.

Per fold (validation split), two votes are cast:
  1. Cosine vote: layer with highest mean pair-diff cosine similarity.
  2. Accuracy vote: layer with highest validation accuracy on a dataset-specific
     metric (``one_of_4`` for icse26_val_1_4; ``pairwise_acc`` for icse26).

Within each fold, ``majority_selected_layer`` is the shared pick when both votes
agree, otherwise the pick from ``--fold-tiebreak`` (defaults to the accuracy metric).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ANALYSIS_DIR = _REPO_ROOT / "scripts" / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from cosine_selection_layer_sim import (  # noqa: E402
    compute_cosine_entries,
    load_bundle,
    load_direction,
    load_fold_result_metrics,
    load_selected_layer,
    make_cosine_args,
    mean_cosine_by_layer,
    resolve_layers,
    safe_model_dir,
    select_layer_by_highest_cosine,
)
from scripts.experiments._common import select_layer_by_metric  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN_STATES_ROOT = _REPO_ROOT.parent / "hidden_states_re2230"
DEFAULT_VECTORS_ROOT = _REPO_ROOT / "vectors"
DEFAULT_RESULTS_ROOT_ICSE26_VAL_1_4 = _REPO_ROOT / "results_icse26_val_1_4"
DEFAULT_RESULTS_ROOT_ICSE26 = _REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "phase2_task2_majority_vote_layer"


@dataclass(frozen=True)
class FoldVotes:
    fold: int
    cosine_layer: int
    cosine_mean: float
    accuracy_layer: int
    accuracy_score: float
    agree: bool
    majority_selected_layer: int
    fold_selection_mode: str
    results_selected_layer: Optional[int]


PER_FOLD_COLUMNS = [
    "model",
    "dataset_source",
    "benchmark",
    "distribution",
    "fold",
    "selection_split",
    "accuracy_vote_metric",
    "framing",
    "representation",
    "method",
    "fit_framing",
    "val_framing",
    "test_framing",
    "cosine_selected_layer",
    "cosine_mean",
    "accuracy_selected_layer",
    "accuracy_vote_score",
    "votes_agree",
    "majority_selected_layer",
    "fold_selection_mode",
    "results_selected_layer",
    "per_fold_majority_consensus_layer",
    "majority_vote_layer",
    "majority_vote_total_votes",
]

LAYER_COUNT_COLUMNS = [
    "model",
    "dataset_source",
    "benchmark",
    "distribution",
    "accuracy_vote_metric",
    "layer",
    "per_fold_majority_votes",
    "pooled_total_votes",
    "cosine_votes",
    "accuracy_votes",
    "mean_val_accuracy_vote_metric",
    "is_per_fold_majority_consensus",
    "is_pooled_majority_winner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a layer by cross-fold majority vote (cosine + 1-of-4)."
    )
    parser.add_argument("--hidden-states-root", type=Path, default=DEFAULT_HIDDEN_STATES_ROOT)
    parser.add_argument("--vectors-root", type=Path, default=DEFAULT_VECTORS_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Results tree with val_per_layer per fold (default depends on --dataset-source).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset-source",
        default="icse26_val_1_4",
        help="Split family for hidden states and results (icse26, icse26_val_1_4, mutation, ...).",
    )
    parser.add_argument(
        "--vectors-dataset-source",
        default=None,
        help=(
            "Split family for fit directions in vectors/ "
            "(default: icse26 when dataset-source is icse26_val_1_4, else same as dataset-source)."
        ),
    )
    parser.add_argument("--benchmark", required=True, choices=["bigcodebench", "humaneval"])
    parser.add_argument("--distribution", default="id")
    parser.add_argument(
        "--selection-split",
        default="validate",
        choices=["validate", "test"],
        help="Split used for per-fold layer votes (default: validate).",
    )
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--method", default="mean_diff", choices=["mean_diff", "pca"])
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--accuracy-vote-metric",
        default=None,
        choices=["one_of_4", "pairwise_acc"],
        help=(
            "Validation metric for the accuracy vote (default: one_of_4 for "
            "icse26_val_1_4, pairwise_acc for icse26 and other datasets)."
        ),
    )
    parser.add_argument(
        "--fold-tiebreak",
        default=None,
        choices=["one_of_4", "pairwise_acc", "cosine"],
        help=(
            "When cosine and the accuracy vote disagree within a fold, which "
            "signal wins (default: same as --accuracy-vote-metric)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def default_accuracy_vote_metric(dataset_source: str) -> str:
    if dataset_source == "icse26_val_1_4":
        return "one_of_4"
    return "pairwise_acc"


def resolve_run_config(cli: argparse.Namespace) -> None:
    cli.accuracy_vote_metric = cli.accuracy_vote_metric or default_accuracy_vote_metric(
        cli.dataset_source
    )
    cli.fold_tiebreak = cli.fold_tiebreak or cli.accuracy_vote_metric
    if cli.results_root is None:
        cli.results_root = default_results_root(cli.dataset_source)
    cli.vectors_dataset_source = default_vectors_dataset_source(
        cli.dataset_source,
        cli.vectors_dataset_source,
    )


def default_results_root(dataset_source: str) -> Path:
    if dataset_source == "icse26_val_1_4":
        return DEFAULT_RESULTS_ROOT_ICSE26_VAL_1_4
    return DEFAULT_RESULTS_ROOT_ICSE26


def default_vectors_dataset_source(dataset_source: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    if dataset_source == "icse26_val_1_4":
        return "icse26"
    return dataset_source


def output_paths(cli: argparse.Namespace) -> Tuple[Path, Path, Path]:
    base = (
        cli.output_dir
        / safe_model_dir(cli.model)
        / cli.dataset_source
        / cli.benchmark
        / cli.distribution
        / cli.selection_split
        / cli.framing
        / cli.representation
        / cli.method
        / cli.fit_framing
        / f"val={cli.val_framing}__test={cli.test_framing}"
    )
    return (
        base / "majority_vote_per_fold.csv",
        base / "majority_vote_layer_counts.csv",
        base / "majority_vote_summary.json",
    )


def select_accuracy_vote_layer(
    val_per_layer: Dict[int, Dict[str, float]],
    metric: str,
) -> Tuple[int, float]:
    if not val_per_layer:
        raise ValueError("val_per_layer is empty; cannot select an accuracy-vote layer.")
    sample_metrics = next(iter(val_per_layer.values()))
    if metric not in sample_metrics:
        raise KeyError(
            f"val_per_layer does not contain {metric!r}. "
            f"Available keys: {sorted(sample_metrics)}. "
            f"For icse26 use --accuracy-vote-metric pairwise_acc; "
            f"for icse26_val_1_4 use one_of_4."
        )
    layer = select_layer_by_metric(val_per_layer, metric=metric)
    return layer, float(val_per_layer[layer][metric])


def compute_cosine_vote(cli: argparse.Namespace, fold: int) -> Tuple[int, float]:
    cosine_args = make_cosine_args(cli, fold, cli.selection_split)
    bundle = load_bundle(cosine_args)
    direction = load_direction(cosine_args)
    layers = resolve_layers(
        cosine_args,
        bundle[cosine_args.representation].shape[1],
        direction,
    )
    entries = compute_cosine_entries(cosine_args, bundle, direction, layers)
    layer_to_mean = mean_cosine_by_layer(entries)
    layer = select_layer_by_highest_cosine(layer_to_mean)
    return layer, layer_to_mean[layer]


def mean_val_metric_across_folds(
    fold_metrics: Dict[int, Dict[str, Dict[int, Dict[str, float]]]],
    metric: str,
) -> Dict[int, float]:
    sums: Dict[int, List[float]] = {}
    for metrics in fold_metrics.values():
        val = metrics.get("val", {})
        for layer, layer_metrics in val.items():
            value = layer_metrics.get(metric)
            if value is not None:
                sums.setdefault(int(layer), []).append(float(value))
    return {layer: float(np.mean(values)) for layer, values in sums.items() if values}


def select_majority_layer_for_fold(
    cosine_layer: int,
    accuracy_layer: int,
    fold_tiebreak: str,
) -> Tuple[int, str]:
    """Pick one layer for a single fold from its two validation votes."""
    if cosine_layer == accuracy_layer:
        return cosine_layer, "unanimous"
    if fold_tiebreak == "cosine":
        return cosine_layer, "tiebreak_cosine"
    return accuracy_layer, f"tiebreak_{fold_tiebreak}"


def pick_consensus_layer(
    counts: Counter,
    mean_val_metric: Dict[int, float],
    *,
    accuracy_votes: Optional[Counter] = None,
    cosine_votes: Optional[Counter] = None,
) -> int:
    """Pick the layer with the highest count; break ties with optional secondary keys."""
    accuracy_votes = accuracy_votes or Counter()
    cosine_votes = cosine_votes or Counter()
    if not counts:
        raise ValueError("No votes collected; cannot select a consensus layer.")
    return max(
        counts,
        key=lambda layer: (
            counts[layer],
            accuracy_votes[layer],
            cosine_votes[layer],
            mean_val_metric.get(layer, float("-inf")),
            layer,
        ),
    )


def pick_pooled_majority_layer(
    fold_votes: List[FoldVotes],
    mean_val_metric: Dict[int, float],
) -> Tuple[int, Dict[int, int], Counter, Counter]:
    """Pool every fold's cosine and accuracy winners into one vote tally."""
    pooled_votes = Counter()
    cosine_votes = Counter()
    accuracy_votes = Counter()

    for vote in fold_votes:
        pooled_votes[vote.cosine_layer] += 1
        pooled_votes[vote.accuracy_layer] += 1
        cosine_votes[vote.cosine_layer] += 1
        accuracy_votes[vote.accuracy_layer] += 1

    winner = pick_consensus_layer(
        pooled_votes,
        mean_val_metric,
        accuracy_votes=accuracy_votes,
        cosine_votes=cosine_votes,
    )
    return winner, dict(pooled_votes), cosine_votes, accuracy_votes


def pick_per_fold_majority_consensus(
    fold_votes: List[FoldVotes],
    mean_val_metric: Dict[int, float],
) -> Tuple[int, Counter]:
    per_fold_counts = Counter(vote.majority_selected_layer for vote in fold_votes)
    winner = pick_consensus_layer(per_fold_counts, mean_val_metric)
    return winner, per_fold_counts


def metric_at_layer(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    split: str,
    layer: int,
    key: str,
) -> Optional[float]:
    metric_split = "val" if split == "validate" else split
    value = metrics.get(metric_split, {}).get(layer, {}).get(key)
    return None if value is None else float(value)


def process_fold(cli: argparse.Namespace, fold: int) -> Tuple[FoldVotes, Dict[str, Dict[int, Dict[str, float]]]]:
    ref_args = make_cosine_args(cli, fold, cli.selection_split)
    metrics = load_fold_result_metrics(ref_args, fold)
    val_per_layer = metrics["val"]

    cosine_layer, cosine_mean = compute_cosine_vote(cli, fold)
    accuracy_layer, accuracy_score = select_accuracy_vote_layer(
        val_per_layer, cli.accuracy_vote_metric
    )
    majority_selected_layer, fold_selection_mode = select_majority_layer_for_fold(
        cosine_layer,
        accuracy_layer,
        cli.fold_tiebreak,
    )

    results_selected_layer: Optional[int] = None
    try:
        results_selected_layer = load_selected_layer(ref_args)
    except (FileNotFoundError, KeyError):
        logger.warning("Fold %s: no results selected_layer; leaving column empty", fold)

    vote = FoldVotes(
        fold=fold,
        cosine_layer=cosine_layer,
        cosine_mean=cosine_mean,
        accuracy_layer=accuracy_layer,
        accuracy_score=accuracy_score,
        agree=cosine_layer == accuracy_layer,
        majority_selected_layer=majority_selected_layer,
        fold_selection_mode=fold_selection_mode,
        results_selected_layer=results_selected_layer,
    )
    return vote, metrics


def write_per_fold_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_FOLD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_layer_counts_csv(
    path: Path,
    cli: argparse.Namespace,
    pooled_vote_counts: Dict[int, int],
    per_fold_majority_counts: Counter,
    cosine_votes: Counter,
    accuracy_votes: Counter,
    mean_val_metric: Dict[int, float],
    pooled_winner: int,
    per_fold_consensus: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_layers = sorted(
        set(pooled_vote_counts)
        | set(per_fold_majority_counts)
        | set(cosine_votes)
        | set(accuracy_votes)
    )
    rows: List[Dict[str, object]] = []
    for layer in all_layers:
        rows.append(
            {
                "model": cli.model,
                "dataset_source": cli.dataset_source,
                "benchmark": cli.benchmark,
                "distribution": cli.distribution,
                "accuracy_vote_metric": cli.accuracy_vote_metric,
                "layer": layer,
                "per_fold_majority_votes": per_fold_majority_counts.get(layer, 0),
                "pooled_total_votes": pooled_vote_counts.get(layer, 0),
                "cosine_votes": cosine_votes.get(layer, 0),
                "accuracy_votes": accuracy_votes.get(layer, 0),
                "mean_val_accuracy_vote_metric": mean_val_metric.get(layer),
                "is_per_fold_majority_consensus": layer == per_fold_consensus,
                "is_pooled_majority_winner": layer == pooled_winner,
            }
        )
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAYER_COUNT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def mean_metric_at_selected_layers(
    fold_votes: List[FoldVotes],
    fold_metrics: Dict[int, Dict[str, Dict[int, Dict[str, float]]]],
    split: str,
    layer_attr: str,
    metric: str,
) -> Optional[float]:
    scores = []
    for vote in fold_votes:
        layer = getattr(vote, layer_attr)
        value = metric_at_layer(fold_metrics[vote.fold], split, layer, metric)
        if value is not None:
            scores.append(value)
    return float(np.mean(scores)) if scores else None


def mean_metric_at_fixed_layer(
    fold_votes: List[FoldVotes],
    fold_metrics: Dict[int, Dict[str, Dict[int, Dict[str, float]]]],
    split: str,
    layer: int,
    metric: str,
) -> Optional[float]:
    scores = []
    for vote in fold_votes:
        value = metric_at_layer(fold_metrics[vote.fold], split, layer, metric)
        if value is not None:
            scores.append(value)
    return float(np.mean(scores)) if scores else None


def write_summary_json(
    path: Path,
    cli: argparse.Namespace,
    pooled_winner: int,
    per_fold_consensus: int,
    pooled_vote_counts: Dict[int, int],
    per_fold_majority_counts: Counter,
    cosine_votes: Counter,
    accuracy_votes: Counter,
    fold_votes: List[FoldVotes],
    mean_val_metric: Dict[int, float],
    fold_metrics: Dict[int, Dict[str, Dict[int, Dict[str, float]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_metric = cli.accuracy_vote_metric

    summary = {
        "model": cli.model,
        "dataset_source": cli.dataset_source,
        "vectors_dataset_source": cli.vectors_dataset_source,
        "benchmark": cli.benchmark,
        "distribution": cli.distribution,
        "selection_split": cli.selection_split,
        "accuracy_vote_metric": accuracy_metric,
        "fold_tiebreak": cli.fold_tiebreak,
        "framing": cli.framing,
        "representation": cli.representation,
        "method": cli.method,
        "fit_framing": cli.fit_framing,
        "val_framing": cli.val_framing,
        "test_framing": cli.test_framing,
        "folds": [vote.fold for vote in fold_votes],
        "n_folds": len(fold_votes),
        "pooled_total_votes": 2 * len(fold_votes),
        "per_fold_majority_consensus_layer": per_fold_consensus,
        "per_fold_majority_consensus_count": per_fold_majority_counts[per_fold_consensus],
        "per_fold_majority_counts_by_layer": {
            str(layer): count for layer, count in sorted(per_fold_majority_counts.items())
        },
        "majority_vote_layer": pooled_winner,
        "majority_vote_count": pooled_vote_counts[pooled_winner],
        "pooled_vote_counts_by_layer": {
            str(layer): count for layer, count in sorted(pooled_vote_counts.items())
        },
        "cosine_votes_by_layer": {str(layer): cosine_votes[layer] for layer in sorted(cosine_votes)},
        "accuracy_votes_by_layer": {
            str(layer): accuracy_votes[layer] for layer in sorted(accuracy_votes)
        },
        "n_folds_unanimous": sum(
            1 for vote in fold_votes if vote.fold_selection_mode == "unanimous"
        ),
        "n_folds_tiebreak_accuracy": sum(
            1
            for vote in fold_votes
            if vote.fold_selection_mode == f"tiebreak_{accuracy_metric}"
        ),
        "n_folds_tiebreak_cosine": sum(
            1 for vote in fold_votes if vote.fold_selection_mode == "tiebreak_cosine"
        ),
        f"mean_val_{accuracy_metric}_cv_majority_selected": mean_metric_at_selected_layers(
            fold_votes,
            fold_metrics,
            cli.selection_split,
            "majority_selected_layer",
            accuracy_metric,
        ),
        "mean_test_one_of_4_cv_majority_selected": mean_metric_at_selected_layers(
            fold_votes,
            fold_metrics,
            "test",
            "majority_selected_layer",
            "one_of_4",
        ),
        f"mean_val_{accuracy_metric}_at_pooled_winner": mean_metric_at_fixed_layer(
            fold_votes,
            fold_metrics,
            cli.selection_split,
            pooled_winner,
            accuracy_metric,
        ),
        "mean_test_one_of_4_at_pooled_winner": mean_metric_at_fixed_layer(
            fold_votes,
            fold_metrics,
            "test",
            pooled_winner,
            "one_of_4",
        ),
        "per_fold": [
            {
                "fold": vote.fold,
                "cosine_layer": vote.cosine_layer,
                "cosine_mean": vote.cosine_mean,
                "accuracy_layer": vote.accuracy_layer,
                "accuracy_score": vote.accuracy_score,
                "agree": vote.agree,
                "majority_selected_layer": vote.majority_selected_layer,
                "fold_selection_mode": vote.fold_selection_mode,
                "results_selected_layer": vote.results_selected_layer,
                f"val_{accuracy_metric}_at_majority_selected": metric_at_layer(
                    fold_metrics[vote.fold],
                    cli.selection_split,
                    vote.majority_selected_layer,
                    accuracy_metric,
                ),
                "test_one_of_4_at_majority_selected": metric_at_layer(
                    fold_metrics[vote.fold],
                    "test",
                    vote.majority_selected_layer,
                    "one_of_4",
                ),
            }
            for vote in fold_votes
        ],
        f"mean_val_{accuracy_metric}_by_layer": {
            str(layer): value for layer, value in sorted(mean_val_metric.items())
        },
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    cli = parse_args()
    resolve_run_config(cli)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    per_fold_path, layer_counts_path, summary_path = output_paths(cli)
    logger.info("Dataset source: %s", cli.dataset_source)
    logger.info("Vectors dataset source: %s", cli.vectors_dataset_source)
    logger.info("Hidden states root: %s", cli.hidden_states_root.resolve())
    logger.info("Vectors root: %s", cli.vectors_root.resolve())
    logger.info("Results root: %s", cli.results_root.resolve())
    logger.info("Selection split: %s", cli.selection_split)
    logger.info("Accuracy vote metric: %s", cli.accuracy_vote_metric)
    logger.info("Fold tiebreak: %s", cli.fold_tiebreak)
    logger.info("Folds: %s", cli.folds)
    logger.info("Per-fold CSV: %s", per_fold_path.resolve())
    logger.info("Layer counts CSV: %s", layer_counts_path.resolve())
    logger.info("Summary JSON: %s", summary_path.resolve())

    start = time.time()
    fold_votes: List[FoldVotes] = []
    fold_metrics: Dict[int, Dict[str, Dict[int, Dict[str, float]]]] = {}

    for fold in cli.folds:
        logger.info("=" * 80)
        logger.info("Processing fold %s", fold)
        vote, metrics = process_fold(cli, fold)
        fold_votes.append(vote)
        fold_metrics[fold] = metrics
        logger.info(
            "Fold %s votes: cosine=%s (mean=%.4f), %s=%s (score=%.4f), "
            "majority_selected=%s (%s)",
            fold,
            vote.cosine_layer,
            vote.cosine_mean,
            cli.accuracy_vote_metric,
            vote.accuracy_layer,
            vote.accuracy_score,
            vote.majority_selected_layer,
            vote.fold_selection_mode,
        )

    mean_val_metric = mean_val_metric_across_folds(fold_metrics, cli.accuracy_vote_metric)
    pooled_winner, pooled_vote_counts, cosine_votes, accuracy_votes = pick_pooled_majority_layer(
        fold_votes, mean_val_metric
    )
    per_fold_consensus, per_fold_majority_counts = pick_per_fold_majority_consensus(
        fold_votes, mean_val_metric
    )

    per_fold_rows: List[Dict[str, object]] = []
    for vote in fold_votes:
        per_fold_rows.append(
            {
                "model": cli.model,
                "dataset_source": cli.dataset_source,
                "benchmark": cli.benchmark,
                "distribution": cli.distribution,
                "fold": vote.fold,
                "selection_split": cli.selection_split,
                "accuracy_vote_metric": cli.accuracy_vote_metric,
                "framing": cli.framing,
                "representation": cli.representation,
                "method": cli.method,
                "fit_framing": cli.fit_framing,
                "val_framing": cli.val_framing,
                "test_framing": cli.test_framing,
                "cosine_selected_layer": vote.cosine_layer,
                "cosine_mean": vote.cosine_mean,
                "accuracy_selected_layer": vote.accuracy_layer,
                "accuracy_vote_score": vote.accuracy_score,
                "votes_agree": vote.agree,
                "majority_selected_layer": vote.majority_selected_layer,
                "fold_selection_mode": vote.fold_selection_mode,
                "results_selected_layer": vote.results_selected_layer,
                "per_fold_majority_consensus_layer": per_fold_consensus,
                "majority_vote_layer": pooled_winner,
                "majority_vote_total_votes": pooled_vote_counts[pooled_winner],
            }
        )

    write_per_fold_csv(per_fold_path, per_fold_rows)
    write_layer_counts_csv(
        layer_counts_path,
        cli,
        pooled_vote_counts,
        per_fold_majority_counts,
        cosine_votes,
        accuracy_votes,
        mean_val_metric,
        pooled_winner,
        per_fold_consensus,
    )
    write_summary_json(
        summary_path,
        cli,
        pooled_winner,
        per_fold_consensus,
        pooled_vote_counts,
        per_fold_majority_counts,
        cosine_votes,
        accuracy_votes,
        fold_votes,
        mean_val_metric,
        fold_metrics,
    )

    logger.info(
        "Per-fold majority consensus layer = %s (%s/%s folds)",
        per_fold_consensus,
        per_fold_majority_counts[per_fold_consensus],
        len(fold_votes),
    )
    logger.info(
        "Pooled majority-vote layer = %s (%s/%s votes, %s/%s folds unanimous)",
        pooled_winner,
        pooled_vote_counts[pooled_winner],
        2 * len(fold_votes),
        sum(1 for vote in fold_votes if vote.fold_selection_mode == "unanimous"),
        len(fold_votes),
    )
    logger.info("Wrote outputs in %.2f seconds", time.time() - start)


if __name__ == "__main__":
    main()
