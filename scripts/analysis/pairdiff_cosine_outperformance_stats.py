#!/usr/bin/env python3
"""Cross-fold statistics: layers that beat the selected layer on cosine similarity.

Purpose
-------
For each fold, uses the experiment ``selected_layer`` from ``results.json`` as a
reference. Within the requested layer set (default: top-percentile layers, which
always includes the selected layer), finds other layers whose **mean cosine
similarity** exceeds the selected layer's mean.

Aggregates across folds:
- how often such outperformance occurs
- average val_acc percentile of outperforming layers
- average val_acc gap vs. the selected layer
- average cosine similarity gain vs. the selected layer

Inputs
------
- Per-fold ``summary.json`` files under ``--cosine-output-dir`` (from
  ``plot_pairdiff_reading_cosine.py``)
- ``results.json`` per fold for selected layer and val accuracies

Outputs
-------
- JSON and CSV summaries under ``--output-dir`` (default:
  ``outputs/phase2_task2_cosine/outperformance_stats``)

Related scripts
---------------
- ``run_pairdiff_outperformance_stats.sh``: batch runner
- ``run_pairdiff_top_percentile_graphs.sh``: generates prerequisite cosine summaries
- ``plot_pairdiff_reading_cosine.py``: produces per-fold ``summary.json``

Usage
-----
::

    python scripts/analysis/pairdiff_cosine_outperformance_stats.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --benchmark bigcodebench --split test \\
        --folds 0 1 2 3 4 5 6 7 8 9 \\
        --layers top-percentile
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from plot_pairdiff_reading_cosine import (
    Args as CosineArgs,
    load_direction,
    load_fold_result_metrics,
    load_selected_layer,
    resolve_layers,
    select_top_percentile_layers,
)
from src.hidden_states import safe_model_dir
from src.vectors.direction import Direction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Args(CosineArgs):
    folds: List[int] = field(default_factory=lambda: list(range(10)))
    cosine_output_dir: Path = Path("outputs/phase2_task2_cosine/top_percentile")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cross-fold statistics for layers that outperform the experiment "
            "selected layer on mean cosine similarity."
        )
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--cosine-output-dir", default="outputs/phase2_task2_cosine/top_percentile")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-source", default="icse26")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--distribution", default="id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--layers",
        default="top-percentile",
        help="top-percentile, selected, all, or comma-separated layer indices",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile threshold when --layers top-percentile",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Folds to include in the aggregate statistics",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2_task2_cosine/outperformance_stats",
        help="Directory for JSON/CSV summary outputs",
    )
    ns = parser.parse_args()
    return Args(
        hidden_states_root=Path("."),
        vectors_root=Path(ns.vectors_root),
        results_root=Path(ns.results_root),
        model=ns.model,
        dataset_source=ns.dataset_source,
        benchmark=ns.benchmark,
        distribution=ns.distribution,
        split=ns.split,
        fold=0,
        framing=ns.framing,
        representation=ns.representation,
        method=ns.method,
        fit_framing=ns.fit_framing,
        val_framing=ns.val_framing,
        test_framing=ns.test_framing,
        layers=ns.layers,
        max_tasks=None,
        output_dir=Path(ns.output_dir),
        percentile=ns.percentile,
        folds=ns.folds,
        cosine_output_dir=Path(ns.cosine_output_dir),
    )


def replace_fold(args: Args, fold: int) -> Args:
    return replace(args, fold=fold)


def build_cosine_fold_dir(args: Args, fold: int) -> Path:
    return (
        args.cosine_output_dir
        / safe_model_dir(args.model)
        / args.dataset_source
        / args.benchmark
        / args.distribution
        / args.split
        / f"fold_{fold}"
        / args.framing
        / args.representation
        / args.method
        / args.fit_framing
        / f"val={args.val_framing}__test={args.test_framing}"
    )


def build_stats_output_dir(args: Args) -> Path:
    out = (
        args.output_dir
        / safe_model_dir(args.model)
        / args.dataset_source
        / args.benchmark
        / args.distribution
        / args.split
        / args.framing
        / args.representation
        / args.method
        / args.fit_framing
        / f"val={args.val_framing}__test={args.test_framing}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_analysis_layers(
    args: Args,
    direction: Direction,
    metrics: Dict[str, Dict[int, Dict[str, float]]],
) -> List[int]:
    if args.layers == "top-percentile":
        experiment_selected = load_selected_layer(args)
        layers = select_top_percentile_layers(
            metrics["val"], args.percentile, selected_layer=experiment_selected
        )
        if not layers:
            raise ValueError(f"No top-percentile layers found for fold {args.fold}")
        return layers
    return resolve_layers(args, 0, direction)


def val_acc_percentile(val_accs: Dict[int, float], layer: int) -> float:
    values = np.array(list(val_accs.values()), dtype=np.float64)
    return float(np.mean(values <= val_accs[layer]) * 100.0)


def load_layer_cosine_means(summary_path: Path) -> Dict[int, float]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    data = json.loads(summary_path.read_text())
    per_layer = data.get("per_layer", {})
    means: Dict[int, float] = {}
    for layer_key, stats in per_layer.items():
        if stats is None:
            continue
        means[int(layer_key)] = float(stats["mean"])
    if not means:
        raise ValueError(f"No per-layer cosine means in {summary_path}")
    return means


def analyze_fold(args: Args, fold: int, direction: Direction) -> Dict[str, object]:
    local_args = replace_fold(args, fold)
    metrics = load_fold_result_metrics(local_args, fold)
    layer_set = resolve_analysis_layers(local_args, direction, metrics)
    val_accs = {
        layer: float(metrics["val"][layer]["pairwise_acc"])
        for layer in layer_set
        if layer in metrics["val"] and metrics["val"][layer].get("pairwise_acc") is not None
    }
    all_val_accs = {
        int(layer): float(info["pairwise_acc"])
        for layer, info in metrics["val"].items()
        if info.get("pairwise_acc") is not None
    }

    summary_path = build_cosine_fold_dir(local_args, fold) / "summary.json"
    cosine_means = load_layer_cosine_means(summary_path)

    reference_layer = load_selected_layer(local_args)
    if reference_layer not in layer_set:
        raise ValueError(
            f"Experiment selected layer {reference_layer} not in analysis layer set "
            f"{layer_set} for fold {fold}"
        )
    reference_val_acc = val_accs[reference_layer]
    reference_cosine = cosine_means[reference_layer]

    outperformers: List[Dict[str, object]] = []
    for layer in sorted(layer_set):
        if layer == reference_layer:
            continue
        layer_cosine = cosine_means.get(layer)
        if layer_cosine is None:
            logger.warning("Fold %s layer %s missing from cosine summary; skipping", fold, layer)
            continue
        if layer_cosine <= reference_cosine:
            continue
        layer_val_acc = val_accs[layer]
        outperformers.append(
            {
                "fold": fold,
                "layer": layer,
                "selected_layer": reference_layer,
                "layer_val_acc": layer_val_acc,
                "selected_layer_val_acc": reference_val_acc,
                "val_acc_diff": reference_val_acc - layer_val_acc,
                "layer_val_acc_percentile": val_acc_percentile(all_val_accs, layer),
                "layer_cosine_mean": layer_cosine,
                "selected_layer_cosine_mean": reference_cosine,
                "cosine_diff": layer_cosine - reference_cosine,
            }
        )

    return {
        "fold": fold,
        "layer_set": layer_set,
        "selected_layer": reference_layer,
        "selected_layer_val_acc": reference_val_acc,
        "selected_layer_cosine_mean": reference_cosine,
        "num_layers_in_set": len(layer_set),
        "num_outperformers": len(outperformers),
        "outperformers": outperformers,
    }


def aggregate_across_folds(fold_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    instances: List[Dict[str, object]] = []
    for fold_result in fold_results:
        instances.extend(fold_result["outperformers"])

    if not instances:
        return {
            "num_outperformance_instances": 0,
            "num_folds_with_outperformers": 0,
            "num_folds_analyzed": len(fold_results),
            "mean_layer_val_acc_percentile": None,
            "mean_val_acc_diff": None,
            "mean_cosine_diff": None,
            "median_layer_val_acc_percentile": None,
            "median_val_acc_diff": None,
            "median_cosine_diff": None,
        }

    percentiles = np.array([float(row["layer_val_acc_percentile"]) for row in instances], dtype=np.float64)
    val_acc_diffs = np.array([float(row["val_acc_diff"]) for row in instances], dtype=np.float64)
    cosine_diffs = np.array([float(row["cosine_diff"]) for row in instances], dtype=np.float64)
    folds_with = len({int(row["fold"]) for row in instances})

    return {
        "num_outperformance_instances": len(instances),
        "num_folds_with_outperformers": folds_with,
        "num_folds_analyzed": len(fold_results),
        "mean_layer_val_acc_percentile": float(percentiles.mean()),
        "mean_val_acc_diff": float(val_acc_diffs.mean()),
        "mean_cosine_diff": float(cosine_diffs.mean()),
        "median_layer_val_acc_percentile": float(np.median(percentiles)),
        "median_val_acc_diff": float(np.median(val_acc_diffs)),
        "median_cosine_diff": float(np.median(cosine_diffs)),
    }


def write_instances_csv(path: Path, fold_results: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "fold",
        "layer",
        "selected_layer",
        "layer_val_acc",
        "selected_layer_val_acc",
        "val_acc_diff",
        "layer_val_acc_percentile",
        "layer_cosine_mean",
        "selected_layer_cosine_mean",
        "cosine_diff",
    ]
    rows: List[Dict[str, object]] = []
    for fold_result in fold_results:
        rows.extend(fold_result["outperformers"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.folds:
        raise ValueError("--folds must include at least one fold index")

    fold_results: List[Dict[str, object]] = []
    for fold in args.folds:
        logger.info("Analyzing fold %s", fold)
        direction = load_direction(replace_fold(args, fold))
        fold_results.append(analyze_fold(args, fold, direction))

    aggregate = aggregate_across_folds(fold_results)
    out_dir = build_stats_output_dir(args)
    layer_tag = args.layers.replace(",", "_")
    stem = f"outperformance_{layer_tag}_p{int(args.percentile)}"

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}_instances.csv"
    payload = {
        "config": {
            "model": args.model,
            "benchmark": args.benchmark,
            "split": args.split,
            "layers": args.layers,
            "percentile": args.percentile,
            "folds": args.folds,
            "cosine_output_dir": str(args.cosine_output_dir),
        },
        "aggregate": aggregate,
        "per_fold": fold_results,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    write_instances_csv(csv_path, fold_results)

    logger.info("Wrote aggregate statistics to %s", json_path)
    logger.info("Wrote per-instance rows to %s", csv_path)
    logger.info(
        "Outperformance instances: %s across %s/%s folds",
        aggregate["num_outperformance_instances"],
        aggregate["num_folds_with_outperformers"],
        aggregate["num_folds_analyzed"],
    )
    if aggregate["num_outperformance_instances"]:
        logger.info(
            "Mean val_acc percentile=%.2f, mean val_acc diff=%.4f, mean cosine diff=%.4f",
            aggregate["mean_layer_val_acc_percentile"],
            aggregate["mean_val_acc_diff"],
            aggregate["mean_cosine_diff"],
        )


if __name__ == "__main__":
    main()
