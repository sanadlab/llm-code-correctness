#!/usr/bin/env python3
"""Compare tiered layer selection vs validation-accuracy layer selection.

For each fold, selects a layer with a three-level criterion:
  1. Highest validation one-of-4 accuracy (when available in ``val_per_layer``)
  2. On ties: highest mean pair-diff cosine similarity on the validate split
  3. On ties: deepest layer (highest layer index)

Compares that tiered layer against the layer chosen by validation pairwise
accuracy only (deepest on ties), matching ``select_layer_by_metric(..., "pairwise_acc")``.

Outputs a per-fold CSV with validation accuracy, mean cosine similarity, and
test one-of-4 for both selected layers. Works with any ``--dataset-source``
(e.g. ``mutation`` split families, ``icse26_val_1_4``, ``icse26``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scripts.experiments._common import select_layer_by_metric

from cosine_selection_layer_sim import (
    compute_cosine_entries,
    load_bundle,
    load_direction,
    load_fold_result_metrics,
    make_cosine_args,
    mean_cosine_by_layer,
    resolve_layers,
    safe_model_dir,
)

logger = logging.getLogger(__name__)


CSV_COLUMNS = [
    "model",
    "dataset_source",
    "benchmark",
    "distribution",
    "fold",
    "framing",
    "representation",
    "method",
    "fit_framing",
    "val_framing",
    "test_framing",
    "tiered_primary_metric",
    "val_acc_selected_layer",
    "tiered_selected_layer",
    "same_selected_layer",
    "val_acc_selected_val_pairwise_acc",
    "val_acc_selected_val_one_of_4",
    "val_acc_selected_mean_cosine",
    "val_acc_selected_test_one_of_4",
    "tiered_selected_val_pairwise_acc",
    "tiered_selected_val_one_of_4",
    "tiered_selected_mean_cosine",
    "tiered_selected_test_one_of_4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tiered (val one-of-4 -> cosine -> deepest) layer selection "
            "against validation pairwise-accuracy selection per fold."
        )
    )
    parser.add_argument("--hidden-states-root", default="hidden_states_sohaila")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset-source",
        default="icse26",
        help="Split family for hidden states and results (e.g. icse26_val_1_4, icse26, s1_baseline).",
    )
    parser.add_argument(
        "--vectors-dataset-source",
        default=None,
        help="Split family for direction vectors (default: same as --dataset-source).",
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--distribution", default="id")
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2_task2_tiered_vs_val_acc_layer_selection",
    )
    return parser.parse_args()


def metric_value(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    split: str,
    layer: int,
    key: str,
) -> Optional[float]:
    value = metrics.get(split, {}).get(layer, {}).get(key)
    return None if value is None else float(value)


def select_layer_tiered(
    val_per_layer: Dict[int, Dict[str, float]],
    layer_to_mean_cosine: Dict[int, float],
    layers: List[int],
) -> tuple[int, str]:
    """Return (layer, primary_metric_used)."""
    has_one_of_4 = any(
        val_per_layer.get(layer, {}).get("one_of_4") is not None for layer in layers
    )
    primary_metric = "one_of_4" if has_one_of_4 else "pairwise_acc"

    eligible = [
        layer
        for layer in layers
        if val_per_layer.get(layer, {}).get(primary_metric) is not None
    ]
    if not eligible:
        raise ValueError(f"No layers with validation metric {primary_metric!r}")

    max_primary = max(val_per_layer[layer][primary_metric] for layer in eligible)
    tied = [
        layer
        for layer in eligible
        if val_per_layer[layer][primary_metric] == max_primary
    ]
    if len(tied) == 1:
        return tied[0], primary_metric

    max_cosine = max(layer_to_mean_cosine[layer] for layer in tied)
    tied = [layer for layer in tied if layer_to_mean_cosine[layer] == max_cosine]
    return max(tied), primary_metric


def build_row_for_fold(cli: argparse.Namespace, fold: int) -> Dict[str, object]:
    cosine_args = make_cosine_args(cli, fold, "validate")
    metrics = load_fold_result_metrics(cosine_args, fold)
    val_per_layer = metrics["val"]

    bundle = load_bundle(cosine_args)
    if cosine_args.representation not in bundle.acts:
        raise KeyError(
            f"Representation {cosine_args.representation!r} not found in bundle acts. "
            f"Available: {list(bundle.acts)}"
        )

    direction = load_direction(cosine_args)
    layers = resolve_layers(
        cosine_args,
        bundle[cosine_args.representation].shape[1],
        direction,
    )
    entries = compute_cosine_entries(cosine_args, bundle, direction, layers)
    layer_to_mean_cosine = mean_cosine_by_layer(entries)

    val_acc_layer = select_layer_by_metric(val_per_layer, "pairwise_acc")
    tiered_layer, primary_metric = select_layer_tiered(
        val_per_layer, layer_to_mean_cosine, layers
    )

    return {
        "model": cli.model,
        "dataset_source": cli.dataset_source,
        "benchmark": cli.benchmark,
        "distribution": cli.distribution,
        "fold": fold,
        "framing": cli.framing,
        "representation": cli.representation,
        "method": cli.method,
        "fit_framing": cli.fit_framing,
        "val_framing": cli.val_framing,
        "test_framing": cli.test_framing,
        "tiered_primary_metric": primary_metric,
        "val_acc_selected_layer": val_acc_layer,
        "tiered_selected_layer": tiered_layer,
        "same_selected_layer": val_acc_layer == tiered_layer,
        "val_acc_selected_val_pairwise_acc": metric_value(
            metrics, "val", val_acc_layer, "pairwise_acc"
        ),
        "val_acc_selected_val_one_of_4": metric_value(
            metrics, "val", val_acc_layer, "one_of_4"
        ),
        "val_acc_selected_mean_cosine": layer_to_mean_cosine.get(val_acc_layer),
        "val_acc_selected_test_one_of_4": metric_value(
            metrics, "test", val_acc_layer, "one_of_4"
        ),
        "tiered_selected_val_pairwise_acc": metric_value(
            metrics, "val", tiered_layer, "pairwise_acc"
        ),
        "tiered_selected_val_one_of_4": metric_value(
            metrics, "val", tiered_layer, "one_of_4"
        ),
        "tiered_selected_mean_cosine": layer_to_mean_cosine.get(tiered_layer),
        "tiered_selected_test_one_of_4": metric_value(
            metrics, "test", tiered_layer, "one_of_4"
        ),
    }


def output_csv_path(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.output_dir)
        / safe_model_dir(cli.model)
        / cli.dataset_source
        / cli.benchmark
        / cli.distribution
        / "tiered_vs_val_acc_layer_selection.csv"
    )


def write_table(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logger.info("Dataset source: %s", cli.dataset_source)
    logger.info(
        "Vectors dataset source: %s",
        cli.vectors_dataset_source or cli.dataset_source,
    )
    logger.info("Folds: %s", cli.folds)

    output_path = output_csv_path(cli)
    logger.info("Output table: %s", output_path.resolve())

    rows: List[Dict[str, object]] = []
    for fold in cli.folds:
        logger.info("Processing fold %s", fold)
        row = build_row_for_fold(cli, fold)
        rows.append(row)
        write_table(rows, output_path)
        logger.info(
            "Fold %s: val_acc layer=%s tiered layer=%s same=%s",
            fold,
            row["val_acc_selected_layer"],
            row["tiered_selected_layer"],
            row["same_selected_layer"],
        )

    logger.info("Wrote comparison table to %s", output_path)


if __name__ == "__main__":
    main()
