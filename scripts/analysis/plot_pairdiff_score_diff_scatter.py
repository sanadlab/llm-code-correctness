#!/usr/bin/env python3
"""Scatter plots of cosine similarity vs. pairwise score_diff.

Purpose
-------
Visualizes ``pairdiff_score_diff_cosine.csv`` as jittered scatter plots (x-axis =
cosine similarity; y-axis is random jitter for visibility). Point shape encodes
whether the correct sample outscores the incorrect sample **in that pair**:

- circle (o): score_diff >= 0
- x: score_diff < 0

Inputs
------
- ``pairdiff_score_diff_cosine.csv`` produced by ``plot_pairdiff_score_diff.py``
- ``results.json`` (when filtering layers via ``--layers selected`` or top-percentile)

Outputs
-------
- Single mode: ``layer_{L}_cosine_score_diff_scatter.png`` per layer
- Cross-fold mode: ``layer_{L}_cosine_score_diff_scatter_cross_fold.png`` (subplot per fold)

Key CLI flags
-------------
- ``--mode``: ``single`` (one CSV) or ``cross-fold`` (grid over folds)
- ``--layers``: ``all``, ``selected``, ``top-percentile``, or comma-separated indices
- ``--score-diff-root``: parent dir with ``fold_*/.../pairdiff_score_diff_cosine.csv``

Related scripts
---------------
- ``run_pairdiff_score_diff_graphs.sh``: prerequisite CSV generation
- ``run_pairdiff_score_diff_scatter_cross_fold.sh``: batch cross-fold scatter grids
- ``plot_pairdiff_task_max_scatter.py``: task-level (1-of-N) marker variant

Usage
-----
::

    python scripts/analysis/plot_pairdiff_score_diff_scatter.py \\
        --mode cross-fold \\
        --score-diff-root outputs/phase2_task2_score_diff/top_percentile/MODEL/.../test \\
        --layers top-percentile --percentile 90
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_pairdiff_reading_cosine import select_top_percentile_layers
from src.hidden_states.cache import vector_common_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter-plot cosine similarity per layer from pairdiff_score_diff_cosine.csv."
    )
    parser.add_argument(
        "--mode",
        choices=("single", "cross-fold"),
        default="single",
        help="single: one CSV and one PNG per layer; cross-fold: grid of folds per layer",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to pairdiff_score_diff_cosine.csv (required for --mode single)",
    )
    parser.add_argument(
        "--score-diff-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing fold_*/.../pairdiff_score_diff_cosine.csv "
            "(required for --mode cross-fold)"
        ),
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Fold indices for --mode cross-fold",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: same directory as --csv or --score-diff-root)",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="all, selected, top-percentile, or comma-separated layer indices",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Results root used when --layers is selected or top-percentile",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile threshold used when --layers top-percentile",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=None,
        help="Optional number of subplot columns for cross-fold grids",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.35,
        help="Half-width of uniform y jitter applied for point visibility",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for y jitter",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=12.0,
        help="Marker size for scatter points",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Marker alpha for scatter points",
    )
    return parser.parse_args()


def parse_explicit_layers(layers_arg: str) -> List[int]:
    values = [item.strip() for item in layers_arg.split(",") if item.strip()]
    if not values:
        raise ValueError("--layers must be all, selected, top-percentile, or comma-separated integers")
    layers: List[int] = []
    for value in values:
        try:
            layers.append(int(value))
        except ValueError as exc:
            raise ValueError(f"Invalid layer index: {value}") from exc
    return layers


def results_json_path(row: Dict[str, str], results_root: Path) -> Path:
    common = vector_common_path(
        row["model"],
        row["dataset_source"],
        row["benchmark"],
        row["distribution"],
        row["method"],
        row["fit_framing"],
        row["representation"],
    )
    eval_segment = f"val={row['val_framing']}__test={row['test_framing']}"
    return results_root / common / eval_segment / f"fold_{int(row['fold'])}" / "results.json"


def load_fold_result_metrics_from_row(
    row: Dict[str, str],
    results_root: Path,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    results_path = results_json_path(row, results_root)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Fold results not found: {results_path}\n"
            f"Check --results-root and CSV metadata columns"
        )
    data = json.loads(results_path.read_text())
    if "test_per_layer" not in data or "val_per_layer" not in data:
        raise KeyError(f"results.json missing per-layer accuracy fields: {results_path}")
    return {
        "test": {int(layer): metrics for layer, metrics in data["test_per_layer"].items()},
        "val": {int(layer): metrics for layer, metrics in data["val_per_layer"].items()},
    }


def load_selected_layer_from_row(row: Dict[str, str], results_root: Path) -> int:
    results_path = results_json_path(row, results_root)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Selected-layer results not found: {results_path}\n"
            f"Check --results-root and CSV metadata columns"
        )
    data = json.loads(results_path.read_text())
    if "selected_layer" not in data:
        raise KeyError(f"results.json missing selected_layer field: {results_path}")
    return int(data["selected_layer"])


def resolve_layer_filter(
    rows: Sequence[Dict[str, str]],
    layers_arg: str,
    results_root: Path,
    percentile: float,
) -> List[int] | None:
    if layers_arg == "all":
        return None
    if layers_arg == "selected":
        return [load_selected_layer_from_row(rows[0], results_root)]
    if layers_arg == "top-percentile":
        metrics = load_fold_result_metrics_from_row(rows[0], results_root)
        experiment_selected = load_selected_layer_from_row(rows[0], results_root)
        layers = select_top_percentile_layers(
            metrics["val"], percentile, selected_layer=experiment_selected
        )
        if not layers:
            raise ValueError("No top-percentile layers found from results.json")
        return layers
    return parse_explicit_layers(layers_arg)


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    required = {
        "layer",
        "cosine_similarity",
        "score_diff",
        "model",
        "dataset_source",
        "benchmark",
        "distribution",
        "method",
        "fit_framing",
        "representation",
        "val_framing",
        "test_framing",
        "fold",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise KeyError(f"CSV missing required columns: {sorted(missing)}")
    return rows


def find_fold_csv(score_diff_root: Path, fold: int) -> Path:
    matches = sorted(score_diff_root.glob(f"fold_{fold}/**/pairdiff_score_diff_cosine.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No pairdiff_score_diff_cosine.csv found under {score_diff_root}/fold_{fold}/"
        )
    if len(matches) > 1:
        logger.warning(
            "Multiple CSV files for fold %s; using %s", fold, matches[0]
        )
    return matches[0]


def group_rows_by_layer(
    rows: Sequence[Dict[str, str]],
    layer_filter: List[int] | None,
) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        layer = int(row["layer"])
        if layer_filter is not None and layer not in layer_filter:
            continue
        grouped[layer].append(row)
    if not grouped:
        available = sorted({int(row["layer"]) for row in rows})
        requested = layer_filter if layer_filter is not None else available
        raise ValueError(
            f"No rows matched the requested layer filter. Requested: {requested}. "
            f"Available in CSV: {available}"
        )
    return dict(grouped)


def split_by_score_sign(
    layer_rows: Sequence[Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    positive_x: List[float] = []
    negative_x: List[float] = []
    for row in layer_rows:
        cosine = float(row["cosine_similarity"])
        if float(row["score_diff"]) >= 0.0:
            positive_x.append(cosine)
        else:
            negative_x.append(cosine)
    return np.array(positive_x, dtype=np.float64), np.array(negative_x, dtype=np.float64)


def jitter_y(count: int, half_width: float, rng: np.random.Generator) -> np.ndarray:
    if count == 0:
        return np.array([], dtype=np.float64)
    return rng.uniform(-half_width, half_width, size=count)


def grid_shape(num_panels: int, grid_cols: Optional[int] = None) -> Tuple[int, int]:
    if grid_cols is not None and grid_cols > 0:
        cols = grid_cols
        rows = math.ceil(num_panels / cols)
        return rows, cols
    cols = math.ceil(math.sqrt(num_panels))
    rows = math.ceil(num_panels / cols)
    return rows, cols


def scatter_on_axis(
    ax: plt.Axes,
    layer_rows: Sequence[Dict[str, str]],
    jitter_half_width: float,
    point_size: float,
    alpha: float,
    rng: np.random.Generator,
    *,
    show_legend: bool = False,
    title: Optional[str] = None,
) -> None:
    pos_x, neg_x = split_by_score_sign(layer_rows)
    pos_y = jitter_y(len(pos_x), jitter_half_width, rng)
    neg_y = jitter_y(len(neg_x), jitter_half_width, rng)

    if len(pos_x):
        ax.scatter(
            pos_x,
            pos_y,
            marker="o",
            s=point_size,
            alpha=alpha,
            color="#2CA02C",
            linewidths=0.4,
            edgecolors="white",
            label=f"score_diff >= 0 (n={len(pos_x)})",
        )
    if len(neg_x):
        ax.scatter(
            neg_x,
            neg_y,
            marker="x",
            s=point_size,
            alpha=alpha,
            color="#D62728",
            linewidths=0.9,
            label=f"score_diff < 0 (n={len(neg_x)})",
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-jitter_half_width * 1.4, jitter_half_width * 1.4)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def plot_title_from_row(row: Dict[str, str], layer: int) -> str:
    model = row.get("model", "unknown")
    benchmark = row.get("benchmark", "unknown")
    split = row.get("split", "unknown")
    fold = row.get("fold", "unknown")
    return f"{model} {benchmark} {split} fold {fold} layer {layer}"


def plot_layer_scatter(
    layer: int,
    layer_rows: List[Dict[str, str]],
    out_dir: Path,
    jitter_half_width: float,
    point_size: float,
    alpha: float,
    rng: np.random.Generator,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    scatter_on_axis(
        ax,
        layer_rows,
        jitter_half_width,
        point_size,
        alpha,
        rng,
        show_legend=True,
        title=plot_title_from_row(layer_rows[0], layer),
    )
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Jittered position")
    fig.tight_layout()

    fig_path = out_dir / f"layer_{layer}_cosine_score_diff_scatter.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path


def plot_layer_cross_fold_grid(
    layer: int,
    fold_to_rows: Dict[int, List[Dict[str, str]]],
    folds: Sequence[int],
    out_dir: Path,
    jitter_half_width: float,
    point_size: float,
    alpha: float,
    rng: np.random.Generator,
    grid_cols: Optional[int],
    metadata_row: Dict[str, str],
) -> Path:
    rows, cols = grid_shape(len(folds), grid_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.8 * rows), squeeze=False)
    model = metadata_row.get("model", "unknown")
    benchmark = metadata_row.get("benchmark", "unknown")

    for idx, fold in enumerate(folds):
        ax = axes[idx // cols][idx % cols]
        fold_rows = fold_to_rows.get(fold, [])
        if fold_rows:
            scatter_on_axis(
                ax,
                fold_rows,
                jitter_half_width,
                point_size,
                alpha,
                rng,
                title=f"fold {fold}",
            )
        else:
            ax.set_title(f"fold {fold}\n(no data in CSV)", fontsize=8)
            ax.axis("off")

    for idx in range(len(folds), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"{model} {benchmark} layer {layer} — cosine vs score_diff by fold", fontsize=11, y=1.04)
    fig.supxlabel("Cosine similarity")
    fig.tight_layout()

    fig_path = out_dir / f"layer_{layer}_cosine_score_diff_scatter_cross_fold.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def load_cross_fold_data(
    score_diff_root: Path,
    folds: Sequence[int],
) -> Tuple[Dict[int, List[Dict[str, str]]], Dict[str, str]]:
    fold_rows: Dict[int, List[Dict[str, str]]] = {}
    metadata_row: Optional[Dict[str, str]] = None
    for fold in folds:
        csv_path = find_fold_csv(score_diff_root, fold)
        rows = load_csv_rows(csv_path)
        fold_rows[fold] = rows
        if metadata_row is None:
            metadata_row = rows[0]
    if metadata_row is None:
        raise ValueError("No fold CSV data loaded")
    return fold_rows, metadata_row


def run_single_mode(args: argparse.Namespace, rng: np.random.Generator) -> None:
    if args.csv is None:
        raise ValueError("--csv is required for --mode single")
    csv_path = args.csv.resolve()
    out_dir = (args.output_dir or csv_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(csv_path)
    layer_filter = resolve_layer_filter(
        rows,
        args.layers,
        args.results_root.resolve(),
        args.percentile,
    )
    if layer_filter is not None:
        logger.info("Plotting layers: %s", layer_filter)
    grouped = group_rows_by_layer(rows, layer_filter)

    for layer in sorted(grouped):
        fig_path = plot_layer_scatter(
            layer=layer,
            layer_rows=grouped[layer],
            out_dir=out_dir,
            jitter_half_width=args.jitter,
            point_size=args.point_size,
            alpha=args.alpha,
            rng=rng,
        )
        logger.info("Wrote layer %s scatter plot to %s", layer, fig_path)

    logger.info("Wrote %s layer scatter plot(s) to %s", len(grouped), out_dir)


def resolve_layers_union_across_folds(
    fold_rows: Dict[int, List[Dict[str, str]]],
    folds: Sequence[int],
    layers_arg: str,
    results_root: Path,
    percentile: float,
) -> List[int]:
    if layers_arg == "all":
        return sorted({int(row["layer"]) for rows in fold_rows.values() for row in rows})

    union: set[int] = set()
    for fold in folds:
        per_fold_layers = resolve_layer_filter(
            fold_rows[fold],
            layers_arg,
            results_root,
            percentile,
        )
        if per_fold_layers is None:
            return sorted({int(row["layer"]) for rows in fold_rows.values() for row in rows})
        union.update(per_fold_layers)
    return sorted(union)


def run_cross_fold_mode(args: argparse.Namespace, rng: np.random.Generator) -> None:
    if args.score_diff_root is None:
        raise ValueError("--score-diff-root is required for --mode cross-fold")
    score_diff_root = args.score_diff_root.resolve()
    out_dir = (args.output_dir or score_diff_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, metadata_row = load_cross_fold_data(score_diff_root, args.folds)
    layers = resolve_layers_union_across_folds(
        fold_rows,
        args.folds,
        args.layers,
        args.results_root.resolve(),
        args.percentile,
    )
    logger.info("Plotting layers (union across folds): %s", layers)

    for layer in layers:
        fold_to_layer_rows: Dict[int, List[Dict[str, str]]] = {}
        for fold, rows in fold_rows.items():
            fold_to_layer_rows[fold] = [row for row in rows if int(row["layer"]) == layer]
        fig_path = plot_layer_cross_fold_grid(
            layer=layer,
            fold_to_rows=fold_to_layer_rows,
            folds=args.folds,
            out_dir=out_dir,
            jitter_half_width=args.jitter,
            point_size=args.point_size,
            alpha=args.alpha,
            rng=rng,
            grid_cols=args.grid_cols,
            metadata_row=metadata_row,
        )
        logger.info("Wrote cross-fold grid for layer %s to %s", layer, fig_path)

    logger.info("Wrote %s cross-fold layer grid(s) to %s", len(layers), out_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(args.seed)

    if args.mode == "single":
        run_single_mode(args, rng)
    else:
        run_cross_fold_mode(args, rng)


if __name__ == "__main__":
    main()
