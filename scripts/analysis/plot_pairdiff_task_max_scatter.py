#!/usr/bin/env python3
"""Scatter plots of cosine similarity vs. task-level score ranking.

Purpose
-------
Visualizes ``pairdiff_task_max_cosine.csv`` as jittered scatter plots. Point shape
encodes whether the correct implementation has the **highest projection score among
all task implementations** (not just the paired incorrect sample):

- circle (o): correct_highest_in_task = 1
- x: any incorrect beats the correct sample

All pairs from the same task share the same marker.

Inputs
------
- ``pairdiff_task_max_cosine.csv`` produced by ``plot_pairdiff_task_max_score_diff.py``
- ``results.json`` (when filtering layers via ``--layers selected`` or top-percentile)

Outputs
-------
- Single mode: ``layer_{L}_cosine_task_max_scatter.png`` per layer
- Cross-fold mode: ``layer_{L}_cosine_task_max_scatter_cross_fold.png`` (subplot per fold)

Key CLI flags
-------------
- ``--mode``: ``single`` or ``cross-fold``
- ``--task-max-root``: parent dir with ``fold_*/.../pairdiff_task_max_cosine.csv``
- ``--layers``: layer filter (default in batch script: top-percentile union across folds)

Related scripts
---------------
- ``plot_pairdiff_task_max_score_diff.py``: prerequisite CSV generation
- ``run_pairdiff_task_max_scatter.sh``: batch pipeline
- ``plot_pairdiff_score_diff_scatter.py``: pairwise score_diff scatter variant

Usage
-----
::

    python scripts/analysis/plot_pairdiff_task_max_scatter.py \\
        --mode cross-fold \\
        --task-max-root outputs/phase2_task2_task_max/top_percentile/MODEL/.../test \\
        --layers top-percentile
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_pairdiff_score_diff_scatter import (
    grid_shape,
    group_rows_by_layer,
    jitter_y,
    plot_title_from_row,
    resolve_layer_filter,
    resolve_layers_union_across_folds,
)
import csv

logger = logging.getLogger(__name__)

CSV_FILENAME = "pairdiff_task_max_cosine.csv"
WIN_COLUMN = "correct_highest_in_task"
REQUIRED_COLUMNS = {
    "layer",
    "cosine_similarity",
    WIN_COLUMN,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter-plot cosine similarity per layer from pairdiff_task_max_cosine.csv."
    )
    parser.add_argument(
        "--mode",
        choices=("single", "cross-fold"),
        default="single",
        help="single: one CSV and one PNG per layer; cross-fold: grid of folds per layer",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Path to pairdiff_task_max_cosine.csv")
    parser.add_argument(
        "--task-max-root",
        type=Path,
        default=None,
        help="Root directory containing fold_*/.../pairdiff_task_max_cosine.csv (cross-fold mode)",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--layers",
        default="all",
        help="all, selected, top-percentile, or comma-separated layer indices",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--grid-cols", type=int, default=None)
    parser.add_argument("--jitter", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise KeyError(f"CSV missing required columns: {sorted(missing)}")
    return rows


def find_fold_task_max_csv(task_max_root, fold: int):
    matches = sorted(task_max_root.glob(f"fold_{fold}/**/{CSV_FILENAME}"))
    if not matches:
        raise FileNotFoundError(
            f"No {CSV_FILENAME} found under {task_max_root}/fold_{fold}/"
        )
    if len(matches) > 1:
        logger.warning("Multiple CSV files for fold %s; using %s", fold, matches[0])
    return matches[0]


def split_by_task_max(
    layer_rows: Sequence[Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    win_x: List[float] = []
    lose_x: List[float] = []
    for row in layer_rows:
        cosine = float(row["cosine_similarity"])
        if int(float(row[WIN_COLUMN])) == 1:
            win_x.append(cosine)
        else:
            lose_x.append(cosine)
    return np.array(win_x, dtype=np.float64), np.array(lose_x, dtype=np.float64)


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
    win_x, lose_x = split_by_task_max(layer_rows)
    win_y = jitter_y(len(win_x), jitter_half_width, rng)
    lose_y = jitter_y(len(lose_x), jitter_half_width, rng)

    if len(win_x):
        ax.scatter(
            win_x,
            win_y,
            marker="o",
            s=point_size,
            alpha=alpha,
            color="#2CA02C",
            linewidths=0.4,
            edgecolors="white",
            label=f"correct highest in task (n={len(win_x)})",
        )
    if len(lose_x):
        ax.scatter(
            lose_x,
            lose_y,
            marker="x",
            s=point_size,
            alpha=alpha,
            color="#D62728",
            linewidths=0.9,
            label=f"incorrect beats correct (n={len(lose_x)})",
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-jitter_half_width * 1.4, jitter_half_width * 1.4)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def plot_layer_scatter(
    layer: int,
    layer_rows: List[Dict[str, str]],
    out_dir,
    jitter_half_width: float,
    point_size: float,
    alpha: float,
    rng: np.random.Generator,
):
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
    fig_path = out_dir / f"layer_{layer}_cosine_task_max_scatter.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path


def plot_layer_cross_fold_grid(
    layer: int,
    fold_to_rows: Dict[int, List[Dict[str, str]]],
    folds: Sequence[int],
    out_dir,
    jitter_half_width: float,
    point_size: float,
    alpha: float,
    rng: np.random.Generator,
    grid_cols: Optional[int],
    metadata_row: Dict[str, str],
):
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

    fig.suptitle(
        f"{model} {benchmark} layer {layer} — cosine vs task-max score ranking by fold",
        fontsize=11,
        y=1.04,
    )
    fig.supxlabel("Cosine similarity")
    fig.tight_layout()

    fig_path = out_dir / f"layer_{layer}_cosine_task_max_scatter_cross_fold.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def load_cross_fold_task_max_data(task_max_root, folds: Sequence[int]):
    fold_rows: Dict[int, List[Dict[str, str]]] = {}
    metadata_row: Optional[Dict[str, str]] = None
    for fold in folds:
        csv_path = find_fold_task_max_csv(task_max_root, fold)
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


def run_cross_fold_mode(args: argparse.Namespace, rng: np.random.Generator) -> None:
    if args.task_max_root is None:
        raise ValueError("--task-max-root is required for --mode cross-fold")
    task_max_root = args.task_max_root.resolve()
    out_dir = (args.output_dir or task_max_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, metadata_row = load_cross_fold_task_max_data(task_max_root, args.folds)
    layers = resolve_layers_union_across_folds(
        fold_rows,
        args.folds,
        args.layers,
        args.results_root.resolve(),
        args.percentile,
    )
    logger.info("Plotting layers (union across folds): %s", layers)

    for layer in layers:
        fold_to_layer_rows = {
            fold: [row for row in rows if int(row["layer"]) == layer]
            for fold, rows in fold_rows.items()
        }
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
