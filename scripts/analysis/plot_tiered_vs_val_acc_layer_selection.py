#!/usr/bin/env python3
"""Bar plot comparing tiered vs val-acc layer selection test one-of-4 accuracy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tiered vs val-acc layer selection comparison from CSV."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more tiered_vs_val_acc_layer_selection.csv paths.",
    )
    parser.add_argument(
        "--metric",
        default="test_one_of_4",
        choices=("test_one_of_4", "val_one_of_4", "val_pairwise_acc"),
        help="Metric to plot (default: test_one_of_4).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: alongside first CSV).",
    )
    parser.add_argument(
        "--per-fold",
        action="store_true",
        help="Plot per-fold grouped bars instead of fold-averaged summary.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def model_label(row: Dict[str, str]) -> str:
    model = row["model"].split("/")[-1]
    source = row["dataset_source"]
    return f"{model}\n({source})"


def metric_columns(metric: str) -> tuple[str, str]:
    if metric == "test_one_of_4":
        return (
            "val_acc_selected_test_one_of_4",
            "tiered_selected_test_one_of_4",
        )
    if metric == "val_one_of_4":
        return (
            "val_acc_selected_val_one_of_4",
            "tiered_selected_val_one_of_4",
        )
    return (
        "val_acc_selected_val_pairwise_acc",
        "tiered_selected_val_pairwise_acc",
    )


def plot_per_fold(
    rows: List[Dict[str, str]],
    label: str,
    metric: str,
    output_path: Path,
) -> None:
    val_col, tiered_col = metric_columns(metric)
    metric_title = metric.replace("_", " ")
    folds = sorted(int(r["fold"]) for r in rows)
    row_by_fold = {int(r["fold"]): r for r in rows}

    x = np.arange(len(folds))
    width = 0.35
    val_vals = [float(row_by_fold[f][val_col]) for f in folds]
    tiered_vals = [float(row_by_fold[f][tiered_col]) for f in folds]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, val_vals, width, label="Val-acc selection", color=colors[0])
    ax.bar(x + width / 2, tiered_vals, width, label="Tiered selection", color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_title)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{label}: {metric_title} by fold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comparison(
    datasets: List[tuple[str, List[Dict[str, str]]]],
    metric: str,
    output_path: Path,
) -> None:
    val_col, tiered_col = metric_columns(metric)
    metric_title = metric.replace("_", " ")

    n_models = len(datasets)
    fig, ax = plt.subplots(figsize=(max(8, 3 * n_models), 6))

    x = np.arange(n_models)
    width = 0.35
    val_means: List[float] = []
    tiered_means: List[float] = []
    val_stds: List[float] = []
    tiered_stds: List[float] = []
    labels: List[str] = []
    n_same: List[int] = []
    n_total: List[int] = []

    for label, rows in datasets:
        labels.append(label)
        val_vals = [float(r[val_col]) for r in rows]
        tiered_vals = [float(r[tiered_col]) for r in rows]
        val_means.append(mean(val_vals))
        tiered_means.append(mean(tiered_vals))
        val_stds.append(float(np.std(val_vals, ddof=1)) if len(val_vals) > 1 else 0.0)
        tiered_stds.append(
            float(np.std(tiered_vals, ddof=1)) if len(tiered_vals) > 1 else 0.0
        )
        n_same.append(sum(1 for r in rows if r["same_selected_layer"] == "True"))
        n_total.append(len(rows))

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    c_val = colors[0] if colors else "#4C72B0"
    c_tiered = colors[1] if len(colors) > 1 else "#DD8452"

    bars_val = ax.bar(
        x - width / 2,
        val_means,
        width,
        yerr=val_stds,
        label="Val-acc selection",
        color=c_val,
        capsize=4,
    )
    bars_tiered = ax.bar(
        x + width / 2,
        tiered_means,
        width,
        yerr=tiered_stds,
        label="Tiered selection",
        color=c_tiered,
        capsize=4,
    )

    ymax = max(
        max(v + s for v, s in zip(val_means, val_stds)),
        max(v + s for v, s in zip(tiered_means, tiered_stds)),
    )
    ax.set_ylim(0, min(1.0, ymax * 1.15 + 0.02))

    for bar_group, means in ((bars_val, val_means), (bars_tiered, tiered_means)):
        for bar, val in zip(bar_group, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric_title)
    ax.set_title(f"Layer selection comparison: {metric_title} (mean ± std over folds)")
    ax.legend(loc="upper right")

    for i, (ns, nt) in enumerate(zip(n_same, n_total)):
        ax.text(
            x[i],
            0.02,
            f"same layer: {ns}/{nt}",
            ha="center",
            va="bottom",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    datasets: List[tuple[str, List[Dict[str, str]]]] = []
    for csv_path in args.csv:
        path = Path(csv_path)
        rows = load_csv(path)
        if not rows:
            raise SystemExit(f"No rows in {path}")
        datasets.append((model_label(rows[0]), rows))

    if args.per_fold:
        if len(datasets) != 1:
            raise SystemExit("--per-fold supports exactly one --csv path.")
        label, rows = datasets[0]
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = (
                Path(args.csv[0]).parent
                / "tiered_vs_val_acc_layer_selection_bar_per_fold.png"
            )
        plot_per_fold(rows, label, args.metric, output_path)
        print(f"Wrote {output_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.csv[0]).parent / "tiered_vs_val_acc_layer_selection_bar.png"

    plot_comparison(datasets, args.metric, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
