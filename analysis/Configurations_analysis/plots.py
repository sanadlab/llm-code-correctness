"""Visualization helpers for RQ1 configuration analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    COL_BEST_AVG,
    COL_SELECTED_AVG,
    COL_SELECTED_STD,
    CONFIG_ID_COL,
    HEATMAP_PAIRS,
    TOP_N,
    UNIVARIATE_FACTORS,
)

# Consistent styling across all figures.
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
PALETTE = "colorblind"


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_bottom_bars(
    config_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = TOP_N,
) -> None:
    """Horizontal bar charts for top/bottom configurations by selected accuracy."""
    plots_dir = output_dir / "plots"
    for ascending, tag in ((False, "top"), (True, "bottom")):
        subset = config_df.sort_values("selected_avg", ascending=ascending).head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(subset))))
        colors = sns.color_palette(PALETTE, n_colors=1)
        ax.barh(subset["config_label"], subset["selected_avg"], color=colors[0])
        ax.set_xlabel("Mean selected-layer accuracy")
        ax.set_title(f"{tag.capitalize()} {top_n} configurations by selected-layer accuracy")
        ax.invert_yaxis()
        _savefig(fig, plots_dir / f"bar_{tag}_configurations_selected_avg.png")


def plot_all_config_ranking_bars(
    by_selected: pd.DataFrame,
    by_best: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Horizontal bar charts ranking all configurations by selected and best avg."""
    plots_dir = output_dir / "plots"
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

    for df, value_col, xlabel, title, filename in (
        (
            by_selected,
            "selected_avg",
            "Mean selected-layer accuracy",
            "All configurations ranked by selected-layer accuracy",
            "bar_all_configs_by_selected_avg.png",
        ),
        (
            by_best,
            "best_avg",
            "Mean best-layer accuracy",
            "All configurations ranked by best-layer accuracy",
            "bar_all_configs_by_best_avg.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(df))))
        color = sns.color_palette(PALETTE, n_colors=1)[0]
        ax.barh(df["config_label"], df[value_col], color=color)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.invert_yaxis()
        for i, val in enumerate(df[value_col]):
            ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)
        _savefig(fig, plots_dir / filename)


def plot_selected_vs_best_scatter(row_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of selected-layer vs actual-best-layer accuracy."""
    plots_dir = output_dir / "plots"
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        row_df[COL_SELECTED_AVG],
        row_df[COL_BEST_AVG],
        alpha=0.55,
        s=40,
        color=sns.color_palette(PALETTE, n_colors=1)[0],
        edgecolors="none",
    )
    lo = min(row_df[COL_SELECTED_AVG].min(), row_df[COL_BEST_AVG].min())
    hi = max(row_df[COL_SELECTED_AVG].max(), row_df[COL_BEST_AVG].max())
    margin = 0.02 * (hi - lo)
    lim = (lo - margin, hi + margin)
    ax.plot(lim, lim, "--", color="gray", linewidth=1, label="y = x (no gap)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Selected-layer accuracy")
    ax.set_ylabel("Actual best-layer accuracy")
    ax.set_title("Selected vs. actual best layer accuracy")
    ax.legend(loc="lower right")
    _savefig(fig, plots_dir / "scatter_selected_vs_best_accuracy.png")


def plot_gap_distribution(row_df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram and KDE of layer-selection gap."""
    plots_dir = output_dir / "plots"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(row_df["gap"], bins=25, kde=True, ax=axes[0], color=sns.color_palette(PALETTE)[0])
    axes[0].set_xlabel("Gap (best − selected)")
    axes[0].set_title("Gap histogram")

    sns.kdeplot(row_df["gap"], fill=True, ax=axes[1], color=sns.color_palette(PALETTE)[1])
    axes[1].set_xlabel("Gap (best − selected)")
    axes[1].set_title("Gap density")

    fig.suptitle("Layer-selection gap distribution", y=1.02)
    _savefig(fig, plots_dir / "gap_distribution.png")


def _heatmap_pivot(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="mean",
    )
    return pivot


def _heatmap_axis_labels(row_col: str, col_col: str, row_df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """Resolve display column names; shorten model names for Model × Method heatmap."""
    plot_df = row_df
    row_key, col_key = row_col, col_col
    if row_col == "Model" and "Model_short" in row_df.columns:
        plot_df = row_df
        row_key = "Model_short"
    if col_col == "Model" and "Model_short" in row_df.columns:
        plot_df = row_df
        col_key = "Model_short"
    return row_key, col_key, plot_df


def plot_heatmaps(row_df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmaps for requested factor pairs (mean selected accuracy)."""
    plots_dir = output_dir / "plots" / "heatmaps"
    for row_col, col_col in HEATMAP_PAIRS:
        row_key, col_key, plot_df = _heatmap_axis_labels(row_col, col_col, row_df)
        pivot = _heatmap_pivot(plot_df, row_key, col_key, COL_SELECTED_AVG)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * pivot.shape[1]), max(4, 0.8 * pivot.shape[0])))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Mean selected accuracy"},
        )
        row_label = "Model" if row_col == "Model" else row_col
        col_label = "Model" if col_col == "Model" else col_col
        ax.set_title(f"Mean selected accuracy: {row_label} × {col_label}")
        fname = f"heatmap_{row_col}_{col_col}.png".replace("->", "to")
        _savefig(fig, plots_dir / fname)


def plot_factor_boxplots(row_df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplots of accuracy and gap by each factor."""
    plots_dir = output_dir / "plots" / "boxplots"
    for factor in UNIVARIATE_FACTORS:
        for y, title_metric, fname_suffix in (
            (COL_SELECTED_AVG, "selected-layer accuracy", "selected_avg"),
            ("gap", "layer-selection gap", "gap"),
        ):
            fig, ax = plt.subplots(figsize=(max(6, 0.9 * row_df[factor].nunique()), 5))
            order = (
                row_df.groupby(factor)[y]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
            sns.boxplot(
                data=row_df,
                x=factor,
                y=y,
                order=order,
                hue=factor,
                palette=PALETTE,
                dodge=False,
                legend=False,
                ax=ax,
            )
            ax.set_xlabel(factor)
            ax.set_ylabel("Accuracy" if y == COL_SELECTED_AVG else "Gap")
            ax.set_title(f"{title_metric.capitalize()} by {factor}")
            ax.tick_params(axis="x", rotation=30)
            _savefig(fig, plots_dir / f"boxplot_{factor}_{fname_suffix}.png")


def plot_accuracy_vs_stability(row_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter of selected accuracy vs standard deviation."""
    plots_dir = output_dir / "plots"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        row_df[COL_SELECTED_AVG],
        row_df[COL_SELECTED_STD],
        c=row_df["gap"],
        cmap="viridis",
        alpha=0.7,
        s=45,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Gap (best − selected)")
    ax.set_xlabel("Selected-layer accuracy")
    ax.set_ylabel("Selected-layer std-dev")
    ax.set_title("Accuracy vs. stability (color = layer-selection gap)")
    _savefig(fig, plots_dir / "scatter_accuracy_vs_std.png")


def plot_gap_by_config(config_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of mean gap per configuration, sorted."""
    plots_dir = output_dir / "plots"
    ordered = config_df.sort_values("gap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(ordered))))
    colors = sns.color_palette("RdYlGn_r", n_colors=len(ordered))
    ax.barh(ordered["config_label"], ordered["gap"], color=colors)
    ax.set_xlabel("Mean gap (best − selected)")
    ax.set_title("Layer-selection gap by configuration")
    ax.invert_yaxis()
    _savefig(fig, plots_dir / "bar_gap_by_configuration.png")


def plot_framing_summary_bars(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar plots for summary_by_FramingKey metrics."""
    plots_dir = output_dir / "plots" / "summaries"
    df = summary_df.sort_values("mean_selected_avg", ascending=False).copy()

    # Selected vs best accuracy (grouped bars).
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["mean_selected_avg"], width, label="Selected avg", color=sns.color_palette(PALETTE)[0])
    ax.bar(x + width / 2, df["mean_best_avg"], width, label="Best avg", color=sns.color_palette(PALETTE)[1])
    ax.set_xticks(x)
    ax.set_xticklabels(df["FramingKey"])
    ax.set_ylabel("Mean accuracy")
    ax.set_xlabel("Framing")
    ax.set_title("Mean accuracy by framing key")
    ax.legend()
    _savefig(fig, plots_dir / "bar_summary_framing_key_accuracy.png")

    # Gap and std-dev side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gap_order = df.sort_values("mean_gap", ascending=True)
    axes[0].barh(gap_order["FramingKey"], gap_order["mean_gap"], color=sns.color_palette("RdYlGn_r", n_colors=len(gap_order)))
    axes[0].set_xlabel("Mean gap (best − selected)")
    axes[0].set_title("Layer-selection gap by framing")
    axes[0].invert_yaxis()

    std_order = df.sort_values("mean_selected_std", ascending=True)
    axes[1].barh(std_order["FramingKey"], std_order["mean_selected_std"], color=sns.color_palette(PALETTE)[2])
    axes[1].set_xlabel("Mean selected std-dev")
    axes[1].set_title("Stability by framing")
    axes[1].invert_yaxis()
    fig.suptitle("Framing key summary", y=1.02)
    _savefig(fig, plots_dir / "bar_summary_framing_key_gap_std.png")


def plot_gap_rows_bars(
    ranking_df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    title: str,
    ascending: bool,
) -> None:
    """Horizontal bar chart for largest/smallest gap row rankings."""
    plots_dir = output_dir / "plots" / "rankings"
    label_col = "configuration" if "configuration" in ranking_df.columns else ranking_df.columns[1]
    ordered = ranking_df.sort_values("gap", ascending=ascending).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.55 * len(ordered))))
    palette = "RdYlGn_r" if not ascending else "RdYlGn"
    colors = sns.color_palette(palette, n_colors=len(ordered))
    ax.barh(ordered[label_col], ordered["gap"], color=colors)
    ax.set_xlabel("Gap (best − selected)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    for i, val in enumerate(ordered["gap"]):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)
    _savefig(fig, plots_dir / filename)


def generate_all_plots(
    row_df: pd.DataFrame,
    config_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = TOP_N,
    summaries: dict[str, pd.DataFrame] | None = None,
    row_rankings: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Generate the full visualization suite."""
    plot_top_bottom_bars(config_df, output_dir, top_n=top_n)
    plot_selected_vs_best_scatter(row_df, output_dir)
    plot_gap_distribution(row_df, output_dir)
    plot_heatmaps(row_df, output_dir)
    plot_factor_boxplots(row_df, output_dir)
    plot_accuracy_vs_stability(row_df, output_dir)
    plot_gap_by_config(config_df, output_dir)

    if summaries and "FramingKey" in summaries:
        plot_framing_summary_bars(summaries["FramingKey"], output_dir)

    if row_rankings:
        if "largest_gap_rows" in row_rankings:
            plot_gap_rows_bars(
                row_rankings["largest_gap_rows"],
                output_dir,
                filename="bar_largest_gap_rows.png",
                title="Largest layer-selection gaps (cell level)",
                ascending=False,
            )
        if "smallest_gap_rows" in row_rankings:
            plot_gap_rows_bars(
                row_rankings["smallest_gap_rows"],
                output_dir,
                filename="bar_smallest_gap_rows.png",
                title="Smallest layer-selection gaps (cell level)",
                ascending=True,
            )
