"""Rank all configurations by selected-layer and best-layer mean accuracy.

Aggregates cell-level results to one row per ConfigID, then produces full
rankings (not truncated to top-N) sorted by:
  1. mean selected-layer accuracy (avg)
  2. mean actual-best-layer accuracy (avg best, 1of4)

Usage (from repo root):
    python -m analysis.Configurations_analysis.rank_all_configurations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR
from .load_data import aggregate_by_config, load_results
from .plots import plot_all_config_ranking_bars

OUTPUT_COLUMNS = [
    "ConfigID",
    "Method",
    "Representation",
    "FramingKey",
    "config_label",
    "selected_avg",
    "best_avg",
    "gap",
    "selected_std",
    "best_std",
    "n_rows",
]


def rank_all_configurations(config_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return full rankings by selected avg, by best avg, and a combined table."""
    base = config_df[OUTPUT_COLUMNS].copy()

    by_selected = base.sort_values("selected_avg", ascending=False).reset_index(drop=True)
    by_selected.insert(0, "rank_by_selected_avg", range(1, len(by_selected) + 1))

    by_best = base.sort_values("best_avg", ascending=False).reset_index(drop=True)
    by_best.insert(0, "rank_by_best_avg", range(1, len(by_best) + 1))

    combined = base.copy()
    combined["rank_by_selected_avg"] = (
        combined["selected_avg"].rank(method="min", ascending=False).astype(int)
    )
    combined["rank_by_best_avg"] = (
        combined["best_avg"].rank(method="min", ascending=False).astype(int)
    )
    combined = combined.sort_values("rank_by_selected_avg").reset_index(drop=True)

    return by_selected, by_best, combined


def _print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    display_cols = [
        c for c in df.columns
        if c in {
            "rank_by_selected_avg", "rank_by_best_avg", "ConfigID", "config_label",
            "selected_avg", "best_avg", "gap",
        }
    ]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank all configurations by selected avg and best avg.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to rq1_results_with_best_layer.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ranking CSV and plot outputs",
    )
    args = parser.parse_args()

    row_df = load_results(args.input_csv)
    config_df = aggregate_by_config(row_df)

    by_selected, by_best, combined = rank_all_configurations(config_df)

    rankings_dir = args.output_dir / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    by_selected.to_csv(rankings_dir / "all_configs_by_selected_avg.csv", index=False)
    by_best.to_csv(rankings_dir / "all_configs_by_best_avg.csv", index=False)
    combined.to_csv(rankings_dir / "all_configs_combined_ranks.csv", index=False)

    plot_all_config_ranking_bars(by_selected, by_best, args.output_dir)

    _print_table(by_selected, "All configurations ranked by selected-layer avg")
    _print_table(by_best, "All configurations ranked by best-layer avg")
    _print_table(combined, "Combined ranks (sorted by selected-layer rank)")

    print(f"\nWrote CSVs to {rankings_dir.resolve()}")
    print(f"Wrote bar charts to {(args.output_dir / 'plots').resolve()}")


if __name__ == "__main__":
    main()
