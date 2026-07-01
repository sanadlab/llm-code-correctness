"""Grouped summary tables for configuration factors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import COL_BEST_AVG, COL_SELECTED_AVG, COL_SELECTED_STD, SUMMARY_FACTOR_COMBOS


def _summary_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            mean_selected_avg=(COL_SELECTED_AVG, "mean"),
            mean_best_avg=(COL_BEST_AVG, "mean"),
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            max_gap=("gap", "max"),
            mean_selected_std=(COL_SELECTED_STD, "mean"),
            n_configurations=(COL_SELECTED_AVG, "count"),
        )
        .sort_values("mean_selected_avg", ascending=False)
    )
    return grouped


def build_all_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build summary tables for each factor and factor combination."""
    summaries: dict[str, pd.DataFrame] = {}
    for combo in SUMMARY_FACTOR_COMBOS:
        key = "__".join(combo)
        summaries[key] = _summary_group(df, combo)
    return summaries


def write_summaries(summaries: dict[str, pd.DataFrame], output_dir: Path) -> None:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for name, table in summaries.items():
        table.to_csv(summaries_dir / f"summary_by_{name}.csv", index=False)

    # Combined long-form index for quick lookup.
    index_rows = []
    for name, table in summaries.items():
        row = {"summary_name": name, "n_groups": len(table)}
        index_rows.append(row)
    pd.DataFrame(index_rows).to_csv(summaries_dir / "_summary_index.csv", index=False)
