"""Ranking configurations by accuracy, gap, and stability."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .config import COL_BEST_AVG, COL_SELECTED_AVG, CONFIG_ID_COL, TOP_N


def _rank_table(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool,
    label_col: str,
    rank_name: str,
    top_n: int = TOP_N,
    extra_cols: tuple[str, ...] = ("gap", "selected_std", "best_avg"),
) -> pd.DataFrame:
    ranked = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    cols = ["rank", label_col, sort_col]
    cols.extend(c for c in extra_cols if c in ranked.columns and c not in cols)
    out = ranked[cols].head(top_n if top_n else len(ranked))
    out = out.rename(columns={sort_col: rank_name, label_col: "configuration"})
    return out


def rank_configurations(
    config_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> dict[str, pd.DataFrame]:
    """Return top/bottom tables at ConfigID aggregation level."""
    label_col = "config_label"
    return {
        "top_by_selected_avg": _rank_table(
            config_df, "selected_avg", ascending=False, label_col=label_col,
            rank_name="mean_selected_avg", top_n=top_n,
        ),
        "bottom_by_selected_avg": _rank_table(
            config_df, "selected_avg", ascending=True, label_col=label_col,
            rank_name="mean_selected_avg", top_n=top_n,
        ),
        "top_by_best_avg": _rank_table(
            config_df, "best_avg", ascending=False, label_col=label_col,
            rank_name="mean_best_avg", top_n=top_n,
        ),
        "smallest_gap": _rank_table(
            config_df, "gap", ascending=True, label_col=label_col,
            rank_name="mean_gap", top_n=top_n,
        ),
        "largest_gap": _rank_table(
            config_df, "gap", ascending=False, label_col=label_col,
            rank_name="mean_gap", top_n=top_n,
        ),
        "most_stable": _rank_table(
            config_df, "selected_std", ascending=True, label_col=label_col,
            rank_name="mean_selected_std", top_n=top_n,
        ),
        "best_conservative_1std": _rank_table(
            config_df, "conservative_1std", ascending=False, label_col=label_col,
            rank_name="mean_conservative_1std", top_n=top_n,
        ),
        "best_conservative_2std": _rank_table(
            config_df, "conservative_2std", ascending=False, label_col=label_col,
            rank_name="mean_conservative_2std", top_n=top_n,
        ),
    }


def rank_rows(
    row_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> dict[str, pd.DataFrame]:
    """Return top/bottom tables at full cell level (config × model × fit × dataset)."""
    label_col = "row_label"
    return {
        "top_rows_by_selected_avg": _rank_table(
            row_df, COL_SELECTED_AVG, ascending=False, label_col=label_col,
            rank_name="selected_avg", top_n=top_n,
            extra_cols=("gap", "std-dev", COL_BEST_AVG),
        ),
        "bottom_rows_by_selected_avg": _rank_table(
            row_df, COL_SELECTED_AVG, ascending=True, label_col=label_col,
            rank_name="selected_avg", top_n=top_n,
            extra_cols=("gap", "std-dev", COL_BEST_AVG),
        ),
        "smallest_gap_rows": _rank_table(
            row_df, "gap", ascending=True, label_col=label_col,
            rank_name="gap", top_n=top_n,
        ),
        "largest_gap_rows": _rank_table(
            row_df, "gap", ascending=False, label_col=label_col,
            rank_name="gap", top_n=top_n,
        ),
    }


def write_rankings(rankings: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rankings_dir = output_dir / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    for name, table in rankings.items():
        table.to_csv(rankings_dir / f"{name}.csv", index=False)
