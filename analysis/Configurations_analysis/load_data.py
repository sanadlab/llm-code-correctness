"""Load and preprocess RQ1 results with best-layer columns."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    COL_BEST_AVG,
    COL_BEST_STD,
    COL_SELECTED_AVG,
    COL_SELECTED_STD,
    CONFIG_ID_COL,
    FIT_SOURCE_LABELS,
    METHOD_LABELS,
    MODEL_LABELS,
    REPRESENTATION_LABELS,
    TEST_DATASET_LABELS,
)


def _short_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load the augmented RQ1 CSV and compute derived metrics."""
    df = pd.read_csv(csv_path)

    numeric_cols = [COL_SELECTED_AVG, COL_SELECTED_STD, COL_BEST_AVG, COL_BEST_STD]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing core metrics (e.g. unmatched best-layer cells).
    df = df.dropna(subset=[COL_SELECTED_AVG, COL_BEST_AVG]).copy()

    df["gap"] = df[COL_BEST_AVG] - df[COL_SELECTED_AVG]
    df["conservative_1std"] = df[COL_SELECTED_AVG] - df[COL_SELECTED_STD]
    df["conservative_2std"] = df[COL_SELECTED_AVG] - 2 * df[COL_SELECTED_STD]

    df["Method_label"] = df["Method"].map(METHOD_LABELS).fillna(df["Method"])
    df["Representation_label"] = df["Representation"].map(REPRESENTATION_LABELS).fillna(df["Representation"])
    df["FitSource_label"] = df["FitSource"].map(FIT_SOURCE_LABELS).fillna(df["FitSource"])
    df["TestDataset_label"] = df["TestDataset"].map(TEST_DATASET_LABELS).fillna(df["TestDataset"])
    df["Model_short"] = df["Model"].map(_short_model)

    df["config_label"] = (
        df[CONFIG_ID_COL].astype(str)
        + " | "
        + df["Method_label"]
        + " / "
        + df["Representation_label"]
        + " / "
        + df["FramingKey"]
    )

    df["row_label"] = (
        df["config_label"]
        + " | "
        + df["FitSource_label"]
        + " / "
        + df["TestDataset_label"]
        + " / "
        + df["Model_short"]
    )

    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level results to one row per ConfigID."""
    agg = (
        df.groupby(CONFIG_ID_COL, as_index=False)
        .agg(
            Method=("Method", "first"),
            Representation=("Representation", "first"),
            FramingKey=("FramingKey", "first"),
            config_label=("config_label", "first"),
            selected_avg=(COL_SELECTED_AVG, "mean"),
            selected_std=(COL_SELECTED_STD, "mean"),
            best_avg=(COL_BEST_AVG, "mean"),
            best_std=(COL_BEST_STD, "mean"),
            gap=("gap", "mean"),
            conservative_1std=("conservative_1std", "mean"),
            conservative_2std=("conservative_2std", "mean"),
            n_rows=(COL_SELECTED_AVG, "count"),
        )
    )
    return agg.sort_values("selected_avg", ascending=False).reset_index(drop=True)


def config_cell_label(row: pd.Series, level: str = "config") -> str:
    """Return a display label for a config or row."""
    if level == "row":
        return str(row.get("row_label", row.get(CONFIG_ID_COL, "")))
    return str(row.get("config_label", row.get(CONFIG_ID_COL, "")))
