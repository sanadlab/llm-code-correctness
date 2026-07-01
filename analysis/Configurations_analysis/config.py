"""Shared constants for RQ1 configuration analysis."""

from __future__ import annotations

from pathlib import Path

# Default paths (relative to llm-code-correctness repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = REPO_ROOT / "analysis" / "ablation" / "rq1_results_with_best_layer.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# CSV column names.
COL_SELECTED_AVG = "avg"
COL_SELECTED_STD = "std-dev"
COL_BEST_AVG = "avg (best, 1of4)"
COL_BEST_STD = "std-dev (best, 1of4)"

# Configuration dimensions (from rq1_matrix_spec).
FACTOR_COLUMNS = [
    "Method",
    "Representation",
    "FramingKey",
    "FitSource",
    "TestDataset",
    "Model",
]

CONFIG_ID_COL = "ConfigID"

# Human-readable labels for plots and tables.
METHOD_LABELS = {"pca": "PCA", "mean_diff": "mean_diff"}
REPRESENTATION_LABELS = {"seq_last": "seq_last", "response_avg": "response_avg"}
FIT_SOURCE_LABELS = {"id": "id", "mbpp": "mbpp", "syn": "synthetic"}
TEST_DATASET_LABELS = {"humaneval": "HumanEval", "bigcodebench": "BigCodeBench"}
MODEL_LABELS = {
    "mistralai__Mistral-7B-Instruct-v0.3": "Mistral",
    "Qwen__Qwen2.5-Coder-7B-Instruct": "Qwen",
    "codellama__CodeLlama-7b-Instruct-hf": "CodeLlama",
    "infly__OpenCoder-8B-Instruct": "OpenCoder",
}

# Heatmap factor pairs requested for cross-tab analysis.
HEATMAP_PAIRS = [
    ("Method", "Representation"),
    ("FramingKey", "Method"),
    ("Model", "Method"),
    ("FitSource", "TestDataset"),
]

# Single-factor columns for boxplots and univariate summaries.
UNIVARIATE_FACTORS = list(FACTOR_COLUMNS)

# Factor combinations for grouped summary tables.
SUMMARY_FACTOR_COMBOS = [
    ["Method"],
    ["Representation"],
    ["FramingKey"],
    ["FitSource"],
    ["TestDataset"],
    ["Model"],
    ["Method", "Representation"],
    ["FramingKey", "Method"],
    ["Model", "Method"],
    ["FitSource", "TestDataset"],
    ["Method", "Representation", "FramingKey"],
]

TOP_N = 10
