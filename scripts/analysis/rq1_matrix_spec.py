"""Shared RQ1 experiment matrix definition and name mappings."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List

METHODS = ["pca", "mean_diff"]
REPRESENTATIONS = ["seq_last", "response_avg"]
FIT_SOURCES = ["id", "mbpp", "syn"]
TEST_DATASETS = ["humaneval", "bigcodebench"]
MODELS = [
    "mistralai__Mistral-7B-Instruct-v0.3",
    "Qwen__Qwen2.5-Coder-7B-Instruct",
    "codellama__CodeLlama-7b-Instruct-hf",
    "infly__OpenCoder-8B-Instruct",
]

FRAMING_MODES = [
    ("N->C", "neutral", "neutral", "concept"),
    ("C->C", "concept", "concept", "concept"),
    ("F->N", "function", "neutral", "neutral"),
    ("F->C", "function", "concept", "concept"),
]

MATRIX_FIELDNAMES = [
    "ConfigID", "Method", "Representation", "FramingKey",
    "FramingFit", "FramingVal", "FramingTest",
    "FitSource", "TestDataset", "Model",
    "avg", "std-dev", "Status", "Notes",
]

RESULT_FIELDNAMES = [
    "avg_one_of_4", "stddev_one_of_4", "avg_pairwise", "stddev_pairwise",
]

OUTPUT_FIELDNAMES = MATRIX_FIELDNAMES + RESULT_FIELDNAMES

FIT_SOURCE_TO_RUNNER: Dict[str, str] = {
    "id": "in-domain",
    "mbpp": "mbpp",
    "syn": "synthetic",
}

FIT_SOURCE_FROM_RUNNER: Dict[str, str] = {v: k for k, v in FIT_SOURCE_TO_RUNNER.items()}

REPRESENTATION_TO_RUNNER: Dict[str, str] = {
    "seq_last": "sequence_last",
    "response_avg": "response_avg",
}

REPRESENTATION_FROM_RUNNER: Dict[str, str] = {
    v: k for k, v in REPRESENTATION_TO_RUNNER.items()
}

MODEL_TO_RUNNER: Dict[str, str] = {
    name: name.replace("__", "/") for name in MODELS
}

MODEL_FROM_RUNNER: Dict[str, str] = {v: k for k, v in MODEL_TO_RUNNER.items()}

# Keys that may appear under bundle.acts / metadata.json "representations".
BUNDLE_REPRESENTATION_KEYS: Dict[str, tuple] = {
    "seq_last": ("sequence_last", "seq_last"),
    "response_avg": ("response_avg",),
}


def bundle_has_representation(meta_or_acts_keys: List[str], matrix_rep: str) -> bool:
    """Return True if a cached bundle exposes `matrix_rep` activations."""
    keys = set(meta_or_acts_keys)
    return any(k in keys for k in BUNDLE_REPRESENTATION_KEYS[matrix_rep])


def representations_in_bundle_dir(bundle_dir: Path) -> List[str]:
    """Read representation names from metadata.json, falling back to bundle.pt."""
    meta_path = bundle_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        reps = meta.get("representations")
        if reps:
            return list(reps)
    bundle_path = bundle_dir / "bundle.pt"
    if bundle_path.is_file():
        from src.hidden_states import HiddenStateBundle
        return list(HiddenStateBundle.load(bundle_path).acts.keys())
    return []


def row_split_family(fit_source: str) -> str:
    if fit_source == "id":
        return "icse26"
    return f"ood_{FIT_SOURCE_TO_RUNNER[fit_source]}"


def row_required_bundle_dirs(
    row: dict,
    hidden_states_root: Path,
    fold: int = 0,
) -> List[tuple]:
    """(label, bundle_dir) pairs that must expose the row's representation."""
    from src.hidden_states import bundle_output_dir

    fit_source = row["FitSource"]
    split_family = row_split_family(fit_source)
    model = MODEL_TO_RUNNER[row["Model"]]
    benchmark = row["TestDataset"]
    fit_prefix = "inner" if fit_source != "id" else "fold"
    test_prefix = "fold"

    dirs = [
        ("fit", row["FramingFit"], fit_prefix),
        ("validate", row["FramingVal"], fit_prefix),
        ("test", row["FramingTest"], test_prefix),
    ]
    out: List[tuple] = []
    for data_split, framing, fold_prefix in dirs:
        out.append((
            data_split,
            bundle_output_dir(
                output_root=hidden_states_root,
                model_name=model,
                split_family=split_family,
                benchmark=benchmark,
                distribution="id",
                data_split=data_split,
                framing=framing,
                fold=fold,
                fold_prefix=fold_prefix,
            ),
        ))
    return out


def row_fit_bundle_dir(
    row: dict,
    hidden_states_root: Path,
    fold: int = 0,
) -> Path:
    """Path to the fit-split cache cell (convenience wrapper)."""
    return row_required_bundle_dirs(row, hidden_states_root, fold=fold)[0][1]


def is_prefilled(method: str, representation: str, framing_key: str) -> bool:
    return (
        (method == "pca" and representation == "seq_last" and framing_key == "N->C")
        or (method == "mean_diff" and representation == "response_avg" and framing_key == "F->N")
    )


def iter_rq1_rows() -> Iterator[dict]:
    """Yield matrix rows in the exact order used by generate_rq1_matrix.py."""
    configs: List[dict] = []
    config_id = 1
    for method, rep, (key, fit_f, val_f, test_f) in product(
        METHODS, REPRESENTATIONS, FRAMING_MODES
    ):
        configs.append({
            "ConfigID": f"C{config_id}",
            "Method": method,
            "Representation": rep,
            "FramingKey": key,
            "FramingFit": fit_f,
            "FramingVal": val_f,
            "FramingTest": test_f,
        })
        config_id += 1

    for cfg in configs:
        for fit_source, test_dataset, model in product(FIT_SOURCES, TEST_DATASETS, MODELS):
            yield {
                "ConfigID": cfg["ConfigID"],
                "Method": cfg["Method"],
                "Representation": cfg["Representation"],
                "FramingKey": cfg["FramingKey"],
                "FramingFit": cfg["FramingFit"],
                "FramingVal": cfg["FramingVal"],
                "FramingTest": cfg["FramingTest"],
                "FitSource": fit_source,
                "TestDataset": test_dataset,
                "Model": model,
                "avg": "",
                "std-dev": "",
                "Status": "pre-filled" if is_prefilled(
                    cfg["Method"], cfg["Representation"], cfg["FramingKey"]
                ) else "pending",
                "Notes": "",
            }


def row_match_key(row: dict) -> tuple:
    return (
        row["Model"],
        row["FitSource"],
        row["TestDataset"],
        row["Method"],
        row["Representation"],
        row["FramingFit"],
        row["FramingVal"],
        row["FramingTest"],
    )


def cell_match_key(cell: dict) -> tuple:
    return (
        MODEL_FROM_RUNNER[cell["model"]],
        FIT_SOURCE_FROM_RUNNER[cell["fit_source"]],
        cell["test_dataset"],
        cell["method"],
        REPRESENTATION_FROM_RUNNER[cell["representation"]],
        cell["fit_framing"],
        cell["val_framing"],
        cell["test_framing"],
    )
