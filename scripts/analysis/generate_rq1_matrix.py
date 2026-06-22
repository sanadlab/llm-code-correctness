#!/usr/bin/env python3
"""
Generate RQ1 experiment matrix CSV with all combinations.

Usage:
    python generate_rq1_matrix.py > matrix.csv
    python generate_rq1_matrix.py | python csv_to_sheets.py /dev/stdin

Outputs tab-separated values to stdout.
"""

import csv
import sys
from itertools import product

methods = ["pca", "mean_diff"]
representations = ["seq_last", "response_avg"]
fit_sources = ["id", "mbpp", "syn"]
test_datasets = ["humaneval", "bigcodebench"]
models = [
    "mistralai__Mistral-7B-Instruct-v0.3",
    "Qwen__Qwen2.5-Coder-7B-Instruct",
    "codellama__CodeLlama-7b-Instruct-hf",
    "infly__OpenCoder-8B-Instruct",
]

framing_modes = [
    ("N->C", "neutral", "neutral", "concept"),
    ("C->C", "concept", "concept", "concept"),
    ("F->N", "function", "neutral", "neutral"),
    ("F->C", "function", "concept", "concept"),
]

configs = []
config_id = 1
for method, rep, (key, fit_f, val_f, test_f) in product(methods, representations, framing_modes):
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

rows = []
for cfg in configs:
    for fit_source, test_dataset, model in product(fit_sources, test_datasets, models):
        method = cfg["Method"]
        rep = cfg["Representation"]
        key = cfg["FramingKey"]
        
        # Mark pre-filled status for specific slices
        is_prefilled = (method == 'pca' and rep == 'seq_last' and key == 'N->C') or \
                       (method == 'mean_diff' and rep == 'response_avg' and key == 'F->N')
        
        rows.append({
            "ConfigID": cfg["ConfigID"],
            "Method": method,
            "Representation": rep,
            "FramingKey": key,
            "FramingFit": cfg["FramingFit"],
            "FramingVal": cfg["FramingVal"],
            "FramingTest": cfg["FramingTest"],
            "FitSource": fit_source,
            "TestDataset": test_dataset,
            "Model": model,
            "avg": "",
            "std-dev": "",
            "Status": "pre-filled" if is_prefilled else "pending",
            "Notes": "",
        })

fieldnames = [
    "ConfigID", "Method", "Representation", "FramingKey",
    "FramingFit", "FramingVal", "FramingTest",
    "FitSource", "TestDataset", "Model",
    "avg", "std-dev", "Status", "Notes"
]

writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(rows)
