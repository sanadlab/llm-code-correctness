"""Generate ablation experiment matrix TSV with aggregated fold metrics.

Walks a fit+score results tree (e.g. fixed_ablation_results/) and emits one row
per ablation cell with mean/std at the validation-selected layer and at the
oracle test-best layer, averaged across folds.

The expensive part is reading ~34k tiny results.json files over NFS, where each
open() is a network round-trip. Two optimisations make this fast:

  1. Files are read in parallel (NFS reads are latency-bound, so threads overlap
     the per-file wait). Tune with --workers.
  2. The per-config aggregates are written to a small JSON cache. Regenerating
     the sheet from the cache is instant; pass --rebuild to force a re-scan.

Usage:
    # First run scans results and writes the cache alongside the sheet:
    python scripts/analysis/generate_ablation_matrix.py --output matrix.tsv

    # Subsequent runs reuse the cache (fast), e.g. to switch metrics/columns:
    python scripts/analysis/generate_ablation_matrix.py --metrics one_of_4 pairwise \\
        | python csv_to_sheets.py /dev/stdin
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

METHODS = ("mean_diff", "pca")
REPRESENTATIONS = ("sequence_last", "response_avg")
REP_DISPLAY = {"sequence_last": "seq_last", "response_avg": "response_avg"}
REP_ON_DISK = {v: k for k, v in REP_DISPLAY.items()}
FIT_FRAMINGS = ("function", "neutral", "concept", "raw")
VAL_FRAMINGS = ("neutral", "concept")
TEST_FRAMINGS = ("neutral", "concept")
FIT_SOURCES = (
    "id", "mbpp", "syn",
    "paired", "weighted",
    "paired_confounds", "weighted_confounds",
    "paired_he", "weighted_he",
    "paired_confounds_he", "weighted_confounds_he",
)
FIT_SOURCE_TO_SPLIT = {
    "id": "icse26",
    "mbpp": "ood_mbpp",
    "syn": "ood_synthetic",
    "paired": "mutants_only_bcb_based/paired",
    "weighted": "mutants_only_bcb_based/weighted",
    "paired_confounds": "mutants_confounds_bcb_based/paired",
    "weighted_confounds": "mutants_confounds_bcb_based/weighted",
    "paired_he": "mutants_only_he_based/paired",
    "weighted_he": "mutants_only_he_based/weighted",
    "paired_confounds_he": "mutants_confounds_he_based/paired",
    "weighted_confounds_he": "mutants_confounds_he_based/weighted",
}
SPLIT_TO_FIT_SOURCE = {v: k for k, v in FIT_SOURCE_TO_SPLIT.items()}
TEST_DATASETS = ("bigcodebench", "humaneval", "mutants_only", "mutants_confounds")
# icse26 benchmarks where mutants/confounds fit sources may use --val-on-test-dataset.
VAL_DATASET_BENCHMARKS = frozenset({"bigcodebench", "humaneval"})
MUTANTS_FIT_SOURCES = frozenset({"paired", "weighted", "paired_he", "weighted_he"})
MUTANTS_CONFOUNDS_FIT_SOURCES = frozenset(
    {"paired_confounds", "weighted_confounds", "paired_confounds_he", "weighted_confounds_he"}
)
VAL_DS_FIT_SOURCES = MUTANTS_FIT_SOURCES | MUTANTS_CONFOUNDS_FIT_SOURCES
MODELS = (
    "mistralai__Mistral-7B-Instruct-v0.3",
    "Qwen__Qwen2.5-Coder-7B-Instruct",
    "codellama__CodeLlama-7b-Instruct-hf",
    "infly__OpenCoder-8B-Instruct",
)
FRAMING_PREFIX = {"function": "F", "neutral": "N", "concept": "C", "raw": "R"}
# All metrics are aggregated and cached regardless of what is rendered, so the
# cache can be reused across different --metrics selections.
ALL_METRICS = ("one_of_4", "pairwise")
METRIC_KEYS = {"one_of_4": "one_of_4", "pairwise": "pairwise_acc"}
METRIC_LABELS = {"one_of_4": "1of4", "pairwise": "pairwise"}
EXPECTED_FOLDS = {
    "icse26": 10,
    "ood_mbpp": 40,
    "ood_synthetic": 40,
    "mutants_only_bcb_based/paired": 10,
    "mutants_only_bcb_based/weighted": 10,
    "mutants_confounds_bcb_based/paired": 10,
    "mutants_confounds_bcb_based/weighted": 10,
    "mutants_only_he_based/paired": 10,
    "mutants_only_he_based/weighted": 10,
    "mutants_confounds_he_based/paired": 10,
    "mutants_confounds_he_based/weighted": 10,
}
DEFAULT_WORKERS = 32


def framing_key(fit: str, val: str, test: str) -> str:
    return f"{FRAMING_PREFIX[fit]}->{FRAMING_PREFIX[val]}->{FRAMING_PREFIX[test]}"


def is_irrelevant_combo(
    representation: str,
    fit_framing: str,
) -> bool:
    """Combos the ablation deliberately skips — omit from the matrix (not 'missing')."""
    # raw fit has no response slice; response_avg is degenerate (see ablation_grid.sh).
    return fit_framing == "raw" and representation == "response_avg"


def eval_segment(
    val_framing: str,
    test_framing: str,
    *,
    val_dataset: Optional[str] = None,
) -> str:
    """Mirror fit_and_score_from_cache._eval_segment for result path lookup."""
    base = f"val={val_framing}__test={test_framing}"
    if val_dataset is not None:
        return f"val_ds={val_dataset}__{base}"
    return base


def config_path(
    model: str,
    split_family: str,
    test_dataset: str,
    method: str,
    fit_framing: str,
    representation: str,
    val_framing: str,
    test_framing: str,
    *,
    val_dataset: Optional[str] = None,
) -> str:
    return (
        f"{model}/{split_family}/{test_dataset}/id/{method}/{fit_framing}/"
        f"{representation}/{eval_segment(val_framing, test_framing, val_dataset=val_dataset)}"
    )


def build_config_registry() -> List[dict]:
    """Return ablation config groups with stable ConfigIDs (irrelevant combos omitted)."""
    configs: List[dict] = []
    for method, representation, fit_framing, val_framing, test_framing in product(
        METHODS, REPRESENTATIONS, FIT_FRAMINGS, VAL_FRAMINGS, TEST_FRAMINGS,
    ):
        if is_irrelevant_combo(representation, fit_framing):
            continue
        configs.append({
            "method": method,
            "representation": representation,
            "fit_framing": fit_framing,
            "val_framing": val_framing,
            "test_framing": test_framing,
            "framing_key": framing_key(fit_framing, val_framing, test_framing),
            "representation_display": REP_DISPLAY[representation],
        })
    for idx, cfg in enumerate(configs, start=1):
        cfg["config_id"] = f"C{idx}"
    return configs


def _best_block(results: dict) -> dict:
    if "test_theoretical_best_layer" in results:
        block = dict(results["test_theoretical_best_layer"])
        block.pop("layer", None)
        return block
    if "test_at_best_layer" in results:
        return dict(results["test_at_best_layer"])
    raise KeyError("results.json missing best-layer test block")


def _read_fold(path: Path) -> Dict[str, Tuple[float, float]]:
    """Return {metric: (selected_value, best_value)} for one fold's results.json."""
    with open(path) as f:
        results = json.load(f)
    selected = results["test_at_selected_layer"]
    best = _best_block(results)
    out: Dict[str, Tuple[float, float]] = {}
    for metric in ALL_METRICS:
        key = METRIC_KEYS[metric]
        out[metric] = (float(selected[key]), float(best[key]))
    return out


def iter_results_paths(results_dir: Path) -> List[Path]:
    """List all results.json files (find is faster than rglob on NFS)."""
    proc = subprocess.run(
        ["find", str(results_dir), "-type", "f", "-name", "results.json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in proc.stdout.splitlines() if line]


def _agg(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def scan_results(results_dir: Path, workers: int) -> Dict[str, dict]:
    """Read every fold in parallel and aggregate per config path.

    Returns {config_path: {"n_folds": int,
                           "selected": {metric: {"mean","std"}},
                           "best":     {metric: {"mean","std"}}}}.
    """
    print("[matrix] listing results.json files...", file=sys.stderr)
    paths = iter_results_paths(results_dir)
    print(f"[matrix] reading {len(paths)} folds with {workers} workers...", file=sys.stderr)

    selected_vals: Dict[str, Dict[str, List[float]]] = {}
    best_vals: Dict[str, Dict[str, List[float]]] = {}

    def configs_of(path: Path) -> str:
        return str(path.parent.parent.relative_to(results_dir))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, fold in zip(paths, pool.map(_read_fold, paths)):
            config = configs_of(path)
            if config not in selected_vals:
                selected_vals[config] = {m: [] for m in ALL_METRICS}
                best_vals[config] = {m: [] for m in ALL_METRICS}
            for metric, (sel, best) in fold.items():
                selected_vals[config][metric].append(sel)
                best_vals[config][metric].append(best)

    aggregates: Dict[str, dict] = {}
    for config in selected_vals:
        n_folds = len(selected_vals[config][ALL_METRICS[0]])
        aggregates[config] = {
            "n_folds": n_folds,
            "selected": {m: _agg(selected_vals[config][m]) for m in ALL_METRICS},
            "best": {m: _agg(best_vals[config][m]) for m in ALL_METRICS},
        }
    print(f"[matrix] aggregated {len(aggregates)} config paths", file=sys.stderr)
    return aggregates


def load_or_build_aggregates(
    results_dir: Path,
    cache_path: Optional[Path],
    rebuild: bool,
    workers: int,
) -> Dict[str, dict]:
    if cache_path is not None and cache_path.is_file() and not rebuild:
        print(f"[matrix] loading cache from {cache_path}", file=sys.stderr)
        with open(cache_path) as f:
            return json.load(f)

    aggregates = scan_results(results_dir, workers)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(aggregates, f)
        print(f"[matrix] wrote cache to {cache_path}", file=sys.stderr)
    return aggregates


def fmt(stat: Optional[Dict[str, float]], field: str) -> str:
    if not stat:
        return ""
    return f"{stat[field]:.4f}"


def status_for(n_folds: int, split_family: str) -> str:
    expected = EXPECTED_FOLDS[split_family]
    if n_folds == 0:
        return "missing"
    if n_folds == expected:
        return "complete"
    return "partial"


def metric_column_names(metrics: Sequence[str]) -> List[str]:
    if len(metrics) == 1:
        return ["avg", "std-dev", "avg (best)", "std-dev (best)"]
    cols: List[str] = []
    for metric in metrics:
        label = METRIC_LABELS[metric]
        cols.extend([
            f"avg ({label})",
            f"std-dev ({label})",
            f"avg (best, {label})",
            f"std-dev (best, {label})",
        ])
    return cols


def base_fieldnames(metrics: Sequence[str], *, include_val_dataset: bool) -> List[str]:
    fields = [
        "ConfigID", "Method", "Representation", "FramingKey",
        "FramingFit", "FramingVal", "FramingTest",
        "FitSource", "TestDataset",
    ]
    if include_val_dataset:
        fields.append("ValDataset")
    fields.extend([
        "Model",
        "n_folds",
        *metric_column_names(metrics),
        "Status", "Notes",
    ])
    return fields


def _val_dataset_variants(
    fit_source: str,
    test_dataset: str,
    *,
    include_val_dataset: bool,
) -> List[Tuple[Optional[str], str]]:
    """Return (val_ds path token, ValDataset column label) pairs for one cell."""
    if not include_val_dataset:
        return [(None, fit_source)]
    variants: List[Tuple[Optional[str], str]] = [(None, fit_source)]
    if (
        fit_source in VAL_DS_FIT_SOURCES
        and test_dataset in VAL_DATASET_BENCHMARKS
    ):
        variants.append((test_dataset, test_dataset))
    return variants


def enumerate_rows(*, include_val_dataset: bool = False) -> Iterable[dict]:
    for cfg in build_config_registry():
        for fit_source, test_dataset, model in product(FIT_SOURCES, TEST_DATASETS, MODELS):
            if test_dataset == "mutants_only" and fit_source not in MUTANTS_FIT_SOURCES:
                continue
            if test_dataset == "mutants_confounds" and fit_source not in MUTANTS_CONFOUNDS_FIT_SOURCES:
                continue
            if fit_source in MUTANTS_CONFOUNDS_FIT_SOURCES and test_dataset == "mutants_only":
                continue
            if fit_source in MUTANTS_FIT_SOURCES and test_dataset == "mutants_confounds":
                continue
            split_family = FIT_SOURCE_TO_SPLIT[fit_source]
            for val_ds, val_dataset_label in _val_dataset_variants(
                fit_source, test_dataset, include_val_dataset=include_val_dataset,
            ):
                spec = {
                    "config_id": cfg["config_id"],
                    "method": cfg["method"],
                    "representation_display": cfg["representation_display"],
                    "framing_key": cfg["framing_key"],
                    "fit_framing": cfg["fit_framing"],
                    "val_framing": cfg["val_framing"],
                    "test_framing": cfg["test_framing"],
                    "fit_source": fit_source,
                    "split_family": split_family,
                    "test_dataset": test_dataset,
                    "model": model,
                    "config_path": config_path(
                        model, split_family, test_dataset,
                        cfg["method"], cfg["fit_framing"], cfg["representation"],
                        cfg["val_framing"], cfg["test_framing"],
                        val_dataset=val_ds,
                    ),
                }
                if include_val_dataset:
                    spec["val_dataset"] = val_dataset_label
                yield spec


def row_to_dict(spec: dict, aggregates: Dict[str, dict], metrics: Sequence[str]) -> dict:
    agg = aggregates.get(spec["config_path"])
    n_folds = agg["n_folds"] if agg else 0
    sel = agg["selected"] if agg else {}
    best = agg["best"] if agg else {}

    row = {
        "ConfigID": spec["config_id"],
        "Method": spec["method"],
        "Representation": spec["representation_display"],
        "FramingKey": spec["framing_key"],
        "FramingFit": spec["fit_framing"],
        "FramingVal": spec["val_framing"],
        "FramingTest": spec["test_framing"],
        "FitSource": spec["fit_source"],
        "TestDataset": spec["test_dataset"],
        "Model": spec["model"],
        "n_folds": n_folds,
        "Status": status_for(n_folds, spec["split_family"]),
        "Notes": "",
    }
    if "val_dataset" in spec:
        row["ValDataset"] = spec["val_dataset"]
    if len(metrics) == 1:
        metric = metrics[0]
        row["avg"] = fmt(sel.get(metric), "mean")
        row["std-dev"] = fmt(sel.get(metric), "std")
        row["avg (best)"] = fmt(best.get(metric), "mean")
        row["std-dev (best)"] = fmt(best.get(metric), "std")
    else:
        for metric in metrics:
            label = METRIC_LABELS[metric]
            row[f"avg ({label})"] = fmt(sel.get(metric), "mean")
            row[f"std-dev ({label})"] = fmt(sel.get(metric), "std")
            row[f"avg (best, {label})"] = fmt(best.get(metric), "mean")
            row[f"std-dev (best, {label})"] = fmt(best.get(metric), "std")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation matrix TSV from fit+score results.",
    )
    parser.add_argument(
        "--results-dir",
        default="fixed_ablation_results",
        help="Root directory containing per-fold results.json trees",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=["one_of_4"],
        help="Metrics to render as columns (default: one_of_4)",
    )
    parser.add_argument(
        "--cache",
        default="analysis/ablation/matrix_cache.json",
        help="Path to the aggregated cache JSON (set to '' to disable caching)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a re-scan of results even if the cache exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel file readers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Field delimiter (default: tab)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path, or '-' for stdout (default)",
    )
    parser.add_argument(
        "--include-val-dataset",
        action="store_true",
        help=(
            "Add ValDataset column and rows for val_ds=<benchmark> results "
            "(layer selection on the OOD validate split; mutants/confounds fit sources)"
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is None and (not results_dir.is_dir()):
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    if (cache_path is None or not cache_path.is_file() or args.rebuild) and not results_dir.is_dir():
        print(f"error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    metrics = list(dict.fromkeys(args.metrics))  # preserve order, dedupe
    include_val_dataset = args.include_val_dataset
    fieldnames = base_fieldnames(metrics, include_val_dataset=include_val_dataset)
    aggregates = load_or_build_aggregates(
        results_dir, cache_path, args.rebuild, args.workers,
    )

    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    try:
        writer = csv.DictWriter(
            out, fieldnames=fieldnames, delimiter=args.delimiter, lineterminator="\n",
        )
        writer.writeheader()
        for spec in enumerate_rows(include_val_dataset=include_val_dataset):
            writer.writerow(row_to_dict(spec, aggregates, metrics))
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Downstream consumer (e.g. a pipe) closed early. Redirect stdout to
        # devnull so the interpreter shutdown flush doesn't raise again, then
        # exit quietly.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)
