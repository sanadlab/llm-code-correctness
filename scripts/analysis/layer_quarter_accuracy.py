"""Cache-only quarter-layer correctness accuracy (ICSE26, Mutation_DS, OOD).

For each fold, measures how often each quarter of layers correctly ranks the
canonical implementation among four candidates using per-fold reading vectors.

Reads only precomputed artifacts:
  - hidden states from ``hidden_states/.../bundle.pt``
  - per-layer reading vectors from ``vectors/.../direction.pt``
  - fold count from ``manifest.json`` (via ``--folds all``)

Layer 0 (embedding output) is excluded from quarter binning by default; pass
``--include-embedding`` to include it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from scripts.analysis.viz_common import (
    classify_labels_strict,
    load_direction_vectors,
    load_test_bundle,
    parse_fold_selector,
    viz_output_path_parts,
)
from src.hidden_states.cache import safe_model_dir

logger = logging.getLogger(__name__)

QUARTER_KEYS = ("Q1", "Q2", "Q3", "Q4")


def quarter_bins(layer_indices: Sequence[int]) -> List[List[int]]:
    """Split *layer_indices* into four contiguous bins via ``np.array_split``."""
    indices = np.asarray(list(layer_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("layer_indices must be non-empty")
    return [bin_.tolist() for bin_ in np.array_split(indices, 4)]


def task_layer_correct(
    acts: np.ndarray,
    labels: Sequence[int],
    task_ids: Sequence[str],
    direction_vectors: Dict[int, np.ndarray],
    layer_indices: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    """Per-task, per-layer strict 1-of-4 ranking correctness for one fold.

    Returns
    -------
    correct :
        Boolean array ``[num_valid_tasks, len(layer_indices)]``.
    kept_task_ids :
        Task ids aligned with rows of *correct* (tasks with exactly four
        implementations and one correct / three incorrect labels).
    """
    layer_indices = list(layer_indices)
    by_task: Dict[str, List[int]] = defaultdict(list)
    for i, tid in enumerate(task_ids):
        by_task[tid].append(i)

    kept_task_ids: List[str] = []
    rows: List[np.ndarray] = []

    for tid in sorted(by_task):
        indices = by_task[tid]
        if len(indices) != 4:
            warnings.warn(
                f"task {tid!r}: expected 4 implementations, got {len(indices)}; skipping",
                stacklevel=2,
            )
            continue

        task_labels = [int(labels[i]) for i in indices]
        _, _, ambiguous = classify_labels_strict(task_labels, [str(i) for i in indices])
        if ambiguous:
            warnings.warn(
                f"task {tid!r}: ambiguous labels {ambiguous}; skipping",
                stacklevel=2,
            )
            continue
        if sum(task_labels) != 1:
            warnings.warn(
                f"task {tid!r}: expected exactly 1 correct label, got "
                f"{sum(task_labels)}; skipping",
                stacklevel=2,
            )
            continue

        correct_idx = next(i for i, lab in zip(indices, task_labels) if lab == 1)
        incorrect_idx = [i for i, lab in zip(indices, task_labels) if lab == 0]
        if len(incorrect_idx) != 3:
            warnings.warn(
                f"task {tid!r}: expected 3 incorrect labels, got "
                f"{len(incorrect_idx)}; skipping",
                stacklevel=2,
            )
            continue

        row = np.empty(len(layer_indices), dtype=bool)
        for col, layer in enumerate(layer_indices):
            if layer not in direction_vectors:
                raise KeyError(f"layer {layer} missing in direction vectors")
            vec = direction_vectors[layer]
            correct_proj = float(acts[correct_idx, layer] @ vec)
            incorrect_projs = [float(acts[i, layer] @ vec) for i in incorrect_idx]
            row[col] = correct_proj > max(incorrect_projs)

        kept_task_ids.append(tid)
        rows.append(row)

    if not rows:
        return np.zeros((0, len(layer_indices)), dtype=bool), kept_task_ids
    return np.stack(rows, axis=0), kept_task_ids


def per_fold_quarter_accuracy(
    correct: np.ndarray,
    bins: Sequence[Sequence[int]],
    layer_indices: Sequence[int],
) -> Dict[str, float]:
    """Mean strict-correctness over all (task, layer) pairs in each quarter bin."""
    if len(bins) != 4:
        raise ValueError(f"expected 4 quarter bins, got {len(bins)}")
    layer_to_col = {layer: col for col, layer in enumerate(layer_indices)}
    out: Dict[str, float] = {}
    for key, bin_layers in zip(QUARTER_KEYS, bins):
        cols = [layer_to_col[layer] for layer in bin_layers]
        if not cols or correct.size == 0:
            out[key] = float("nan")
        else:
            out[key] = float(correct[:, cols].mean())
    return out


def aggregate_across_folds(
    per_fold_results: Sequence[Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Macro mean and std per quarter across folds (equal fold weight, ddof=0)."""
    if not per_fold_results:
        nan = {key: float("nan") for key in QUARTER_KEYS}
        return nan, nan
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for key in QUARTER_KEYS:
        values = np.asarray([fold[key] for fold in per_fold_results], dtype=np.float64)
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values, ddof=0))
    return means, stds


def layer_indices_for_bundle(
    num_layers_total: int,
    *,
    include_embedding: bool,
) -> List[int]:
    """Layer indices used for quarter binning (embedding excluded by default)."""
    start = 0 if include_embedding else 1
    return list(range(start, num_layers_total))


def build_output_paths(
    *,
    output_dir: Path,
    model: str,
    split_family: str,
    benchmark: str,
) -> Tuple[Path, Path]:
    origin_parts = viz_output_path_parts(split_family)
    out_dir = output_dir / safe_model_dir(model) / Path(*origin_parts) / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "quarter_accuracy.csv", out_dir / "quarter_accuracy.json"


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["fold", *QUARTER_KEYS]
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    lines = [
        "  ".join(h.ljust(col_widths[h]) for h in headers),
        "  ".join("-" * col_widths[h] for h in headers),
    ]
    for row in rows:
        lines.append(
            "  ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    headers = ["fold", *QUARTER_KEYS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row[h] for h in headers})


def run_analysis(
    *,
    hidden_states_root: Path,
    vectors_root: Path,
    benchmark: str,
    model: str,
    folds: Sequence[int],
    split_family: str,
    method: str,
    fit_framing: str,
    representation: str,
    test_framing: str,
    include_embedding: bool,
    output_dir: Path,
) -> Dict[str, Any]:
    per_fold_results: Dict[int, Dict[str, float]] = {}
    fold_meta: Dict[int, Dict[str, Any]] = {}
    bins_by_fold: Dict[int, List[List[int]]] = {}

    for fold in folds:
        bundle_blob = load_test_bundle(
            hidden_states_root=hidden_states_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            fold=fold,
            test_framing=test_framing,
        )
        acts_map = bundle_blob["acts"]
        if representation not in acts_map:
            raise KeyError(
                f"fold {fold}: representation {representation!r} missing in bundle acts"
            )
        acts = acts_map[representation].detach().cpu().numpy()
        task_ids = list(bundle_blob["task_ids"])
        labels = bundle_blob["labels"].detach().cpu().tolist()

        direction_vectors = load_direction_vectors(
            vectors_root=vectors_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            method=method,
            fit_framing=fit_framing,
            representation=representation,
            fold=fold,
        )

        layer_indices = layer_indices_for_bundle(
            acts.shape[1],
            include_embedding=include_embedding,
        )
        missing_layers = [layer for layer in layer_indices if layer not in direction_vectors]
        if missing_layers:
            raise KeyError(
                f"fold {fold}: direction vectors missing layer(s) {missing_layers}"
            )

        bins = quarter_bins(layer_indices)
        correct, kept_task_ids = task_layer_correct(
            acts,
            labels,
            task_ids,
            direction_vectors,
            layer_indices,
        )
        if kept_task_ids:
            quarter_acc = per_fold_quarter_accuracy(correct, bins, layer_indices)
        else:
            quarter_acc = {key: float("nan") for key in QUARTER_KEYS}

        per_fold_results[fold] = quarter_acc
        bins_by_fold[fold] = bins
        fold_meta[fold] = {
            "num_tasks": len(kept_task_ids),
            "num_tasks_skipped": len(set(task_ids)) - len(kept_task_ids),
            "layer_indices": layer_indices,
            "quarter_layer_bins": {k: b for k, b in zip(QUARTER_KEYS, bins)},
        }

    mean_by_quarter, std_by_quarter = aggregate_across_folds(
        [per_fold_results[f] for f in folds]
    )

    table_rows: List[Dict[str, Any]] = []
    for fold in folds:
        row: Dict[str, Any] = {"fold": f"fold_{fold}"}
        row.update(per_fold_results[fold])
        table_rows.append(row)

    mean_row: Dict[str, Any] = {"fold": "mean-across-folds"}
    mean_row.update(mean_by_quarter)
    table_rows.append(mean_row)

    std_row: Dict[str, Any] = {"fold": "std-across-folds"}
    std_row.update(std_by_quarter)
    table_rows.append(std_row)

    csv_path, json_path = build_output_paths(
        output_dir=output_dir,
        model=model,
        split_family=split_family,
        benchmark=benchmark,
    )
    write_csv(csv_path, table_rows)

    print(format_table(table_rows))

    summary: Dict[str, Any] = {
        "benchmark": benchmark,
        "model": model,
        "split_family": split_family,
        "method": method,
        "fit_framing": fit_framing,
        "representation": representation,
        "test_framing": test_framing,
        "include_embedding": include_embedding,
        "folds": list(folds),
        "per_fold": {
            str(fold): {
                "quarter_accuracy": per_fold_results[fold],
                **fold_meta[fold],
            }
            for fold in folds
        },
        "mean_across_folds": mean_by_quarter,
        "std_across_folds": std_by_quarter,
        "table_rows": table_rows,
        "output_paths": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarter-layer 1-of-4 strict ranking accuracy from cached bundles.",
    )
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument(
        "--data-root",
        default="dataset/icse26",
        help="ICSE26 root or Mutation_DS split/phase2 dir (for --folds all manifests).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Either explicit fold indices (e.g. 0 3) or 'all'.",
    )
    parser.add_argument("--split-family", default="icse26")
    parser.add_argument("--hidden-states-root", default="hidden_states")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Include layer 0 (embedding output) in quarter binning (excluded by default).",
    )
    parser.add_argument("--output-dir", default="analysis/layer_quarter_accuracy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = Path(args.data_root)
    folds = parse_fold_selector(
        args.folds,
        split_family=args.split_family,
        data_root=data_root,
        benchmark=args.benchmark,
    )

    summary = run_analysis(
        hidden_states_root=Path(args.hidden_states_root),
        vectors_root=Path(args.vectors_root),
        benchmark=args.benchmark,
        model=args.model,
        folds=folds,
        split_family=args.split_family,
        method=args.method,
        fit_framing=args.fit_framing,
        representation=args.representation,
        test_framing=args.test_framing,
        include_embedding=args.include_embedding,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
