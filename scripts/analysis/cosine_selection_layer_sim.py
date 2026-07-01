#!/usr/bin/env python3
"""Compare experiment-selected layers vs cosine-similarity-selected layers.

For each fold, this script selects a layer by simulation:
the layer with the highest mean pair-diff cosine similarity.

It then compares that cosine-selected layer against the original selected_layer
stored in results.json, including test one_of_4 accuracy and val_per_layer metrics.
"""

from __future__ import annotations

print("[cosine_selection_layer_sim] Starting imports...", flush=True)
import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional
import time

print("[cosine_selection_layer_sim] Loading numpy/torch...", flush=True)
import numpy as np
import torch
import torch.nn.functional as F

print("[cosine_selection_layer_sim] Loading project modules...", flush=True)
from src.hidden_states import HiddenStateBundle, bundle_output_dir
from src.hidden_states.cache import vector_common_path
from src.vectors.direction import Direction
print("[cosine_selection_layer_sim] Imports complete.", flush=True)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CosineArgs:
    hidden_states_root: Path
    vectors_root: Path
    results_root: Path
    model: str
    dataset_source: str
    vectors_dataset_source: str
    benchmark: str
    distribution: str
    split: str
    fold: int
    framing: str
    representation: str
    method: str
    fit_framing: str
    val_framing: str
    test_framing: str
    layers: str
    max_tasks: Optional[int]
    output_dir: Path

def log_path_status(name: str, path: Path) -> None:
    logger.info("%s: %s", name, path.resolve())
    logger.info("%s exists: %s", name, path.exists())


CSV_COLUMNS = [
    "model",
    "dataset_source",
    "benchmark",
    "distribution",
    "fold",
    "comparison_split",
    "framing",
    "representation",
    "method",
    "fit_framing",
    "val_framing",
    "test_framing",

    "results_selected_layer",
    "cosine_selected_layer",
    "same_selected_layer",

    "results_selected_mean_cosine",
    "cosine_selected_mean_cosine",

    "results_selected_one_of_4",
    "cosine_selected_one_of_4",

    "results_selected_test_one_of_4",
    "cosine_selected_test_one_of_4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build table comparing results-selected layer vs cosine-selected layer per fold."
    )

    parser.add_argument("--hidden-states-root", default="hidden_states_sohaila")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--results-root", default="results")

    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset-source",
        default="icse26",
        help="Split family for hidden states and results (e.g. icse26_val_1_4).",
    )
    parser.add_argument(
        "--vectors-dataset-source",
        default=None,
        help=(
            "Split family for direction vectors in vectors/ "
            "(default: same as --dataset-source)."
        ),
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--distribution", default="id")

    # Important: use validate by default for cosine-based selection.
    # This avoids selecting based on test data.
    parser.add_argument(
        "--cosine-selection-splits",
        nargs="+",
        default=["validate", "test"],
        choices=["validate", "test"],
        help="Splits used to select cosine-best layer; one output row is emitted per split.",
    )

    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")

    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max-tasks", type=int, default=None)

    parser.add_argument(
        "--output-dir",
        default="outputs/phase2_task2_cosine_selected_layer_table",
    )

    return parser.parse_args()


def make_cosine_args(cli: argparse.Namespace, fold: int, split: str) -> CosineArgs:
    vectors_dataset_source = cli.vectors_dataset_source or cli.dataset_source
    return CosineArgs(
        hidden_states_root=Path(cli.hidden_states_root),
        vectors_root=Path(cli.vectors_root),
        results_root=Path(cli.results_root),
        model=cli.model,
        dataset_source=cli.dataset_source,
        vectors_dataset_source=vectors_dataset_source,
        benchmark=cli.benchmark,
        distribution=cli.distribution,
        split=split,
        fold=fold,
        framing=cli.framing,
        representation=cli.representation,
        method=cli.method,
        fit_framing=cli.fit_framing,
        val_framing=cli.val_framing,
        test_framing=cli.test_framing,
        layers="all",
        max_tasks=cli.max_tasks,
        output_dir=Path(cli.output_dir),
    )


def safe_model_dir(model: str) -> str:
    return model.replace("/", "__")


def build_output_dir(args: CosineArgs) -> Path:
    out = (
        args.output_dir
        / safe_model_dir(args.model)
        / args.dataset_source
        / args.benchmark
        / args.distribution
        / args.split
        / f"fold_{args.fold}"
        / args.framing
        / args.representation
        / args.method
        / args.fit_framing
        / f"val={args.val_framing}__test={args.test_framing}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_bundle(args: CosineArgs) -> HiddenStateBundle:
    out_dir = bundle_output_dir(
        args.hidden_states_root,
        args.model,
        args.dataset_source,
        args.benchmark,
        args.distribution,
        args.split,
        args.framing,
        args.fold,
    )
    bundle_path = out_dir / "bundle.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Hidden-state bundle not found: {bundle_path}\n"
            f"Check --hidden-states-root, --model, --dataset-source, --benchmark, --split, --framing, --fold"
        )
    return HiddenStateBundle.load(bundle_path)


def load_direction(args: CosineArgs) -> Direction:
    common = vector_common_path(
        args.model,
        args.vectors_dataset_source,
        args.benchmark,
        args.distribution,
        args.method,
        args.fit_framing,
        args.representation,
    )
    direction_path = args.vectors_root / common / f"fold_{args.fold}" / "direction.pt"
    if not direction_path.exists():
        raise FileNotFoundError(
            f"Direction file not found: {direction_path}\n"
            f"Check --vectors-root, --vectors-dataset-source, --model, --benchmark, "
            f"--distribution, --method, --fit-framing, --representation, --fold"
        )
    return Direction.load(direction_path)


def load_selected_layer(args: CosineArgs) -> int:
    common = vector_common_path(
        args.model,
        args.dataset_source,
        args.benchmark,
        args.distribution,
        args.method,
        args.fit_framing,
        args.representation,
    )
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
    results_path = (
        args.results_root / common / eval_segment / f"fold_{args.fold}" / "results.json"
    )
    if not results_path.exists():
        raise FileNotFoundError(
            f"Selected-layer results not found: {results_path}\n"
            f"Check --results-root, --model, --dataset-source, --benchmark, --distribution, --method, --fit-framing, --val-framing, --test-framing, --representation, --fold"
        )
    data = json.loads(results_path.read_text())
    if "selected_layer" not in data:
        raise KeyError(f"results.json missing selected_layer field: {results_path}")
    return int(data["selected_layer"])


def load_fold_result_metrics(args: CosineArgs, fold: int) -> Dict[str, Dict[int, Dict[str, float]]]:
    common = vector_common_path(
        args.model,
        args.dataset_source,
        args.benchmark,
        args.distribution,
        args.method,
        args.fit_framing,
        args.representation,
    )
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
    results_path = (
        args.results_root / common / eval_segment / f"fold_{fold}" / "results.json"
    )
    if not results_path.exists():
        raise FileNotFoundError(
            f"Fold results not found: {results_path}\n"
            f"Check --results-root, --model, --dataset-source, --benchmark, --distribution, --method, --fit-framing, --val-framing, --test-framing, --representation, --fold"
        )
    data = json.loads(results_path.read_text())
    if "val_per_layer" not in data:
        raise KeyError(f"results.json missing per-layer accuracy fields: {results_path}")
    metrics: Dict[str, Dict[int, Dict[str, float]]] = {
        "val": {int(layer): layer_metrics for layer, layer_metrics in data["val_per_layer"].items()},
    }
    if "test_per_layer" in data:
        metrics["test"] = {
            int(layer): layer_metrics for layer, layer_metrics in data["test_per_layer"].items()
        }
    return metrics


def resolve_layers(args: CosineArgs, num_layers: int, direction: Direction) -> List[int]:
    del num_layers  # kept for API compatibility with existing callers
    if args.layers == "all":
        return sorted(direction.layers)
    if args.layers == "selected":
        selected = load_selected_layer(args)
        if selected not in direction.layers:
            raise ValueError(
                f"Selected layer {selected} not available in direction vector layers {direction.layers}"
            )
        return [selected]
    values = [item.strip() for item in args.layers.split(",") if item.strip()]
    if not values:
        raise ValueError("--layers must be 'selected', 'all', or a comma-separated list of integers")
    layers: List[int] = []
    for value in values:
        try:
            layer = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid layer index: {value}") from exc
        layers.append(layer)
    missing = [layer for layer in layers if layer not in direction.layers]
    if missing:
        raise ValueError(
            f"Requested layer(s) {missing} not available in direction layers {direction.layers}"
        )
    return layers


def compute_task_pairs(
    bundle: HiddenStateBundle, max_tasks: Optional[int] = None
) -> Dict[str, Dict[str, List[int]]]:
    tasks: Dict[str, Dict[str, List[int]]] = {}
    for idx, (tid, label) in enumerate(zip(bundle.task_ids, bundle.labels.tolist())):
        if tid not in tasks:
            tasks[tid] = {"correct": [], "incorrect": []}
        key = "correct" if int(label) == 1 else "incorrect"
        tasks[tid][key].append(idx)

    if max_tasks is None:
        return tasks

    selected: Dict[str, Dict[str, List[int]]] = {}
    task_count = 0
    for tid, groups in tasks.items():
        selected[tid] = groups
        if groups["correct"] and groups["incorrect"]:
            task_count += 1
        if task_count >= max_tasks:
            break
    return selected


def compute_cosine_entries(
    args: CosineArgs,
    bundle: HiddenStateBundle,
    direction: Direction,
    layers: List[int],
) -> List[Dict[str, object]]:
    acts = bundle[args.representation]
    if acts.dim() != 3:
        raise ValueError(f"Unexpected acts shape {tuple(acts.shape)}: expected [N, num_layers+1, hidden_dim]")
    if not (len(bundle.sample_ids) == len(bundle.task_ids) == len(bundle.labels) == acts.shape[0]):
        raise ValueError(
            "Mismatch between sample metadata and activations length: "
            f"{len(bundle.sample_ids)}, {len(bundle.task_ids)}, {len(bundle.labels)}, {acts.shape[0]}"
        )
    hidden_dim = acts.shape[-1]
    entries: List[Dict[str, object]] = []
    tasks = compute_task_pairs(bundle, args.max_tasks)
    valid_tasks = 0
    total_pairs = 0

    for task_id, groups in tasks.items():
        correct_idx = groups["correct"]
        incorrect_idx = groups["incorrect"]
        if not correct_idx or not incorrect_idx:
            continue
        valid_tasks += 1
        for layer in layers:
            vec = direction.vectors[layer].to(torch.float32)
            sign = direction.signs.get(layer, 1)
            vec = sign * vec
            if vec.numel() != hidden_dim:
                raise ValueError(
                    f"Hidden dimension mismatch for layer {layer}: direction {vec.numel()} vs activations {hidden_dim}"
                )
            h_corr = acts[correct_idx, layer, :].to(torch.float32)
            h_inc = acts[incorrect_idx, layer, :].to(torch.float32)
            diffs = h_corr[:, None, :] - h_inc[None, :, :]
            diffs = diffs.reshape(-1, hidden_dim)
            if diffs.numel() == 0:
                continue
            cosines = F.cosine_similarity(diffs, vec.unsqueeze(0), dim=1, eps=1e-8)
            total_pairs += cosines.numel()
            for pair_i, cosine in enumerate(cosines.tolist()):
                corr_idx = correct_idx[pair_i // len(incorrect_idx)]
                inc_idx = incorrect_idx[pair_i % len(incorrect_idx)]
                entries.append(
                    {
                        "model": args.model,
                        "dataset_source": args.dataset_source,
                        "benchmark": args.benchmark,
                        "distribution": args.distribution,
                        "split": args.split,
                        "fold": args.fold,
                        "framing": args.framing,
                        "representation": args.representation,
                        "method": args.method,
                        "fit_framing": args.fit_framing,
                        "val_framing": args.val_framing,
                        "test_framing": args.test_framing,
                        "layer": layer,
                        "task_id": task_id,
                        "correct_sample_id": bundle.sample_ids[corr_idx],
                        "incorrect_sample_id": bundle.sample_ids[inc_idx],
                        "cosine_similarity": float(cosine),
                    }
                )
    if valid_tasks == 0:
        raise ValueError("No tasks contained both correct and incorrect samples; cannot build pair differences.")
    logger.info("Built %s cosine entries from %s tasks", total_pairs, valid_tasks)
    return entries


def mean_cosine_by_layer(entries: List[Dict[str, object]]) -> Dict[int, float]:
    grouped: Dict[int, List[float]] = {}

    for entry in entries:
        layer = int(entry["layer"])
        grouped.setdefault(layer, []).append(float(entry["cosine_similarity"]))

    return {
        layer: float(np.mean(values))
        for layer, values in grouped.items()
        if values
    }


def select_layer_by_highest_cosine(layer_to_mean_cosine: Dict[int, float]) -> int:
    if not layer_to_mean_cosine:
        raise ValueError("No cosine values found; cannot select a layer.")

    # Tie-breaker: if two layers have the same mean cosine, choose the smaller layer.
    return max(
        layer_to_mean_cosine,
        key=lambda layer: (layer_to_mean_cosine[layer], -layer),
    )


def metric_value(
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    split: str,
    layer: int,
    key: str,
) -> Optional[float]:
    metric_split = "val" if split == "validate" else split
    value = metrics.get(metric_split, {}).get(layer, {}).get(key)
    return None if value is None else float(value)


def build_rows_for_fold(
    cli: argparse.Namespace,
    fold: int,
    results_selected_layer: int,
    metrics: Dict[str, Dict[int, Dict[str, float]]],
) -> List[Dict[str, object]]:
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("START fold %s", fold)

    rows: List[Dict[str, object]] = []
    for split in cli.cosine_selection_splits:
        cosine_args = make_cosine_args(cli, fold, split)

        logger.info("Cosine selection split: %s", cosine_args.split)
        logger.info("Model: %s", cosine_args.model)
        logger.info("Benchmark: %s", cosine_args.benchmark)
        logger.info("Dataset source: %s", cosine_args.dataset_source)
        logger.info("Distribution: %s", cosine_args.distribution)
        logger.info("Representation: %s", cosine_args.representation)

        step_time = time.time()
        logger.info("[fold %s|%s] Loading bundle...", fold, split)
        bundle = load_bundle(cosine_args)
        logger.info(
            "[fold %s|%s] Loaded bundle in %.2f seconds",
            fold,
            split,
            time.time() - step_time,
        )

        if cosine_args.representation not in bundle.acts:
            raise KeyError(
                f"Representation {cosine_args.representation!r} not found in bundle acts. "
                f"Available: {list(bundle.acts)}"
            )

        step_time = time.time()
        logger.info("[fold %s|%s] Loading direction...", fold, split)
        direction = load_direction(cosine_args)
        logger.info(
            "[fold %s|%s] Loaded direction in %.2f seconds",
            fold,
            split,
            time.time() - step_time,
        )

        layers = resolve_layers(
            cosine_args,
            bundle[cosine_args.representation].shape[1],
            direction,
        )
        logger.info("[fold %s|%s] Layers: %s", fold, split, layers)

        step_time = time.time()
        logger.info("[fold %s|%s] Computing cosine entries...", fold, split)
        entries = compute_cosine_entries(cosine_args, bundle, direction, layers)
        logger.info(
            "[fold %s|%s] Computed %s entries in %.2f seconds",
            fold,
            split,
            len(entries),
            time.time() - step_time,
        )

        layer_to_mean_cosine = mean_cosine_by_layer(entries)
        cosine_selected_layer = select_layer_by_highest_cosine(layer_to_mean_cosine)
        logger.info("[fold %s|%s] Cosine-selected layer: %s", fold, split, cosine_selected_layer)

        row = {
            "model": cli.model,
            "dataset_source": cli.dataset_source,
            "benchmark": cli.benchmark,
            "distribution": cli.distribution,
            "fold": fold,
            "comparison_split": split,
            "framing": cli.framing,
            "representation": cli.representation,
            "method": cli.method,
            "fit_framing": cli.fit_framing,
            "val_framing": cli.val_framing,
            "test_framing": cli.test_framing,
            "results_selected_layer": results_selected_layer,
            "cosine_selected_layer": cosine_selected_layer,
            "same_selected_layer": results_selected_layer == cosine_selected_layer,
            "results_selected_mean_cosine": layer_to_mean_cosine.get(results_selected_layer),
            "cosine_selected_mean_cosine": layer_to_mean_cosine.get(cosine_selected_layer),
            "results_selected_one_of_4": metric_value(
                metrics, split, results_selected_layer, "one_of_4"
            ),
            "cosine_selected_one_of_4": metric_value(
                metrics, split, cosine_selected_layer, "one_of_4"
            ),
            "results_selected_test_one_of_4": metric_value(
                metrics, "test", results_selected_layer, "one_of_4"
            ),
            "cosine_selected_test_one_of_4": metric_value(
                metrics, "test", cosine_selected_layer, "one_of_4"
            ),
        }
        rows.append(row)

    logger.info("[fold %s] Finished fold in %.2f seconds", fold, time.time() - start_time)
    logger.info("=" * 80)

    return rows
def write_table(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def selection_split_label(splits: List[str]) -> str:
    return "__".join(splits)


def output_csv_path(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.output_dir)
        / selection_split_label(cli.cosine_selection_splits)
        / cli.model.replace("/", "__")
        / cli.dataset_source
        / cli.benchmark
        / cli.distribution
        / "cosine_selected_layer_comparison.csv"
    )

def main() -> None:
    cli = parse_args()
    # Force a predictable INFO logger to stdout even if parent env preconfigures
    # handlers at stricter levels (common in interactive/HPC shells).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logger.info("Current working directory: %s", Path.cwd())
    logger.info("Dataset source (bundles/results): %s", cli.dataset_source)
    logger.info(
        "Vectors dataset source: %s",
        cli.vectors_dataset_source or cli.dataset_source,
    )
    logger.info("Hidden states root: %s", Path(cli.hidden_states_root).resolve())
    logger.info("Vectors root: %s", Path(cli.vectors_root).resolve())
    logger.info("Results root: %s", Path(cli.results_root).resolve())
    logger.info("Cosine selection splits: %s", cli.cosine_selection_splits)
    logger.info("Output directory argument: %s", Path(cli.output_dir).resolve())
    logger.info("Folds to process: %s", cli.folds)

    total_start = time.time()
    output_path = output_csv_path(cli)
    logger.info("Output table path: %s", output_path.resolve())

    rows = []
    for fold in cli.folds:
        logger.info("About to process fold %s", fold)
        ref_args = make_cosine_args(cli, fold, "validate")
        results_selected_layer = load_selected_layer(ref_args)
        metrics = load_fold_result_metrics(ref_args, fold)
        fold_rows = build_rows_for_fold(cli, fold, results_selected_layer, metrics)
        rows.extend(fold_rows)
        logger.info("Completed fold %s. Rows so far: %s", fold, len(rows))
        # Persist progress after each fold so an interrupted run still leaves a
        # complete table for all finished folds.
        write_table(rows, output_path)
        logger.info("Checkpointed table after fold %s", fold)

    logger.info("Wrote comparison table to %s", output_path)
    logger.info("Total runtime: %.2f seconds", time.time() - total_start)
    
if __name__ == "__main__":
    print("[cosine_selection_layer_sim] Entering main()...", flush=True)
    main()