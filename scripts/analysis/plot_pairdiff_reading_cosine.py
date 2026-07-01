#!/usr/bin/env python3
"""Core pair-diff cosine analysis: hidden states vs. learned reading direction.

Purpose
-------
For each task, forms all correct×incorrect sample pairs and measures how well the
hidden-state difference (h_correct - h_incorrect) aligns with the per-layer reading
direction learned from validation data.

Inputs
------
- Hidden-state bundles (``bundle.pt``) under ``--hidden-states-root``
- Reading directions (``direction.pt``) under ``--vectors-root``
- Layer-selection metadata (``results.json``) under ``--results-root``

Outputs
-------
- ``cosine_similarities.csv``: one row per pair/layer with cosine similarity
- ``summary.json``: per-layer mean/median/std of cosine similarities
- Optional plots:
  - per-layer histograms (``layer_*_cosine_hist.png``)
  - multi-layer overlay (``overlay_layers_cosine.png``)
  - top-percentile layer density overlay (``top_*_percentile_layers_cosine_overlay_fold_*.png``)
  - validation/test overlay with dashed val + solid test lines
    (``top_*_percentile_layers_val_test_cosine_overlay_fold_*.png``)
  - cross-fold overlay of the selected layer (``overlay_folds/``)

Key CLI flags
-------------
- ``--layers``: ``selected``, ``all``, or comma-separated indices
- ``--top-percentile``: plot layers at/above a val pairwise-acc percentile
- ``--top-percentile-val-test-overlay``: overlay validate (dashed) and test (solid)
- ``--overlay`` / ``--overlay-folds``: combine layers or folds on one figure

Related scripts
---------------
- ``run_pairdiff_all_models.sh``: batch run for multiple models (selected layer)
- ``run_pairdiff_top_percentile_graphs.sh``: top-percentile test overlays per fold
- ``run_pairdiff_top_percentile_val_test_graphs.sh``: val/test top-percentile overlays
- ``pairdiff_cosine_outperformance_stats.py``: stats on layers beating selected layer

Usage
-----
::

    python scripts/analysis/plot_pairdiff_reading_cosine.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --benchmark bigcodebench --split test --fold 0 \\
        --layers all --top-percentile --percentile 90
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.hidden_states import HiddenStateBundle, bundle_output_dir, safe_model_dir
from src.hidden_states.cache import vector_common_path
from src.vectors.direction import Direction

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "model",
    "dataset_source",
    "benchmark",
    "distribution",
    "split",
    "fold",
    "framing",
    "representation",
    "method",
    "fit_framing",
    "val_framing",
    "test_framing",
    "layer",
    "task_id",
    "correct_sample_id",
    "incorrect_sample_id",
    "cosine_similarity",
]


@dataclass(frozen=True)
class Args:
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
    overlay: bool = False
    folds: Optional[List[int]] = None
    overlay_folds: bool = False
    top_percentile: bool = False
    top_percentile_val_test_overlay: bool = False
    percentile: float = 90.0



def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity between hidden-state pair differences and learned reading directions."
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
    parser.add_argument("--split", default="test")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--framing", default="neutral")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--layers",
        default="selected",
        help="selected, all, or comma-separated layer indices",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit analysis to the first N tasks with both correct and incorrect samples.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Plot multiple requested layers on a single overlay figure",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional folds for overlay-folds mode, comparing the same selected layer across folds.",
    )
    parser.add_argument(
        "--overlay-folds",
        action="store_true",
        help="Overlay the selected/specified layer across multiple folds.",
    )
    parser.add_argument(
        "--top-percentile",
        action="store_true",
        help="Plot the top percentile layers by test pairwise accuracy for the fold.",
    )
    parser.add_argument(
        "--top-percentile-val-test-overlay",
        action="store_true",
        help="Overlay top-percentile layer cosine densities for validation (dashed) and test (solid).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for selecting top layers by test pairwise accuracy.",
    )
    parser.add_argument("--output-dir", default="outputs/phase2_task2_cosine")
    args = parser.parse_args()
    vectors_dataset_source = args.vectors_dataset_source or args.dataset_source
    return Args(
        hidden_states_root=Path(args.hidden_states_root),
        vectors_root=Path(args.vectors_root),
        results_root=Path(args.results_root),
        model=args.model,
        dataset_source=args.dataset_source,
        vectors_dataset_source=vectors_dataset_source,
        benchmark=args.benchmark,
        distribution=args.distribution,
        split=args.split,
        fold=args.fold,
        framing=args.framing,
        representation=args.representation,
        method=args.method,
        fit_framing=args.fit_framing,
        val_framing=args.val_framing,
        test_framing=args.test_framing,
        layers=args.layers,
        max_tasks=args.max_tasks,
        output_dir=Path(args.output_dir),
        overlay=args.overlay,
        folds=args.folds,
        overlay_folds=args.overlay_folds,
        top_percentile=args.top_percentile,
        top_percentile_val_test_overlay=args.top_percentile_val_test_overlay,
        percentile=args.percentile,
    )


def load_bundle(args: Args) -> HiddenStateBundle:
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


def load_direction(args: Args) -> Direction:
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


def load_selected_layer(args: Args) -> int:
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
        raise KeyError(
            f"results.json missing selected_layer field: {results_path}"
        )
    return int(data["selected_layer"])


def load_fold_result_metrics(args: Args, fold: int) -> Dict[str, Dict[int, Dict[str, float]]]:
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
        raise KeyError(
            f"results.json missing val_per_layer field: {results_path}"
        )
    metrics: Dict[str, Dict[int, Dict[str, float]]] = {
        "val": {int(layer): layer_metrics for layer, layer_metrics in data["val_per_layer"].items()},
    }
    if "test_per_layer" in data:
        metrics["test"] = {
            int(layer): layer_metrics for layer, layer_metrics in data["test_per_layer"].items()
        }
    return metrics


def _smooth_density(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hist, bin_edges = np.histogram(values, bins=120, range=(-1.0, 1.0), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    try:
        from scipy.ndimage import gaussian_filter1d

        smooth = gaussian_filter1d(hist, sigma=1.5)
    except Exception:
        smooth = hist
    return bin_centers, smooth


def select_top_percentile_layers(
    val_layer_metrics: Dict[int, Dict[str, float]],
    percentile: float,
    selected_layer: Optional[int] = None,
) -> List[int]:
    values = [metrics.get("pairwise_acc") for metrics in val_layer_metrics.values()]
    values = [value for value in values if value is not None]
    if not values:
        raise ValueError("No pairwise_acc values available for percentile selection.")
    threshold = float(np.percentile(np.array(values, dtype=np.float64), percentile))
    selected = [
        layer
        for layer, metrics in val_layer_metrics.items()
        if metrics.get("pairwise_acc") is not None and metrics["pairwise_acc"] >= threshold
    ]
    if selected_layer is not None and selected_layer not in selected:
        selected.append(selected_layer)
    return sorted(selected, key=lambda layer: (-val_layer_metrics[layer]["pairwise_acc"], layer))


def plot_top_percentile_layers(
    args: Args,
    entries: List[Dict[str, object]],
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    out_dir: Path,
    experiment_selected_layer: int,
) -> None:
    test_metrics = metrics["test"]
    val_metrics = metrics["val"]
    selected_layers = sorted({int(entry["layer"]) for entry in entries})
    if not selected_layers:
        logger.warning("No layers selected for top percentile plot for fold %s", args.fold)
        return

    grouped: Dict[int, List[float]] = defaultdict(list)
    for entry in entries:
        layer = int(entry["layer"])
        grouped[layer].append(float(entry["cosine_similarity"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, layer in enumerate(selected_layers):
        values = np.array(grouped[layer], dtype=np.float32)
        if values.size == 0:
            continue
        bin_centers, smooth = _smooth_density(values)

        color = colors[i % len(colors)] if colors else None
        line_width = 3.0 if layer == experiment_selected_layer else 1.5
        cosine_sim_mean = float(np.mean(values))
        val_acc = val_metrics.get(layer, {}).get("pairwise_acc", float("nan"))
        test_acc = test_metrics.get(layer, {}).get("one_of_4", float("nan"))
        label_text = (
            f"layer {layer} (Cosine_Sim: {cosine_sim_mean:.3f} | Val_Acc: {val_acc:.4f} | Test_Acc: {test_acc:.4f})"
        )
        if layer == experiment_selected_layer:
            label_text += " <SELECTED>"
        ax.plot(bin_centers, smooth, label=label_text, color=color, linewidth=line_width)

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(
        f"Top {args.percentile:.0f}th percentile layer cosine similarity overlay for fold {args.fold}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / f"top_{int(args.percentile)}th_percentile_layers_cosine_overlay_fold_{args.fold}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote top-percentile cosine overlay plot to %s", fig_path)


def plot_top_percentile_val_test_overlay(
    args: Args,
    val_entries: List[Dict[str, object]],
    test_entries: List[Dict[str, object]],
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    out_dir: Path,
    experiment_selected_layer: int,
    top_layers: List[int],
) -> None:
    test_metrics = metrics["test"]
    val_metrics = metrics["val"]
    if not top_layers:
        logger.warning("No layers selected for val/test top percentile plot for fold %s", args.fold)
        return

    grouped_val: Dict[int, List[float]] = defaultdict(list)
    for entry in val_entries:
        grouped_val[int(entry["layer"])].append(float(entry["cosine_similarity"]))
    grouped_test: Dict[int, List[float]] = defaultdict(list)
    for entry in test_entries:
        grouped_test[int(entry["layer"])].append(float(entry["cosine_similarity"]))

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, layer in enumerate(top_layers):
        color = colors[i % len(colors)] if colors else None
        line_width = 3.0 if layer == experiment_selected_layer else 1.5
        selected_suffix = " <SELECTED>" if layer == experiment_selected_layer else ""
        val_acc = val_metrics.get(layer, {}).get("pairwise_acc", float("nan"))
        test_acc = test_metrics.get(layer, {}).get("one_of_4", float("nan"))

        val_values = np.array(grouped_val.get(layer, []), dtype=np.float32)
        if val_values.size > 0:
            bin_centers, smooth = _smooth_density(val_values)
            val_mean = float(np.mean(val_values))
            val_label = (
                f"layer {layer} val (Cosine_Sim: {val_mean:.3f} | Val_Acc: {val_acc:.4f} | Test_Acc: {test_acc:.4f})"
                f"{selected_suffix}"
            )
            ax.plot(
                bin_centers,
                smooth,
                label=val_label,
                color=color,
                linewidth=line_width,
                linestyle="--",
            )

        test_values = np.array(grouped_test.get(layer, []), dtype=np.float32)
        if test_values.size > 0:
            bin_centers, smooth = _smooth_density(test_values)
            test_mean = float(np.mean(test_values))
            test_label = (
                f"layer {layer} test (Cosine_Sim: {test_mean:.3f} | Val_Acc: {val_acc:.4f} | Test_Acc: {test_acc:.4f})"
                f"{selected_suffix}"
            )
            ax.plot(
                bin_centers,
                smooth,
                label=test_label,
                color=color,
                linewidth=line_width,
                linestyle="-",
            )

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(
        f"Top {args.percentile:.0f}th percentile layer cosine similarity "
        f"(validation dashed, test solid) for fold {args.fold}"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig_path = (
        out_dir
        / f"top_{int(args.percentile)}th_percentile_layers_val_test_cosine_overlay_fold_{args.fold}.png"
    )
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote top-percentile val/test cosine overlay plot to %s", fig_path)


def resolve_layers(args: Args, num_layers: int, direction: Direction) -> List[int]:
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
    layers = []
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


def build_output_dir(args: Args) -> Path:
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


def build_overlay_folds_output_dir(args: Args) -> Path:
    out = (
        args.output_dir
        / safe_model_dir(args.model)
        / args.dataset_source
        / args.benchmark
        / args.distribution
        / args.split
        / "overlay_folds"
        / args.framing
        / args.representation
        / args.method
        / args.fit_framing
        / f"val={args.val_framing}__test={args.test_framing}"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def replace_fold(args: Args, fold: int) -> Args:
    return replace(args, fold=fold)


def compute_cosine_entries_for_fold(
    args: Args,
    fold: int,
    layers: List[int],
) -> List[Dict[str, object]]:
    local_args = replace_fold(args, fold)
    bundle = load_bundle(local_args)
    if local_args.representation not in bundle.acts:
        raise KeyError(
            f"Representation {local_args.representation!r} not found in bundle acts. Available: {list(bundle.acts)}"
        )
    direction = load_direction(local_args)
    return compute_cosine_entries(local_args, bundle, direction, layers)


def compute_cosine_entries_for_split(
    args: Args,
    split: str,
    layers: List[int],
) -> List[Dict[str, object]]:
    local_args = replace(args, split=split)
    bundle = load_bundle(local_args)
    if local_args.representation not in bundle.acts:
        raise KeyError(
            f"Representation {local_args.representation!r} not found in bundle acts. Available: {list(bundle.acts)}"
        )
    direction = load_direction(local_args)
    return compute_cosine_entries(local_args, bundle, direction, layers)


def plot_folds_overlay(
    args: Args,
    entries_by_fold: Dict[int, List[Dict[str, object]]],
    out_dir: Path,
) -> None:
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, (fold, entries) in enumerate(sorted(entries_by_fold.items())):
        values = np.array([float(entry["cosine_similarity"]) for entry in entries], dtype=np.float32)
        if values.size == 0:
            continue
        hist, bin_edges = np.histogram(values, bins=120, range=(-1.0, 1.0), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        try:
            from scipy.ndimage import gaussian_filter1d

            smooth = gaussian_filter1d(hist, sigma=1.5)
        except Exception:
            smooth = hist
        color = colors[i % len(colors)] if colors else None
        selected_layers = sorted({int(entry["layer"]) for entry in entries})
        layer_label = selected_layers[0] if len(selected_layers) == 1 else selected_layers
        ax.plot(
            bin_centers,
            smooth,
            label=f"fold {fold} (layer {layer_label})",
            color=color,
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="best")
    fold_labels = ",".join(str(fold) for fold in sorted(entries_by_fold))
    ax.set_title(
        f"{args.model} {args.benchmark} {args.split} folds {fold_labels} overlay"
    )
    fig.tight_layout()
    fig_path = out_dir / f"overlay_folds_{fold_labels}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def compute_fold_overlay_entries(args: Args) -> Dict[int, List[Dict[str, object]]]:
    if args.folds is None:
        raise ValueError("--overlay-folds requires --folds to be set")
    entries_by_fold: Dict[int, List[Dict[str, object]]] = {}
    for fold in args.folds:
        local_args = replace_fold(args, fold)
        bundle = load_bundle(local_args)
        if local_args.representation not in bundle.acts:
            raise KeyError(
                f"Representation {local_args.representation!r} not found in bundle acts. Available: {list(bundle.acts)}"
            )
        direction = load_direction(local_args)
        layers = resolve_layers(local_args, bundle[local_args.representation].shape[1], direction)
        if len(layers) != 1:
            raise ValueError(
                "Overlay-folds mode requires exactly one layer, either --layers selected or a single explicit layer index."
            )
        entries_by_fold[fold] = compute_cosine_entries(local_args, bundle, direction, layers)
    return entries_by_fold


def compute_task_pairs(bundle: HiddenStateBundle, max_tasks: Optional[int] = None) -> Dict[str, Dict[str, List[int]]]:
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
    args: Args,
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


def summarize(entries: List[Dict[str, object]], layers: List[int]) -> Dict[str, object]:
    by_layer: Dict[int, List[float]] = {layer: [] for layer in layers}
    for entry in entries:
        by_layer[int(entry["layer"])].append(float(entry["cosine_similarity"]))
    summary: Dict[str, object] = {
        "num_samples": len(entries),
        "num_tasks": len({entry["task_id"] for entry in entries}),
        "num_pairs": len(entries),
        "layers": layers,
        "per_layer": {},
    }
    for layer, cosines in by_layer.items():
        if not cosines:
            summary["per_layer"][str(layer)] = None
            continue
        values = torch.tensor(cosines, dtype=torch.float32)
        summary["per_layer"][str(layer)] = {
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "count": values.numel(),
        }
    return summary


def write_csv(entries: List[Dict[str, object]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)


def plot_layer_histograms(args: Args, entries: List[Dict[str, object]], out_dir: Path) -> None:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["layer"])].append(float(entry["cosine_similarity"]))
    for layer, values in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=80, range=(-1.0, 1.0), density=True, color="#4C72B0", alpha=0.8)
        mean = float(torch.tensor(values, dtype=torch.float32).mean().item())
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(mean, color="#DD8452", linestyle="-", linewidth=1.5, label=f"mean={mean:.3f}")
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{args.model} {args.benchmark} {args.split} fold {args.fold} layer {layer}"
        )
        ax.legend()
        fig.tight_layout()
        fig_path = out_dir / f"layer_{layer}_cosine_hist.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)


def plot_layers_overlay(args: Args, entries: List[Dict[str, object]], out_dir: Path) -> None:
    """Plot multiple layers' density estimates on the same axes."""
    import numpy as np

    grouped: Dict[int, List[float]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["layer"])].append(float(entry["cosine_similarity"]))

    layers = sorted(grouped.keys())
    if not layers:
        return

    xs = np.linspace(-1.0, 1.0, 512)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, layer in enumerate(layers):
        values = np.array(grouped[layer], dtype=np.float32)
        if values.size == 0:
            continue
        # Simple density via histogram smoothing
        hist, bin_edges = np.histogram(values, bins=120, range=(-1.0, 1.0), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Smooth with a small Gaussian kernel via convolution
        try:
            from scipy.ndimage import gaussian_filter1d

            smooth = gaussian_filter1d(hist, sigma=1.5)
        except Exception:
            # fallback: no smoothing
            smooth = hist
        color = colors[i % len(colors)] if colors else None
        ax.plot(bin_centers, smooth, label=f"layer {layer}", color=color)

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="best")
    ax.set_title(f"{args.model} {args.benchmark} {args.split} fold {args.fold} overlay")
    fig.tight_layout()
    fig_path = out_dir / "overlay_layers_cosine.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main() -> None:
    global args
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.overlay_folds:
        entries_by_fold = compute_fold_overlay_entries(args)
        out_dir = build_overlay_folds_output_dir(args)
        all_entries = [entry for fold_entries in entries_by_fold.values() for entry in fold_entries]
        csv_path = out_dir / "cosine_similarities_overlay_folds.csv"
        summary_path = out_dir / "summary.json"
        write_csv(all_entries, csv_path)
        summary_data = summarize(
            all_entries,
            sorted({int(entry["layer"]) for entry in all_entries}),
        )
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        plot_folds_overlay(args, entries_by_fold, out_dir)
        if args.top_percentile:
            for fold in sorted(entries_by_fold):
                local_args = replace_fold(args, fold)
                metrics = load_fold_result_metrics(local_args, fold)
                experiment_selected = load_selected_layer(local_args)
                top_layers = select_top_percentile_layers(
                    metrics["val"], local_args.percentile, selected_layer=experiment_selected
                )
                if not top_layers:
                    logger.warning("No top-percentile layers found for fold %s", fold)
                    continue
                bundle = load_bundle(local_args)
                direction = load_direction(local_args)
                top_entries = compute_cosine_entries(local_args, bundle, direction, top_layers)
                plot_top_percentile_layers(
                    local_args, top_entries, metrics, out_dir, experiment_selected
                )
        logger.info("Wrote overlay-folds CSV to %s", csv_path)
        logger.info("Wrote overlay-folds summary to %s", summary_path)
        logger.info("Wrote overlay-folds plot to %s", out_dir)
        return

    out_dir = build_output_dir(args)
    if (
        args.top_percentile_val_test_overlay
        and not args.top_percentile
        and not args.overlay
    ):
        metrics = load_fold_result_metrics(args, args.fold)
        experiment_selected = load_selected_layer(args)
        top_layers = select_top_percentile_layers(
            metrics["val"], args.percentile, selected_layer=experiment_selected
        )
        if top_layers:
            val_entries = compute_cosine_entries_for_split(args, "validate", top_layers)
            test_entries = compute_cosine_entries_for_split(args, "test", top_layers)
            plot_top_percentile_val_test_overlay(
                args,
                val_entries,
                test_entries,
                metrics,
                out_dir,
                experiment_selected,
                top_layers,
            )
        else:
            logger.warning("No top-percentile layers found for val/test overlay on fold %s", args.fold)
        logger.info("Wrote plots to %s", out_dir)
        return

    bundle = load_bundle(args)
    if args.representation not in bundle.acts:
        raise KeyError(
            f"Representation {args.representation!r} not found in bundle acts. Available: {list(bundle.acts)}"
        )
    direction = load_direction(args)
    layers = resolve_layers(args, bundle[args.representation].shape[1], direction)
    entries = compute_cosine_entries(args, bundle, direction, layers)
    csv_path = out_dir / "cosine_similarities.csv"
    summary_path = out_dir / "summary.json"
    write_csv(entries, csv_path)
    summary_data = summarize(entries, layers)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    #plot_layer_histograms(args, entries, out_dir)
    if args.overlay:
        plot_layers_overlay(args, entries, out_dir)
    if args.top_percentile:
        metrics = load_fold_result_metrics(args, args.fold)
        experiment_selected = load_selected_layer(args)
        top_layers = select_top_percentile_layers(
            metrics["val"], args.percentile, selected_layer=experiment_selected
        )
        if top_layers:
            top_entries = compute_cosine_entries(args, bundle, direction, top_layers)
            plot_top_percentile_layers(args, top_entries, metrics, out_dir, experiment_selected)
        else:
            logger.warning("No top-percentile layers found for fold %s", args.fold)
    if args.top_percentile_val_test_overlay:
        metrics = load_fold_result_metrics(args, args.fold)
        experiment_selected = load_selected_layer(args)
        top_layers = select_top_percentile_layers(
            metrics["val"], args.percentile, selected_layer=experiment_selected
        )
        if top_layers:
            val_entries = compute_cosine_entries_for_split(args, "validate", top_layers)
            test_entries = compute_cosine_entries_for_split(args, "test", top_layers)
            plot_top_percentile_val_test_overlay(
                args,
                val_entries,
                test_entries,
                metrics,
                out_dir,
                experiment_selected,
                top_layers,
            )
        else:
            logger.warning("No top-percentile layers found for val/test overlay on fold %s", args.fold)
    logger.info("Wrote CSV to %s", csv_path)
    logger.info("Wrote summary to %s", summary_path)
    logger.info("Wrote plots to %s", out_dir)


if __name__ == "__main__":
    main()
