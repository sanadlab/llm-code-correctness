#!/usr/bin/env python3
"""Pairwise score-diff analysis: cosine similarity + projection-score markers.

Purpose
-------
Extends pair-diff cosine analysis by attaching projection scores from ``detailed.json``.
For each correct×incorrect pair and layer, computes:

- ``score_diff`` = correct projection score − incorrect projection score
- ``cosine_similarity`` between (h_correct − h_incorrect) and the reading direction

Inputs
------
- Hidden-state bundles, reading directions (same as ``plot_pairdiff_reading_cosine.py``)
- Per-sample projection scores in ``results/.../fold_*/detailed.json``

Outputs
-------
- ``pairdiff_score_diff_cosine.csv``: pair metadata, both scores, score_diff, cosine
- Density overlay PNG with rug markers:
  - circle (o): score_diff >= 0 (correct beats this incorrect)
  - x: score_diff < 0 (this incorrect beats correct)

Key CLI flags
-------------
- ``--layers``: ``top-percentile`` (default), ``selected``, ``all``, or explicit list
- ``--csv-layers``: layers written to CSV (defaults to ``--layers``; use ``all`` for scatter)
- ``--percentile``: threshold for top-percentile layer selection (val pairwise_acc)

Related scripts
---------------
- ``run_pairdiff_score_diff_graphs.sh``: batch CSV + overlay generation per fold
- ``plot_pairdiff_score_diff_scatter.py``: jittered scatter plots from the CSV
- ``plot_pairdiff_task_max_score_diff.py``: task-level (1-of-N) score ranking variant

Usage
-----
::

    python scripts/analysis/plot_pairdiff_score_diff.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --benchmark bigcodebench --split test --fold 0 \\
        --layers top-percentile --csv-layers all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from plot_pairdiff_reading_cosine import (
    Args as CosineArgs,
    compute_task_pairs,
    load_bundle,
    load_direction,
    load_fold_result_metrics,
    load_selected_layer,
    resolve_layers,
    select_top_percentile_layers,
)
from src.hidden_states import HiddenStateBundle, safe_model_dir
from src.hidden_states.cache import vector_common_path

logger = logging.getLogger(__name__)

SPLIT_TO_DETAILED = {
    "test": "test",
    "validate": "val",
}

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
    "correct_score",
    "incorrect_score",
    "score_diff",
    "cosine_similarity",
]


@dataclass(frozen=True)
class Args(CosineArgs):
    csv_layers: Optional[str] = field(default=None)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pair score differences (correct - incorrect) and cosine similarities, "
            "then plot layer overlays with score-diff markers."
        )
    )
    parser.add_argument("--hidden-states-root", default="hidden_states_sohaila")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-source", default="icse26")
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
        default="top-percentile",
        help="top-percentile, selected, all, or comma-separated layer indices",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit analysis to the first N tasks with both correct and incorrect samples.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile threshold used when --layers top-percentile",
    )
    parser.add_argument(
        "--csv-layers",
        default=None,
        help="Layers written to CSV (default: same as --layers). Use 'all' for cross-fold scatter grids.",
    )
    parser.add_argument("--output-dir", default="outputs/phase2_task2_score_diff")
    args = parser.parse_args()
    return Args(
        hidden_states_root=Path(args.hidden_states_root),
        vectors_root=Path(args.vectors_root),
        results_root=Path(args.results_root),
        model=args.model,
        dataset_source=args.dataset_source,
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
        percentile=args.percentile,
        csv_layers=args.csv_layers,
    )


def resolve_analysis_layers(
    args: Args,
    direction,
    metrics: Dict[str, Dict[int, Dict[str, float]]],
) -> List[int]:
    if args.layers == "top-percentile":
        experiment_selected = load_selected_layer(args)
        layers = select_top_percentile_layers(
            metrics["val"], args.percentile, selected_layer=experiment_selected
        )
        if not layers:
            raise ValueError(f"No top-percentile layers found for fold {args.fold}")
        return layers
    return resolve_layers(args, 0, direction)


def layer_selection_label(args: Args, layers: List[int]) -> str:
    if args.layers == "top-percentile":
        return f"Top {args.percentile:.0f}th percentile layers"
    if args.layers == "selected":
        return f"Selected layer {layers[0]}"
    if args.layers == "all":
        return "All layers"
    return f"Layers {', '.join(str(layer) for layer in layers)}"


def overlay_plot_filename(args: Args, layers: List[int]) -> str:
    if args.layers == "top-percentile":
        return f"top_{int(args.percentile)}th_percentile_layers_cosine_score_diff_fold_{args.fold}.png"
    if args.layers == "selected":
        return f"selected_layer_{layers[0]}_cosine_score_diff_fold_{args.fold}.png"
    if args.layers == "all":
        return f"all_layers_cosine_score_diff_fold_{args.fold}.png"
    layer_tag = "_".join(str(layer) for layer in layers)
    return f"layers_{layer_tag}_cosine_score_diff_fold_{args.fold}.png"


def detailed_json_path(args: Args) -> Path:
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
    return args.results_root / common / eval_segment / f"fold_{args.fold}" / "detailed.json"


def load_detailed_json(args: Args) -> Tuple[List[int], Dict[str, object], Dict[str, List[float]]]:
    path = detailed_json_path(args)
    if not path.exists():
        raise FileNotFoundError(
            f"detailed.json not found: {path}\n"
            f"Check --results-root, --model, --benchmark, --method, --fit-framing, "
            f"--representation, --val-framing, --test-framing, --fold"
        )
    data = json.loads(path.read_text())
    split_key = SPLIT_TO_DETAILED.get(args.split)
    if split_key is None:
        raise ValueError(
            f"Unsupported split {args.split!r} for detailed.json lookup. "
            f"Expected one of {sorted(SPLIT_TO_DETAILED)}"
        )
    if split_key not in data:
        raise KeyError(f"detailed.json missing {split_key!r} split: {path}")
    split_data = data[split_key]
    layers = [int(layer) for layer in data["layers"]]
    score_map = {
        sample_id: scores
        for sample_id, scores in zip(split_data["sample_ids"], split_data["scores"])
    }
    return layers, split_data, score_map


def load_detailed_scores(args: Args) -> Tuple[List[int], Dict[str, List[float]]]:
    layers, _, score_map = load_detailed_json(args)
    return layers, score_map


def build_task_groups_from_detailed(
    split_data: Dict[str, object],
) -> Dict[str, List[Tuple[str, int]]]:
    task_groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for sample_id, task_id, label in zip(
        split_data["sample_ids"],
        split_data["task_ids"],
        split_data["labels"],
    ):
        task_groups[str(task_id)].append((str(sample_id), int(label)))
    return task_groups


def correct_highest_in_task(
    score_map: Dict[str, List[float]],
    task_groups: Dict[str, List[Tuple[str, int]]],
    task_id: str,
    correct_sample_id: str,
    layer: int,
    layer_to_index: Dict[int, int],
) -> bool:
    correct_score = score_at_layer(score_map, correct_sample_id, layer, layer_to_index)
    for sample_id, _label in task_groups[task_id]:
        if sample_id == correct_sample_id:
            continue
        if score_at_layer(score_map, sample_id, layer, layer_to_index) > correct_score:
            return False
    return True


def task_score_summary(
    score_map: Dict[str, List[float]],
    task_groups: Dict[str, List[Tuple[str, int]]],
    task_id: str,
    layer: int,
    layer_to_index: Dict[int, int],
) -> Tuple[float, float]:
    scores = [
        score_at_layer(score_map, sample_id, layer, layer_to_index)
        for sample_id, _label in task_groups[task_id]
    ]
    incorrect_scores = [
        score_at_layer(score_map, sample_id, layer, layer_to_index)
        for sample_id, label in task_groups[task_id]
        if label == 0
    ]
    return max(scores), max(incorrect_scores) if incorrect_scores else float("nan")


def score_at_layer(
    score_map: Dict[str, List[float]],
    sample_id: str,
    layer: int,
    layer_to_index: Dict[int, int],
) -> float:
    if sample_id not in score_map:
        raise KeyError(f"Sample {sample_id!r} not found in detailed.json scores")
    layer_idx = layer_to_index[layer]
    return float(score_map[sample_id][layer_idx])


def compute_pairdiff_entries(
    args: Args,
    bundle: HiddenStateBundle,
    direction,
    layers: List[int],
    score_map: Dict[str, List[float]],
    detailed_layers: List[int],
) -> List[Dict[str, object]]:
    acts = bundle[args.representation]
    if acts.dim() != 3:
        raise ValueError(f"Unexpected acts shape {tuple(acts.shape)}: expected [N, num_layers+1, hidden_dim]")
    if not (len(bundle.sample_ids) == len(bundle.task_ids) == len(bundle.labels) == acts.shape[0]):
        raise ValueError(
            "Mismatch between sample metadata and activations length: "
            f"{len(bundle.sample_ids)}, {len(bundle.task_ids)}, {len(bundle.labels)}, {acts.shape[0]}"
        )

    layer_to_index = {layer: idx for idx, layer in enumerate(detailed_layers)}
    missing_layers = [layer for layer in layers if layer not in layer_to_index]
    if missing_layers:
        raise ValueError(
            f"Layer(s) {missing_layers} not present in detailed.json layers {detailed_layers}"
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
                corr_sample = bundle.sample_ids[corr_idx]
                inc_sample = bundle.sample_ids[inc_idx]
                corr_score = score_at_layer(score_map, corr_sample, layer, layer_to_index)
                inc_score = score_at_layer(score_map, inc_sample, layer, layer_to_index)

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
                        "correct_sample_id": corr_sample,
                        "incorrect_sample_id": inc_sample,
                        "correct_score": corr_score,
                        "incorrect_score": inc_score,
                        "score_diff": corr_score - inc_score,
                        "cosine_similarity": float(cosine),
                    }
                )

    if valid_tasks == 0:
        raise ValueError("No tasks contained both correct and incorrect samples; cannot build pair differences.")
    logger.info("Built %s pairdiff entries from %s tasks", total_pairs, valid_tasks)
    return entries


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


def write_csv(entries: List[Dict[str, object]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)


def _smooth_density(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hist, bin_edges = np.histogram(values, bins=120, range=(-1.0, 1.0), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    try:
        from scipy.ndimage import gaussian_filter1d

        smooth = gaussian_filter1d(hist, sigma=1.5)
    except Exception:
        smooth = hist
    return bin_centers, smooth


def plot_layers_overlay_with_score_markers(
    args: Args,
    entries: List[Dict[str, object]],
    metrics: Dict[str, Dict[int, Dict[str, float]]],
    layers: List[int],
    out_dir: Path,
    experiment_selected_layer: int,
) -> None:
    test_metrics = metrics["test"]
    val_metrics = metrics["val"]
    selected_layers = sorted({int(entry["layer"]) for entry in entries})
    if not selected_layers:
        logger.warning("No layers selected for score-diff plot for fold %s", args.fold)
        return

    grouped_cosine: Dict[int, List[float]] = defaultdict(list)
    grouped_positive: Dict[int, List[float]] = defaultdict(list)
    grouped_negative: Dict[int, List[float]] = defaultdict(list)
    for entry in entries:
        layer = int(entry["layer"])
        cosine = float(entry["cosine_similarity"])
        grouped_cosine[layer].append(cosine)
        if float(entry["score_diff"]) >= 0.0:
            grouped_positive[layer].append(cosine)
        else:
            grouped_negative[layer].append(cosine)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    rug_y = 0.04

    for i, layer in enumerate(selected_layers):
        values = np.array(grouped_cosine[layer], dtype=np.float32)
        if values.size == 0:
            continue

        bin_centers, smooth = _smooth_density(values)
        color = colors[i % len(colors)] if colors else None
        line_width = 3.0 if layer == experiment_selected_layer else 1.5
        cosine_sim_mean = float(np.mean(values))
        val_acc = val_metrics.get(layer, {}).get("pairwise_acc", float("nan"))
        test_acc = test_metrics.get(layer, {}).get("one_of_4", float("nan"))
        label_text = (
            f"layer {layer} ({cosine_sim_mean:.3f} | {val_acc:.3f} | {test_acc:.3f})"
        )
        if layer == experiment_selected_layer:
            label_text += " <SELECTED>"
        ax.plot(bin_centers, smooth, label=label_text, color=color, linewidth=line_width)

        pos_x = grouped_positive[layer]
        neg_x = grouped_negative[layer]
        if pos_x:
            ax.scatter(
                pos_x,
                [rug_y] * len(pos_x),
                marker="o",
                s=10,
                alpha=0.25,
                color=color,
                linewidths=0,
            )
        if neg_x:
            ax.scatter(
                neg_x,
                [rug_y] * len(neg_x),
                marker="x",
                s=12,
                alpha=0.35,
                color=color,
                linewidths=0.6,
            )

    ax.scatter([], [], marker="o", color="black", s=30, label="score_diff >= 0")
    ax.scatter([], [], marker="x", color="black", s=30, label="score_diff < 0")

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(
        f"{layer_selection_label(args, layers)} cosine similarity overlay for fold {args.fold} "
        f"(x/o = score_diff sign)"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / overlay_plot_filename(args, layers)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote score-diff cosine overlay plot to %s", fig_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bundle = load_bundle(args)
    if args.representation not in bundle.acts:
        raise KeyError(
            f"Representation {args.representation!r} not found in bundle acts. Available: {list(bundle.acts)}"
        )
    direction = load_direction(args)
    detailed_layers, score_map = load_detailed_scores(args)

    metrics = load_fold_result_metrics(args, args.fold)
    experiment_selected = load_selected_layer(args)
    plot_layers = resolve_analysis_layers(args, direction, metrics)
    csv_layer_spec = args.csv_layers if args.csv_layers is not None else args.layers
    csv_layers = resolve_analysis_layers(
        replace(args, layers=csv_layer_spec), direction, metrics
    )
    logger.info(
        "Plot layers: %s | CSV layers: %s | experiment selected layer: %s",
        plot_layers,
        csv_layers,
        experiment_selected,
    )

    entries = compute_pairdiff_entries(
        args, bundle, direction, csv_layers, score_map, detailed_layers
    )
    out_dir = build_output_dir(args)
    csv_path = out_dir / "pairdiff_score_diff_cosine.csv"
    write_csv(entries, csv_path)
    plot_layer_set = set(plot_layers)
    plot_entries = [entry for entry in entries if int(entry["layer"]) in plot_layer_set]
    plot_layers_overlay_with_score_markers(
        args, plot_entries, metrics, plot_layers, out_dir, experiment_selected
    )
    logger.info("Wrote CSV to %s", csv_path)
    logger.info("Wrote plots to %s", out_dir)


if __name__ == "__main__":
    main()
