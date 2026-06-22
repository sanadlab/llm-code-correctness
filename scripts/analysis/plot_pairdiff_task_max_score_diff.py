#!/usr/bin/env python3
"""Task-level score ranking analysis: cosine similarity + 1-of-N score markers.

Purpose
-------
Like ``plot_pairdiff_score_diff.py``, computes cosine similarity for each
correct×incorrect pair, but marker semantics use **all implementations of the
task** from ``detailed.json`` (e.g. 1 correct + 3 incorrect on test):

- ``correct_highest_in_task = 1``: correct has the highest projection score in the task
- ``correct_highest_in_task = 0``: at least one incorrect beats the correct sample

This is stricter than pairwise score_diff and aligns with 1-of-N selection.

Inputs
------
- Hidden-state bundles, reading directions, ``detailed.json`` projection scores

Outputs
-------
- ``pairdiff_task_max_cosine.csv`` with columns including ``correct_score``,
  ``max_incorrect_score``, ``task_max_score``, ``correct_highest_in_task``,
  and ``cosine_similarity``

Related scripts
---------------
- ``plot_pairdiff_task_max_scatter.py``: scatter plots from this CSV
- ``run_pairdiff_task_max_scatter.sh``: batch CSV + scatter generation
- ``plot_pairdiff_score_diff.py``: pairwise (not task-level) score variant

Usage
-----
::

    python scripts/analysis/plot_pairdiff_task_max_score_diff.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --benchmark bigcodebench --split test --fold 0 \\
        --csv-layers all
"""

from __future__ import annotations

import csv
import logging
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from plot_pairdiff_reading_cosine import (
    compute_task_pairs,
    load_bundle,
    load_direction,
    load_fold_result_metrics,
    load_selected_layer,
    select_top_percentile_layers,
)
from plot_pairdiff_score_diff import (
    Args,
    build_output_dir,
    build_task_groups_from_detailed,
    correct_highest_in_task,
    load_detailed_json,
    parse_args,
    resolve_analysis_layers,
    score_at_layer,
    task_score_summary,
)

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
    "correct_score",
    "max_incorrect_score",
    "task_max_score",
    "correct_highest_in_task",
    "cosine_similarity",
]


def compute_pairdiff_task_max_entries(
    args: Args,
    bundle,
    direction,
    layers: List[int],
    score_map: Dict[str, List[float]],
    detailed_layers: List[int],
    task_groups: Dict[str, List[Tuple[str, int]]],
) -> List[Dict[str, object]]:
    import torch
    import torch.nn.functional as F

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
        if task_id not in task_groups:
            raise KeyError(f"Task {task_id!r} missing from detailed.json task groups")
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
            task_max_score, max_incorrect_score = task_score_summary(
                score_map, task_groups, task_id, layer, layer_to_index
            )

            for pair_i, cosine in enumerate(cosines.tolist()):
                corr_idx = correct_idx[pair_i // len(incorrect_idx)]
                inc_idx = incorrect_idx[pair_i % len(incorrect_idx)]
                corr_sample = bundle.sample_ids[corr_idx]
                inc_sample = bundle.sample_ids[inc_idx]
                corr_score = score_at_layer(score_map, corr_sample, layer, layer_to_index)
                correct_wins = correct_highest_in_task(
                    score_map,
                    task_groups,
                    task_id,
                    corr_sample,
                    layer,
                    layer_to_index,
                )

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
                        "max_incorrect_score": max_incorrect_score,
                        "task_max_score": task_max_score,
                        "correct_highest_in_task": int(correct_wins),
                        "cosine_similarity": float(cosine),
                    }
                )

    if valid_tasks == 0:
        raise ValueError("No tasks contained both correct and incorrect samples; cannot build pair differences.")
    logger.info("Built %s task-max pairdiff entries from %s tasks", total_pairs, valid_tasks)
    return entries


def write_csv(entries: List[Dict[str, object]], path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)


def process_fold(args: Args) -> Path:
    bundle = load_bundle(args)
    if args.representation not in bundle.acts:
        raise KeyError(
            f"Representation {args.representation!r} not found in bundle acts. Available: {list(bundle.acts)}"
        )
    direction = load_direction(args)
    detailed_layers, split_data, score_map = load_detailed_json(args)
    task_groups = build_task_groups_from_detailed(split_data)

    metrics = load_fold_result_metrics(args, args.fold)
    experiment_selected = load_selected_layer(args)
    plot_layers = resolve_analysis_layers(args, direction, metrics)
    csv_layer_spec = args.csv_layers if args.csv_layers is not None else args.layers
    csv_layers = resolve_analysis_layers(
        replace(args, layers=csv_layer_spec), direction, metrics
    )
    logger.info(
        "[fold %s] plot layers: %s | CSV layers: %s | experiment selected layer: %s",
        args.fold,
        plot_layers,
        csv_layers,
        experiment_selected,
    )

    entries = compute_pairdiff_task_max_entries(
        args, bundle, direction, csv_layers, score_map, detailed_layers, task_groups
    )
    out_dir = build_output_dir(args)
    csv_path = out_dir / "pairdiff_task_max_cosine.csv"
    write_csv(entries, csv_path)
    logger.info("[fold %s] wrote CSV to %s", args.fold, csv_path)
    return csv_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    folds: Sequence[int] = args.folds if args.folds is not None else [args.fold]
    for fold in folds:
        logger.info("Processing fold %s", fold)
        process_fold(replace(args, fold=fold))

    logger.info("Finished %s fold(s)", len(folds))


if __name__ == "__main__":
    main()
