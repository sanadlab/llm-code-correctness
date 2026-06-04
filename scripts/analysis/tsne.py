"""ICSE2026 cache-only tSNE visualization with dynamic selectors.

This script reads only precomputed artifacts:
  - selected layers from `results/.../results.json`
  - hidden states from `hidden_states/.../bundle.pt`

No model extraction/inference is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.hidden_states.cache import bundle_output_dir, safe_model_dir


def _read_selected_layer(
    *,
    results_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    method: str,
    fit_framing: str,
    representation: str,
    val_framing: str,
    test_framing: str,
    fold: int,
) -> int:
    results_path = (
        results_root
        / safe_model_dir(model)
        / split_family
        / benchmark
        / "id"
        / method
        / fit_framing
        / representation
        / f"val={val_framing}__test={test_framing}"
        / f"fold_{fold}"
        / "results.json"
    )
    if not results_path.is_file():
        raise FileNotFoundError(
            f"missing selected-layer file: {results_path}"
        )
    with open(results_path) as f:
        payload = json.load(f)
    if "selected_layer" not in payload:
        raise KeyError(f"`selected_layer` missing in {results_path}")
    return int(payload["selected_layer"])


def _resolve_all_folds(data_root: Path, benchmark: str) -> List[int]:
    manifest_path = data_root / benchmark / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest for --folds all: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    num_folds = int(manifest["num_folds"])
    return list(range(num_folds))


def _parse_fold_selector(folds_arg: Sequence[str], data_root: Path, benchmark: str) -> List[int]:
    if len(folds_arg) == 1 and folds_arg[0] == "all":
        return _resolve_all_folds(data_root, benchmark)
    try:
        parsed = sorted({int(x) for x in folds_arg})
    except ValueError as exc:
        raise ValueError("--folds must be integers or 'all'") from exc
    if not parsed:
        raise ValueError("--folds resolved to an empty set")
    if any(f < 0 for f in parsed):
        raise ValueError("--folds must be non-negative")
    return parsed


def _parse_layer_selector(layers_arg: Sequence[str]) -> Tuple[str, List[int] | None]:
    if len(layers_arg) == 1 and layers_arg[0] in {"selected", "all"}:
        return layers_arg[0], None
    try:
        parsed = sorted({int(x) for x in layers_arg})
    except ValueError as exc:
        raise ValueError("--layers must be integers, 'selected', or 'all'") from exc
    if not parsed:
        raise ValueError("--layers resolved to an empty set")
    if any(l < 0 for l in parsed):
        raise ValueError("--layers must be non-negative")
    return "explicit", parsed


def _parse_task_selector(tasks_arg: Sequence[str] | None) -> Tuple[bool, List[str] | None]:
    if not tasks_arg:
        return False, None
    if len(tasks_arg) == 1 and tasks_arg[0] == "all":
        return True, None
    task_ids = sorted({x for x in tasks_arg if x})
    if not task_ids:
        raise ValueError("--tasks resolved to an empty set")
    return True, task_ids


def _safe_task_id(task_id: str) -> str:
    # ICSE task ids look like "<Benchmark>/<numeric_id>".
    # Since output already sits under <benchmark>/, keep only the right segment
    # for cleaner names (e.g. "BigCodeBench/208" -> "208").
    leaf = task_id.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", leaf)


def _dataset_results_path(data_root: Path, benchmark: str) -> Path:
    return data_root / benchmark / f"icse26_{benchmark}.jsonl"


def _load_task_correctness_index(
    *,
    data_root: Path,
    benchmark: str,
    model: str,
    method: str,
    fit_framing: str,
    representation: str,
    val_framing: str,
    test_framing: str,
) -> Dict[Tuple[int, str], bool]:
    results_path = _dataset_results_path(data_root, benchmark)
    if not results_path.is_file():
        raise FileNotFoundError(f"missing dataset results file: {results_path}")

    model_safe = safe_model_dir(model)
    index: Dict[Tuple[int, str], bool] = {}
    with open(results_path) as f:
        for line in f:
            row = json.loads(line)
            if row["model"] != model_safe:
                continue
            if row["method"] != method:
                continue
            if row["framing"] != fit_framing:
                continue
            if row["rep_type"] != representation:
                continue
            if row["val_condition"] != val_framing:
                continue
            if row["test_condition"] != test_framing:
                continue
            key = (int(row["fold"]), row["task_id"])
            if key in index:
                raise ValueError(f"duplicate dataset entry for {key} in {results_path}")
            index[key] = bool(row["correct"])
    return index


def _load_test_bundle(
    *,
    hidden_states_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    fold: int,
    test_framing: str,
) -> Dict[str, object]:
    out_dir = bundle_output_dir(
        output_root=hidden_states_root,
        model_name=model,
        split_family=split_family,
        benchmark=benchmark,
        distribution="id",
        data_split="test",
        framing=test_framing,
        fold=fold,
    )
    bundle_path = out_dir / "bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"missing cached test bundle: {bundle_path}")
    # Use direct torch.load for compatibility with older torch versions.
    return torch.load(bundle_path, map_location="cpu")


def _classify_labels_strict(
    labels: Sequence[int], sample_ids: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return masks for correct/incorrect and a list of ambiguous sample IDs."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.shape[0] != len(sample_ids):
        raise ValueError("labels/sample_ids length mismatch")

    is_correct = labels_arr == 1
    is_incorrect = labels_arr == 0
    ambiguous = [
        sample_ids[i]
        for i in range(labels_arr.shape[0])
        if not (is_correct[i] or is_incorrect[i])
    ]
    return is_correct, is_incorrect, ambiguous


def _compute_tsne(
    features: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    learning_rate: float,
    max_iter: int,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"expected 2D features, got shape={features.shape}")
    if features.shape[0] < 2:
        raise ValueError("need at least 2 samples for tSNE")
    if perplexity >= features.shape[0]:
        raise ValueError(
            f"perplexity ({perplexity}) must be < n_samples ({features.shape[0]})"
        )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        learning_rate=learning_rate,
        init="pca",
        max_iter=max_iter,
        n_jobs=1,
    )
    return tsne.fit_transform(features)


def _plot_on_axis(
    *,
    ax,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
    title: str,
    background_coords: np.ndarray | None = None,
) -> None:
    if background_coords is not None and background_coords.shape[0] > 0:
        ax.scatter(
            background_coords[:, 0],
            background_coords[:, 1],
            s=6,
            alpha=0.2,
            color="lightgray",
            label="fold background",
            zorder=1,
        )
    ax.scatter(
        coords[is_incorrect, 0],
        coords[is_incorrect, 1],
        s=12,
        alpha=1.0,
        color="salmon",
        label="incorrect (label=0)",
        zorder=2,
    )
    ax.scatter(
        coords[is_correct, 0],
        coords[is_correct, 1],
        s=12,
        alpha=1.0,
        color="skyblue",
        label="correct (label=1)",
        zorder=2,
    )
    ax.set_title(title)
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)


def _grid_shape(n: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _save_figure(fig, out_path: Path) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_single_scatter(
    *,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_on_axis(
        ax=ax,
        coords=coords,
        is_correct=is_correct,
        is_incorrect=is_incorrect,
        title=title,
    )
    _save_figure(fig, out_path)


def _build_layer_grid(
    *,
    panels: Sequence[
        Tuple[int, np.ndarray, np.ndarray, np.ndarray]
        | Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
    ],
    selected_layer: int,
    fig_title: str,
    out_path: Path,
) -> None:
    rows, cols = _grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_1d = np.array(axes).reshape(-1)
    for i, panel in enumerate(panels):
        if len(panel) == 5:
            layer, coords, is_correct, is_incorrect, background_coords = panel
        else:
            layer, coords, is_correct, is_incorrect = panel
            background_coords = None
        suffix = " (selected)" if layer == selected_layer else ""
        _plot_on_axis(
            ax=axes_1d[i],
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            title=f"layer={layer}{suffix}",
            background_coords=background_coords,
        )
    for j in range(len(panels), len(axes_1d)):
        axes_1d[j].axis("off")
    fig.suptitle(fig_title, fontsize=14)
    _save_figure(fig, out_path)


def _build_fold_grid(
    *,
    panels: Sequence[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    fig_title: str,
    out_path: Path,
) -> None:
    rows, cols = _grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_1d = np.array(axes).reshape(-1)
    for i, (fold, layer, coords, is_correct, is_incorrect) in enumerate(panels):
        _plot_on_axis(
            ax=axes_1d[i],
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            title=f"fold={fold}, layer={layer}",
        )
    for j in range(len(panels), len(axes_1d)):
        axes_1d[j].axis("off")
    fig.suptitle(fig_title, fontsize=14)
    _save_figure(fig, out_path)


def _resolve_layers_for_fold(
    *,
    layer_mode: str,
    explicit_layers: List[int] | None,
    selected_layer: int,
    num_layers: int,
) -> List[int]:
    if layer_mode == "selected":
        return [selected_layer]
    if layer_mode == "all":
        return list(range(num_layers))
    assert explicit_layers is not None
    bad = [l for l in explicit_layers if l >= num_layers]
    if bad:
        raise ValueError(
            f"requested layer(s) {bad} not present; bundle has layers [0, {num_layers - 1}]"
        )
    return list(explicit_layers)


def _layer_tag(layer_mode: str, layers: List[int]) -> str:
    if layer_mode == "selected":
        return str(layers[0])
    if layer_mode == "all":
        return "all"
    return "_".join(str(x) for x in layers)


def _cross_fold_dirname(
    *,
    folds: Sequence[int],
    folds_is_all: bool,
    layer_mode: str,
) -> str:
    # Any explicit layer selection is treated as a custom request bucket.
    if layer_mode == "explicit":
        return "custom"
    if folds_is_all:
        return "fold_all"
    if len(folds) > 1:
        return "custom"
    return f"fold_{folds[0]}"


def run_dynamic(
    *,
    hidden_states_root: Path,
    results_root: Path,
    data_root: Path,
    benchmark: str,
    model: str,
    folds: Sequence[int],
    layer_mode: str,
    explicit_layers: List[int] | None,
    split_family: str,
    method: str,
    fit_framing: str,
    representation: str,
    val_framing: str,
    test_framing: str,
    seed: int,
    perplexity: float,
    per_task_perplexity: float,
    learning_rate: float,
    max_iter: int,
    output_dir: Path,
    folds_is_all: bool,
    task_mode: bool,
    explicit_task_ids: List[str] | None,
    task_view_mode: str,
    show_background: bool,
) -> Dict[str, object]:
    # Cache fold-level data for reuse across layer/fold panels.
    fold_data: Dict[int, Dict[str, object]] = {}
    for fold in folds:
        selected_layer = _read_selected_layer(
            results_root=results_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            method=method,
            fit_framing=fit_framing,
            representation=representation,
            val_framing=val_framing,
            test_framing=test_framing,
            fold=fold,
        )
        bundle_blob = _load_test_bundle(
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
                f"representation {representation!r} missing in bundle acts; "
                f"available={list(acts_map.keys())}"
            )
        acts = acts_map[representation].detach().cpu().numpy()
        sample_ids = list(bundle_blob["sample_ids"])
        task_ids = list(bundle_blob["task_ids"])
        labels = bundle_blob["labels"].detach().cpu().tolist()
        is_correct, is_incorrect, ambiguous = _classify_labels_strict(labels, sample_ids)
        if ambiguous:
            preview = ", ".join(ambiguous[:10])
            raise ValueError(
                f"fold {fold}: found {len(ambiguous)} ambiguous labels (neither 0 nor 1). "
                f"First IDs: {preview}"
            )
        fold_data[fold] = {
            "selected_layer": selected_layer,
            "acts": acts,
            "sample_ids": sample_ids,
            "task_ids": task_ids,
            "labels": labels,
            "is_correct": is_correct,
            "is_incorrect": is_incorrect,
        }

    resolved_layers_per_fold: Dict[int, List[int]] = {}
    for fold in folds:
        acts = fold_data[fold]["acts"]
        selected_layer = int(fold_data[fold]["selected_layer"])
        num_layers = int(acts.shape[1])
        resolved_layers_per_fold[fold] = _resolve_layers_for_fold(
            layer_mode=layer_mode,
            explicit_layers=explicit_layers,
            selected_layer=selected_layer,
            num_layers=num_layers,
        )

    tsne_cache: Dict[Tuple[int, int, str, str], np.ndarray] = {}
    task_index_by_fold: Dict[int, Dict[str, List[int]]] = {
        fold: {} for fold in folds
    }
    for fold in folds:
        for i, tid in enumerate(fold_data[fold]["task_ids"]):
            task_index_by_fold[fold].setdefault(tid, []).append(i)

    def _coords_for(fold: int, layer: int, task_id: str | None = None) -> np.ndarray:
        key = (
            fold,
            layer,
            task_id if task_id is not None else "__all__",
            task_view_mode if task_id is not None else "__all__",
        )
        if key in tsne_cache:
            return tsne_cache[key]

        acts = fold_data[fold]["acts"]
        if task_id is None:
            features = acts[:, layer, :]
            effective_perplexity = perplexity
        else:
            idx = task_index_by_fold[fold].get(task_id, [])
            if len(idx) != 4:
                raise ValueError(
                    f"fold {fold} task {task_id}: expected 4 candidates, got {len(idx)}"
                )
            if task_view_mode == "global_subset":
                # Fit on all fold points at this layer, then keep this task's 4 coords.
                coords_all = _coords_for(fold, layer, None)
                tsne_cache[key] = coords_all[idx, :]
                return tsne_cache[key]
            features = acts[idx, layer, :]
            effective_perplexity = per_task_perplexity

        tsne_cache[key] = _compute_tsne(
            features,
            seed=seed,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
        )
        return tsne_cache[key]

    out_root = output_dir / safe_model_dir(model) / "icse26" / benchmark
    outputs: List[str] = []

    correctness_index: Dict[Tuple[int, str], bool] | None = None
    if task_mode:
        correctness_index = _load_task_correctness_index(
            data_root=data_root,
            benchmark=benchmark,
            model=model,
            method=method,
            fit_framing=fit_framing,
            representation=representation,
            val_framing=val_framing,
            test_framing=test_framing,
        )

    if task_mode:
        assert correctness_index is not None
        for fold in folds:
            layers = resolved_layers_per_fold[fold]
            fold_task_ids = sorted(set(fold_data[fold]["task_ids"]))
            tasks = fold_task_ids if explicit_task_ids is None else explicit_task_ids
            missing = [t for t in tasks if t not in set(fold_task_ids)]
            if missing:
                raise ValueError(
                    f"fold {fold}: requested task ids not found: {missing[:5]}"
                )
            for task_id in tasks:
                idx = [i for i, tid in enumerate(fold_data[fold]["task_ids"]) if tid == task_id]
                labels_task = [fold_data[fold]["labels"][i] for i in idx]
                is_correct_task, is_incorrect_task, ambiguous_task = _classify_labels_strict(
                    labels_task, [fold_data[fold]["sample_ids"][i] for i in idx]
                )
                if ambiguous_task:
                    raise ValueError(
                        f"fold {fold} task {task_id}: ambiguous labels: {ambiguous_task[:5]}"
                    )

                use_background = (
                    show_background
                    and task_view_mode == "global_subset"
                )
                panels = []
                for layer in layers:
                    coords_all = None
                    if use_background:
                        coords_all = _coords_for(fold, layer, None)
                    panels.append(
                        (
                            layer,
                            _coords_for(fold, layer, task_id),
                            is_correct_task,
                            is_incorrect_task,
                            coords_all if use_background else None,
                        )
                    )

                layer_tag = _layer_tag(layer_mode, layers)
                task_safe = _safe_task_id(task_id)
                mode_suffix = "" if task_view_mode == "local" else f"_{task_view_mode}"
                if use_background:
                    mode_suffix = f"{mode_suffix}_bg"
                lookup_key = (fold, task_id)
                if lookup_key not in correctness_index:
                    raise KeyError(
                        f"no dataset result for fold={fold}, task_id={task_id!r} "
                        f"(model={safe_model_dir(model)!r}, method={method!r}, "
                        f"framing={fit_framing!r}, rep_type={representation!r}, "
                        f"val_condition={val_framing!r}, test_condition={test_framing!r})"
                    )
                outcome = "success" if correctness_index[lookup_key] else "fail"
                out_path = (
                    out_root
                    / f"fold_{fold}"
                    / "tasks"
                    / outcome
                    / f"fold_{fold}_task_{task_safe}_layer_{layer_tag}{mode_suffix}.png"
                )
                _build_layer_grid(
                    panels=panels,
                    selected_layer=int(fold_data[fold]["selected_layer"]),
                    fig_title=f"ICSE26 {benchmark} | fold={fold} | task={task_id}",
                    out_path=out_path,
                )
                outputs.append(str(out_path))
    elif len(folds) == 1:
        fold = folds[0]
        layers = resolved_layers_per_fold[fold]
        if len(layers) == 1:
            layer = layers[0]
            out_path = out_root / f"fold_{fold}" / f"fold_{fold}_layer_{layer}.png"
            _build_single_scatter(
                coords=_coords_for(fold, layer),
                is_correct=fold_data[fold]["is_correct"],
                is_incorrect=fold_data[fold]["is_incorrect"],
                title=f"ICSE26 {benchmark} | fold={fold}, layer={layer}",
                out_path=out_path,
            )
            outputs.append(str(out_path))
        else:
            panels = []
            for layer in layers:
                panels.append(
                    (
                        layer,
                        _coords_for(fold, layer),
                        fold_data[fold]["is_correct"],
                        fold_data[fold]["is_incorrect"],
                    )
                )
            layer_tag = _layer_tag(layer_mode, layers)
            out_path = (
                out_root
                / f"fold_{fold}"
                / f"fold_{fold}_layer_{layer_tag}.png"
            )
            _build_layer_grid(
                panels=panels,
                selected_layer=int(fold_data[fold]["selected_layer"]),
                fig_title=f"ICSE26 {benchmark} | fold={fold}",
                out_path=out_path,
            )
            outputs.append(str(out_path))
    else:
        # One layer per fold -> one fold-grid figure.
        one_layer_each = all(len(resolved_layers_per_fold[f]) == 1 for f in folds)
        if one_layer_each:
            container = _cross_fold_dirname(
                folds=folds,
                folds_is_all=folds_is_all,
                layer_mode=layer_mode,
            )
            panels = []
            for fold in folds:
                layer = resolved_layers_per_fold[fold][0]
                panels.append(
                    (
                        fold,
                        layer,
                        _coords_for(fold, layer),
                        fold_data[fold]["is_correct"],
                        fold_data[fold]["is_incorrect"],
                    )
                )
            layer_tag = _layer_tag(layer_mode, resolved_layers_per_fold[folds[0]])
            out_path = out_root / container / f"{container}_layer_{layer_tag}.png"
            _build_fold_grid(
                panels=panels,
                fig_title=f"ICSE26 {benchmark} | folds={folds}",
                out_path=out_path,
            )
            outputs.append(str(out_path))
        else:
            # Multi-fold and multi-layer -> one layer-grid figure per fold.
            container = _cross_fold_dirname(
                folds=folds,
                folds_is_all=folds_is_all,
                layer_mode=layer_mode,
            )
            for fold in folds:
                layers = resolved_layers_per_fold[fold]
                panels = []
                for layer in layers:
                    panels.append(
                        (
                            layer,
                            _coords_for(fold, layer),
                            fold_data[fold]["is_correct"],
                            fold_data[fold]["is_incorrect"],
                        )
                    )
                layer_tag = _layer_tag(layer_mode, layers)
                out_path = (
                    out_root
                    / container
                    / f"fold_{fold}_layer_{layer_tag}.png"
                )
                _build_layer_grid(
                    panels=panels,
                    selected_layer=int(fold_data[fold]["selected_layer"]),
                    fig_title=f"ICSE26 {benchmark} | fold={fold}",
                    out_path=out_path,
                )
                outputs.append(str(out_path))

    summary: Dict[str, object] = {
        "model": model,
        "benchmark": benchmark,
        "data_root": str(data_root),
        "folds": list(folds),
        "layer_mode": layer_mode,
        "explicit_layers": explicit_layers,
        "task_mode": task_mode,
        "explicit_task_ids": explicit_task_ids,
        "task_view_mode": task_view_mode,
        "show_background": show_background,
        "seed": seed,
        "perplexity": perplexity,
        "per_task_perplexity": per_task_perplexity,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "output_paths": outputs,
    }
    if task_mode:
        summary["task_outcome_layout"] = "tasks_success_fail_flat"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument("--data-root", default="dataset/icse26")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Either explicit fold indices (e.g. 0 3) or 'all'.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["selected"],
        help="One of: 'selected', 'all', or explicit layer indices (e.g. 2 5 12).",
    )
    parser.add_argument("--split-family", default="icse26")
    parser.add_argument("--hidden-states-root", default="hidden_states")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="Perplexity for non-task visualizations.")
    parser.add_argument("--per-task-perplexity", type=float, default=2.0,
                        help="Perplexity for --tasks mode (default tuned for 4 points/task).")
    parser.add_argument("--learning-rate", type=float, default=200.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Enable per-task mode: use 'all' or explicit task_id list.")
    parser.add_argument(
        "--task-view-mode",
        choices=["local", "global_subset"],
        default="local",
        help="Per-task behavior: local=fit tSNE on 4 task points; "
             "global_subset=fit on all fold points then display task subset.",
    )
    parser.add_argument(
        "--show-background",
        action="store_true",
        help="In global_subset per-task mode, draw faint fold background points "
             "behind the highlighted task points.",
    )
    parser.add_argument("--output-dir", default="visualizations/tsne")
    args = parser.parse_args()

    folds = _parse_fold_selector(
        folds_arg=args.folds,
        data_root=Path(args.data_root),
        benchmark=args.benchmark,
    )
    all_folds = _resolve_all_folds(Path(args.data_root), args.benchmark)
    folds_is_all = folds == all_folds
    layer_mode, explicit_layers = _parse_layer_selector(args.layers)
    task_mode, explicit_task_ids = _parse_task_selector(args.tasks)
    if args.show_background and args.task_view_mode != "global_subset":
        raise ValueError(
            "--show-background requires --task-view-mode global_subset"
        )
    if args.show_background and not task_mode:
        raise ValueError("--show-background requires --tasks")

    summary = run_dynamic(
        hidden_states_root=Path(args.hidden_states_root),
        results_root=Path(args.results_root),
        data_root=Path(args.data_root),
        benchmark=args.benchmark,
        model=args.model,
        folds=folds,
        layer_mode=layer_mode,
        explicit_layers=explicit_layers,
        split_family=args.split_family,
        method=args.method,
        fit_framing=args.fit_framing,
        representation=args.representation,
        val_framing=args.val_framing,
        test_framing=args.test_framing,
        seed=args.seed,
        perplexity=args.perplexity,
        per_task_perplexity=args.per_task_perplexity,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        output_dir=Path(args.output_dir),
        folds_is_all=folds_is_all,
        task_mode=task_mode,
        explicit_task_ids=explicit_task_ids,
        task_view_mode=args.task_view_mode,
        show_background=args.show_background,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
