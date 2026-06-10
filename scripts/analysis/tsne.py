"""Cache-only tSNE visualization with dynamic selectors (ICSE26, Mutation_DS, OOD).

This script reads only precomputed artifacts:
  - selected layers from `results/.../results.json`
  - hidden states from `hidden_states/.../bundle.pt`

No model extraction/inference is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.manifold import TSNE

from scripts.analysis.viz_common import (
    LayerHighlight,
    apply_layer_axes_style,
    build_fold_grid,
    build_layer_grid,
    build_single_scatter,
    classify_labels_strict,
    cross_fold_dirname,
    dataset_results_path,
    highlighted_subplot_title_kwargs,
    layer_highlight_state,
    layer_tag,
    load_fold_results,
    load_task_correctness_index,
    load_test_bundle,
    parse_fold_selector,
    parse_layer_selector,
    parse_task_selector,
    resolve_all_folds,
    resolve_layers_for_fold,
    safe_task_id,
    scatter_correctness,
    scatter_task_implementations,
    set_layer_subplot_title,
    viz_dataset_label,
    viz_output_path_parts,
)
from src.hidden_states.cache import safe_model_dir


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
    background_coords: np.ndarray | None = None,
    sample_ids: Sequence[str] | None = None,
    add_legend: bool = True,
) -> None:
    if sample_ids is not None:
        scatter_task_implementations(
            ax=ax,
            coords=coords,
            sample_ids=sample_ids,
            is_correct=is_correct,
            background_coords=background_coords,
            add_legend=add_legend,
        )
    else:
        scatter_correctness(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            background_coords=background_coords,
            add_legend=add_legend,
        )
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    ax.grid(alpha=0.2)


def _make_render_tsne_layer_panel(use_implementation_markers: bool):
    def _render(
        ax,
        panel: Any,
        highlight: LayerHighlight,
        *,
        add_legend: bool = True,
    ) -> None:
        sample_ids = None
        if len(panel) == 6:
            layer, coords, is_correct, is_incorrect, background_coords, sample_ids = panel
        elif len(panel) == 5:
            layer, coords, is_correct, is_incorrect, background_coords = panel
        else:
            layer, coords, is_correct, is_incorrect = panel
            background_coords = None
        apply_layer_axes_style(ax, highlight)
        _plot_on_axis(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            background_coords=background_coords,
            sample_ids=sample_ids if use_implementation_markers else None,
            add_legend=add_legend,
        )
        set_layer_subplot_title(ax, layer, highlight)

    return _render


def _render_tsne_fold_panel(
    ax,
    panel: Any,
    highlight: LayerHighlight,
    *,
    add_legend: bool = True,
) -> None:
    fold, layer, coords, is_correct, is_incorrect = panel
    apply_layer_axes_style(ax, highlight)
    _plot_on_axis(
        ax=ax,
        coords=coords,
        is_correct=is_correct,
        is_incorrect=is_incorrect,
        add_legend=add_legend,
    )
    ax.set_title(
        f"fold={fold}, layer={layer}",
        **highlighted_subplot_title_kwargs(highlight),
    )


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
    implementation_markers: bool,
) -> Dict[str, object]:
    render_layer_panel = _make_render_tsne_layer_panel(implementation_markers)
    fold_results_by_fold: Dict[int, dict] = {}

    fold_data: Dict[int, Dict[str, object]] = {}
    for fold in folds:
        fold_results = load_fold_results(
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
        fold_results_by_fold[fold] = fold_results
        selected_layer = int(fold_results["selected_layer"])
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
                f"representation {representation!r} missing in bundle acts; "
                f"available={list(acts_map.keys())}"
            )
        acts = acts_map[representation].detach().cpu().numpy()
        sample_ids = list(bundle_blob["sample_ids"])
        task_ids = list(bundle_blob["task_ids"])
        labels = bundle_blob["labels"].detach().cpu().tolist()
        is_correct, is_incorrect, ambiguous = classify_labels_strict(labels, sample_ids)
        if ambiguous:
            preview = ", ".join(ambiguous[:10])
            raise ValueError(
                f"fold {fold}: found {len(ambiguous)} ambiguous labels (neither 0 nor 1). "
                f"First IDs: {preview}"
            )
        fold_data[fold] = {
            "fold_results": fold_results,
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
        resolved_layers_per_fold[fold] = resolve_layers_for_fold(
            layer_mode=layer_mode,
            explicit_layers=explicit_layers,
            selected_layer=selected_layer,
            num_layers=num_layers,
            fold_results=fold_data[fold]["fold_results"],
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

    dataset_label = viz_dataset_label(split_family)
    out_root = (
        output_dir
        / safe_model_dir(model)
        / Path(*viz_output_path_parts(split_family))
        / benchmark
    )
    outputs: List[str] = []

    correctness_index: Dict[Tuple[int, str], bool] | None = None
    if task_mode:
        correctness_index = load_task_correctness_index(
            split_family=split_family,
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
                is_correct_task, is_incorrect_task, ambiguous_task = classify_labels_strict(
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
                    sample_ids_task = [fold_data[fold]["sample_ids"][i] for i in idx]
                    panels.append(
                        (
                            layer,
                            _coords_for(fold, layer, task_id),
                            is_correct_task,
                            is_incorrect_task,
                            coords_all if use_background else None,
                            sample_ids_task,
                        )
                    )

                layer_tag_str = layer_tag(layer_mode, layers)
                task_safe = safe_task_id(task_id)
                mode_suffix = "" if task_view_mode == "local" else f"_{task_view_mode}"
                if use_background:
                    mode_suffix = f"{mode_suffix}_bg"
                lookup_key = (fold, task_id)
                if lookup_key not in correctness_index:
                    results_path = dataset_results_path(
                        split_family=split_family,
                        benchmark=benchmark,
                        data_root=data_root,
                    )
                    raise KeyError(
                        f"no dataset result for fold={fold}, task_id={task_id!r} "
                        f"(model={safe_model_dir(model)!r}, method={method!r}, "
                        f"framing={fit_framing!r}, rep_type={representation!r}, "
                        f"val_condition={val_framing!r}, test_condition={test_framing!r}, "
                        f"results_file={results_path})"
                    )
                outcome = "success" if correctness_index[lookup_key] else "fail"
                out_path = (
                    out_root
                    / f"fold_{fold}"
                    / "tasks"
                    / outcome
                    / f"fold_{fold}_task_{task_safe}_layer_{layer_tag_str}{mode_suffix}.png"
                )
                build_layer_grid(
                    panels=panels,
                    fold_results=fold_data[fold]["fold_results"],
                    fig_title=f"{dataset_label} {benchmark} | fold={fold} | task={task_id}",
                    out_path=out_path,
                    render_panel=render_layer_panel,
                )
                outputs.append(str(out_path))
    elif len(folds) == 1:
        fold = folds[0]
        layers = resolved_layers_per_fold[fold]
        if len(layers) == 1:
            layer = layers[0]
            highlight = layer_highlight_state(fold_data[fold]["fold_results"], layer)
            out_path = out_root / f"fold_{fold}" / f"fold_{fold}_layer_{layer}.png"
            build_single_scatter(
                plot_axes=lambda ax: _plot_on_axis(
                    ax=ax,
                    coords=_coords_for(fold, layer),
                    is_correct=fold_data[fold]["is_correct"],
                    is_incorrect=fold_data[fold]["is_incorrect"],
                ),
                title=f"{dataset_label} {benchmark} | fold={fold}, layer={layer}",
                out_path=out_path,
                highlight=highlight,
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
            layer_tag_str = layer_tag(layer_mode, layers)
            out_path = (
                out_root
                / f"fold_{fold}"
                / f"fold_{fold}_layer_{layer_tag_str}.png"
            )
            build_layer_grid(
                panels=panels,
                fold_results=fold_data[fold]["fold_results"],
                fig_title=f"{dataset_label} {benchmark} | fold={fold}",
                out_path=out_path,
                render_panel=render_layer_panel,
            )
            outputs.append(str(out_path))
    else:
        one_layer_each = all(len(resolved_layers_per_fold[f]) == 1 for f in folds)
        if one_layer_each:
            container = cross_fold_dirname(
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
            layer_tag_str = layer_tag(layer_mode, resolved_layers_per_fold[folds[0]])
            out_path = out_root / container / f"{container}_layer_{layer_tag_str}.png"
            build_fold_grid(
                panels=panels,
                fold_results_by_fold=fold_results_by_fold,
                fig_title=f"{dataset_label} {benchmark} | folds={folds}",
                out_path=out_path,
                render_panel=_render_tsne_fold_panel,
            )
            outputs.append(str(out_path))
        else:
            container = cross_fold_dirname(
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
                layer_tag_str = layer_tag(layer_mode, layers)
                out_path = (
                    out_root
                    / container
                    / f"fold_{fold}_layer_{layer_tag_str}.png"
                )
                build_layer_grid(
                    panels=panels,
                    fold_results=fold_data[fold]["fold_results"],
                    fig_title=f"{dataset_label} {benchmark} | fold={fold}",
                    out_path=out_path,
                    render_panel=render_layer_panel,
                )
                outputs.append(str(out_path))

    summary: Dict[str, object] = {
        "model": model,
        "split_family": split_family,
        "dataset_label": dataset_label,
        "benchmark": benchmark,
        "data_root": str(data_root),
        "folds": list(folds),
        "layer_mode": layer_mode,
        "explicit_layers": explicit_layers,
        "task_mode": task_mode,
        "explicit_task_ids": explicit_task_ids,
        "task_view_mode": task_view_mode,
        "show_background": show_background,
        "implementation_markers": implementation_markers,
        "seed": seed,
        "perplexity": perplexity,
        "per_task_perplexity": per_task_perplexity,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "output_paths": outputs,
    }
    if task_mode:
        summary["task_outcome_layout"] = "tasks_success_fail_flat"
        summary["dataset_results_path"] = str(
            dataset_results_path(
                split_family=split_family,
                benchmark=benchmark,
                data_root=data_root,
            )
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument(
        "--data-root",
        default="dataset/icse26",
        help="ICSE26 root (dataset/icse26) or Mutation_DS split/phase2 dir; "
             "used for --folds all manifests and --tasks outcome JSONL.",
    )
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
        help="One of: 'selected', 'relevant', 'all', or explicit layer indices (e.g. 2 5 12).",
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
    parser.add_argument(
        "--implementation-markers",
        action="store_true",
        help="In --tasks mode, use distinct markers per implementation id.",
    )
    parser.add_argument("--output-dir", default="visualizations/tsne")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    folds = parse_fold_selector(
        args.folds,
        split_family=args.split_family,
        data_root=data_root,
        benchmark=args.benchmark,
    )
    all_folds = resolve_all_folds(
        split_family=args.split_family,
        data_root=data_root,
        benchmark=args.benchmark,
    )
    folds_is_all = folds == all_folds
    layer_mode, explicit_layers = parse_layer_selector(args.layers)
    task_mode, explicit_task_ids = parse_task_selector(args.tasks)
    if args.show_background and args.task_view_mode != "global_subset":
        raise ValueError(
            "--show-background requires --task-view-mode global_subset"
        )
    if args.show_background and not task_mode:
        raise ValueError("--show-background requires --tasks")
    if args.implementation_markers and not task_mode:
        raise ValueError("--implementation-markers requires --tasks")

    summary = run_dynamic(
        hidden_states_root=Path(args.hidden_states_root),
        results_root=Path(args.results_root),
        data_root=data_root,
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
        implementation_markers=args.implementation_markers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
