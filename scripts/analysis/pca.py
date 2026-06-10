"""Cache-only PCA visualization with reading-vector overlays (ICSE26, Mutation_DS, OOD).

This script reads only precomputed artifacts:
  - selected layers from `results/.../results.json`
  - optional correctness-direction scores from `results/.../detailed.json`
  - hidden states from `hidden_states/.../bundle.pt`
  - per-layer reading vectors from `vectors/.../direction.pt`

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
from sklearn.decomposition import PCA

from scripts.analysis.viz_common import (
    LayerHighlight,
    apply_layer_axes_style,
    approach_filename_suffix,
    approach_norm_filename_suffix,
    build_fold_grid,
    build_layer_grid,
    build_single_scatter,
    classify_labels_strict,
    cross_fold_dirname,
    dataset_results_path,
    highlighted_subplot_title_kwargs,
    layer_highlight_state,
    layer_tag,
    load_direction_vectors,
    load_fold_detailed,
    load_fold_results,
    load_task_correctness_index,
    load_test_bundle,
    parse_approach_selector,
    parse_fold_selector,
    parse_layer_selector,
    parse_projection_source,
    parse_task_selector,
    pca_approach_dirname,
    test_projection_scores,
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

_ARROW_DISPLAY_FRACTION = 0.3


def _reading_vector(
    direction_vectors: Dict[int, np.ndarray],
    layer: int,
) -> np.ndarray:
    if layer not in direction_vectors:
        raise KeyError(f"layer {layer} missing in direction vectors")
    return direction_vectors[layer]


def _compute_pca_approach_a(
    features: np.ndarray,
    reading: np.ndarray,
    *,
    normalize_vector: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if features.ndim != 2:
        raise ValueError(f"expected 2D features, got shape={features.shape}")
    if features.shape[0] < 2:
        raise ValueError("need at least 2 samples for PCA")

    v = reading / np.linalg.norm(reading) if normalize_vector else reading
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(features)
    arrow = pca.components_ @ v
    v_norm = float(np.linalg.norm(v))
    captured = float(np.linalg.norm(arrow) / v_norm) if v_norm > 0 else 0.0
    return coords, arrow, captured


def _compute_pca_approach_b(
    features: np.ndarray,
    reading: np.ndarray,
    *,
    normalize_vector: bool,
    seed: int,
    x_scores: np.ndarray | None = None,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"expected 2D features, got shape={features.shape}")
    if features.shape[0] < 1:
        raise ValueError("need at least 1 sample for PCA approach B")

    r_norm = float(np.linalg.norm(reading))
    if r_norm == 0:
        raise ValueError("reading vector has zero norm")
    u = reading / r_norm
    v = u if normalize_vector else reading
    if x_scores is not None:
        if x_scores.shape[0] != features.shape[0]:
            raise ValueError("x_scores/features length mismatch")
        x = x_scores.astype(np.float64)
    else:
        x = features @ v
    xc = features - features.mean(axis=0)
    w = xc - (xc @ u)[:, None] * u[None, :]
    if np.allclose(w, 0.0) or float(np.std(w)) < 1e-12:
        y = np.zeros(features.shape[0], dtype=np.float64)
    else:
        y = PCA(n_components=1, random_state=seed).fit_transform(w)[:, 0]
    return np.stack([x, y], axis=1)


def _panel_data_span(coords: np.ndarray) -> float:
    """Largest axis span of the plotted points (used for display-only arrow scaling)."""
    if coords.shape[0] == 0:
        return 1.0
    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    span = max(x_span, y_span)
    if span < 1e-12:
        span = max(
            float(np.std(coords[:, 0])),
            float(np.std(coords[:, 1])),
            1e-6,
        )
    return max(span, 1e-9)


def _arrow_display_vector(coords: np.ndarray, arrow: np.ndarray) -> np.ndarray:
    """Scale arrow to a fixed fraction of the panel data range (display only)."""
    arrow_norm = float(np.linalg.norm(arrow))
    if arrow_norm < 1e-12:
        return np.zeros(2, dtype=np.float64)
    span = _panel_data_span(coords)
    return arrow * (_ARROW_DISPLAY_FRACTION * span / arrow_norm)


def _set_panel_limits(ax, coords: np.ndarray, display_arrow: np.ndarray) -> None:
    """Fit axes to data plus origin and arrow tip so the arrow stays visible."""
    span = _panel_data_span(coords)
    margin = 0.15 * span
    xs = np.concatenate([coords[:, 0], [0.0, display_arrow[0]]])
    ys = np.concatenate([coords[:, 1], [0.0, display_arrow[1]]])
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)


def _scatter_pca_points(
    *,
    ax,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
    sample_ids: Sequence[str] | None = None,
    add_legend: bool = True,
) -> None:
    if sample_ids is not None:
        scatter_task_implementations(
            ax=ax,
            coords=coords,
            sample_ids=sample_ids,
            is_correct=is_correct,
            add_legend=add_legend,
        )
    else:
        scatter_correctness(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            add_legend=add_legend,
        )


def _plot_pca_approach_a_axis(
    *,
    ax,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
    arrow: np.ndarray,
    sample_ids: Sequence[str] | None = None,
    add_legend: bool = True,
) -> None:
    display_arrow = _arrow_display_vector(coords, arrow)
    _scatter_pca_points(
        ax=ax,
        coords=coords,
        is_correct=is_correct,
        is_incorrect=is_incorrect,
        sample_ids=sample_ids,
        add_legend=add_legend,
    )
    if float(np.linalg.norm(display_arrow)) > 0:
        ax.annotate(
            "",
            xy=(display_arrow[0], display_arrow[1]),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            zorder=3,
        )
    _set_panel_limits(ax, coords, display_arrow)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)


def _plot_pca_approach_b_axis(
    *,
    ax,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
    normalize_vector: bool,
    sample_ids: Sequence[str] | None = None,
    add_legend: bool = True,
) -> None:
    _scatter_pca_points(
        ax=ax,
        coords=coords,
        is_correct=is_correct,
        is_incorrect=is_incorrect,
        sample_ids=sample_ids,
        add_legend=add_legend,
    )
    xlabel = "correctness direction (reading vector)"
    if normalize_vector:
        xlabel += " (unit-normalized)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("top residual PC (⊥ correctness)")
    ax.grid(alpha=0.2)


def _make_render_pca_a_panel(normalize_vector: bool, use_implementation_markers: bool):
    def _render(
        ax,
        panel: Any,
        highlight: LayerHighlight,
        *,
        add_legend: bool = True,
    ) -> None:
        sample_ids = None
        if len(panel) == 7:
            layer, coords, is_correct, is_incorrect, arrow, captured, sample_ids = panel
        else:
            layer, coords, is_correct, is_incorrect, arrow, captured = panel
        apply_layer_axes_style(ax, highlight)
        _plot_pca_approach_a_axis(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            arrow=arrow,
            sample_ids=sample_ids if use_implementation_markers else None,
            add_legend=add_legend,
        )
        set_layer_subplot_title(
            ax,
            layer,
            highlight,
            extra=f"captured={captured:.3f}",
        )

    return _render


def _make_render_pca_b_panel(normalize_vector: bool, use_implementation_markers: bool):
    def _render(
        ax,
        panel: Any,
        highlight: LayerHighlight,
        *,
        add_legend: bool = True,
    ) -> None:
        sample_ids = None
        if len(panel) == 5:
            layer, coords, is_correct, is_incorrect, sample_ids = panel
        else:
            layer, coords, is_correct, is_incorrect = panel
        apply_layer_axes_style(ax, highlight)
        _plot_pca_approach_b_axis(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            normalize_vector=normalize_vector,
            sample_ids=sample_ids if use_implementation_markers else None,
            add_legend=add_legend,
        )
        set_layer_subplot_title(ax, layer, highlight)

    return _render


def _make_render_pca_a_fold_panel():
    def _render(
        ax,
        panel: Any,
        highlight: LayerHighlight,
        *,
        add_legend: bool = True,
    ) -> None:
        fold, layer, coords, is_correct, is_incorrect, arrow, captured = panel
        apply_layer_axes_style(ax, highlight)
        _plot_pca_approach_a_axis(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            arrow=arrow,
            add_legend=add_legend,
        )
        title = f"fold={fold}, layer={layer}\ncaptured={captured:.3f}"
        ax.set_title(title, **highlighted_subplot_title_kwargs(highlight))

    return _render


def _make_render_pca_b_fold_panel(normalize_vector: bool):
    def _render(
        ax,
        panel: Any,
        highlight: LayerHighlight,
        *,
        add_legend: bool = True,
    ) -> None:
        fold, layer, coords, is_correct, is_incorrect = panel
        apply_layer_axes_style(ax, highlight)
        _plot_pca_approach_b_axis(
            ax=ax,
            coords=coords,
            is_correct=is_correct,
            is_incorrect=is_incorrect,
            normalize_vector=normalize_vector,
            add_legend=add_legend,
        )
        ax.set_title(
            f"fold={fold}, layer={layer}",
            **highlighted_subplot_title_kwargs(highlight),
        )

    return _render


def run_dynamic(
    *,
    hidden_states_root: Path,
    results_root: Path,
    vectors_root: Path,
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
    output_dir: Path,
    folds_is_all: bool,
    task_mode: bool,
    explicit_task_ids: List[str] | None,
    approaches: Sequence[str],
    normalize_vector: bool,
    implementation_markers: bool,
    projection_source: str,
) -> Dict[str, object]:
    use_saved_projections = projection_source == "saved"
    fold_results_by_fold: Dict[int, dict] = {}
    fold_detailed_by_fold: Dict[int, dict] = {}
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
        if use_saved_projections:
            fold_detailed_by_fold[fold] = load_fold_detailed(
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
        selected_layer = int(fold_results["selected_layer"])
        bundle_blob = load_test_bundle(
            hidden_states_root=hidden_states_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            fold=fold,
            test_framing=test_framing,
        )
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
            "direction_vectors": direction_vectors,
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

    pca_cache: Dict[Tuple[int, int, str, str, str], Any] = {}
    task_index_by_fold: Dict[int, Dict[str, List[int]]] = {
        fold: {} for fold in folds
    }
    for fold in folds:
        for i, tid in enumerate(fold_data[fold]["task_ids"]):
            task_index_by_fold[fold].setdefault(tid, []).append(i)

    def _result_for(
        fold: int,
        layer: int,
        approach: str,
        task_id: str | None = None,
    ) -> Any:
        key = (
            fold,
            layer,
            task_id if task_id is not None else "__all__",
            approach,
            projection_source,
        )
        if key in pca_cache:
            return pca_cache[key]

        acts = fold_data[fold]["acts"]
        if task_id is None:
            features = acts[:, layer, :]
            plot_sample_ids = fold_data[fold]["sample_ids"]
        else:
            idx = task_index_by_fold[fold].get(task_id, [])
            if len(idx) != 4:
                raise ValueError(
                    f"fold {fold} task {task_id}: expected 4 candidates, got {len(idx)}"
                )
            features = acts[idx, layer, :]
            plot_sample_ids = [fold_data[fold]["sample_ids"][i] for i in idx]

        reading = _reading_vector(fold_data[fold]["direction_vectors"], layer)
        if approach == "A":
            coords, arrow, captured = _compute_pca_approach_a(
                features,
                reading,
                normalize_vector=normalize_vector,
                seed=seed,
            )
            pca_cache[key] = (coords, arrow, captured)
        else:
            x_scores = None
            if use_saved_projections:
                x_scores = test_projection_scores(
                    fold_detailed_by_fold[fold],
                    layer=layer,
                    sample_ids=plot_sample_ids,
                )
            coords = _compute_pca_approach_b(
                features,
                reading,
                normalize_vector=normalize_vector,
                seed=seed,
                x_scores=x_scores,
            )
            pca_cache[key] = coords
        return pca_cache[key]

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

    def _emit_outputs_for_approach(approach: str) -> None:
        approach_suffix = approach_filename_suffix(approach, normalize_vector)
        render_layer = (
            _make_render_pca_a_panel(normalize_vector, implementation_markers)
            if approach == "A"
            else _make_render_pca_b_panel(normalize_vector, implementation_markers)
        )
        render_fold = (
            _make_render_pca_a_fold_panel()
            if approach == "A"
            else _make_render_pca_b_fold_panel(normalize_vector)
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

                    sample_ids_task = [fold_data[fold]["sample_ids"][i] for i in idx]
                    panels = []
                    for layer in layers:
                        if approach == "A":
                            coords, arrow, captured = _result_for(
                                fold, layer, approach, task_id
                            )
                            panels.append(
                                (
                                    layer,
                                    coords,
                                    is_correct_task,
                                    is_incorrect_task,
                                    arrow,
                                    captured,
                                    sample_ids_task,
                                )
                            )
                        else:
                            coords = _result_for(fold, layer, approach, task_id)
                            panels.append(
                                (
                                    layer,
                                    coords,
                                    is_correct_task,
                                    is_incorrect_task,
                                    sample_ids_task,
                                )
                            )

                    layer_tag_str = layer_tag(layer_mode, layers)
                    task_safe = safe_task_id(task_id)
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
                    task_filename_suffix = approach_norm_filename_suffix(normalize_vector)
                    out_path = (
                        out_root
                        / f"fold_{fold}"
                        / "tasks"
                        / outcome
                        / pca_approach_dirname(approach)
                        / f"fold_{fold}_task_{task_safe}_layer_{layer_tag_str}{task_filename_suffix}.png"
                    )
                    build_layer_grid(
                        panels=panels,
                        fold_results=fold_data[fold]["fold_results"],
                        fig_title=f"{dataset_label} {benchmark} | fold={fold} | task={task_id}",
                        out_path=out_path,
                        render_panel=render_layer,
                    )
                    outputs.append(str(out_path))
        elif len(folds) == 1:
            fold = folds[0]
            layers = resolved_layers_per_fold[fold]
            if len(layers) == 1:
                layer = layers[0]
                highlight = layer_highlight_state(fold_data[fold]["fold_results"], layer)
                out_path = (
                    out_root
                    / f"fold_{fold}"
                    / f"fold_{fold}_layer_{layer}{approach_suffix}.png"
                )
                if approach == "A":
                    coords, arrow, captured = _result_for(fold, layer, approach)
                    build_single_scatter(
                        plot_axes=lambda ax: _plot_pca_approach_a_axis(
                            ax=ax,
                            coords=coords,
                            is_correct=fold_data[fold]["is_correct"],
                            is_incorrect=fold_data[fold]["is_incorrect"],
                            arrow=arrow,
                        ),
                        title=f"{dataset_label} {benchmark} | fold={fold}, layer={layer}",
                        out_path=out_path,
                        highlight=highlight,
                    )
                else:
                    coords = _result_for(fold, layer, approach)
                    build_single_scatter(
                        plot_axes=lambda ax, c=coords: _plot_pca_approach_b_axis(
                            ax=ax,
                            coords=c,
                            is_correct=fold_data[fold]["is_correct"],
                            is_incorrect=fold_data[fold]["is_incorrect"],
                            normalize_vector=normalize_vector,
                        ),
                        title=f"{dataset_label} {benchmark} | fold={fold}, layer={layer}",
                        out_path=out_path,
                        highlight=highlight,
                    )
                outputs.append(str(out_path))
            else:
                panels = []
                for layer in layers:
                    if approach == "A":
                        coords, arrow, captured = _result_for(fold, layer, approach)
                        panels.append(
                            (
                                layer,
                                coords,
                                fold_data[fold]["is_correct"],
                                fold_data[fold]["is_incorrect"],
                                arrow,
                                captured,
                            )
                        )
                    else:
                        coords = _result_for(fold, layer, approach)
                        panels.append(
                            (
                                layer,
                                coords,
                                fold_data[fold]["is_correct"],
                                fold_data[fold]["is_incorrect"],
                            )
                        )
                layer_tag_str = layer_tag(layer_mode, layers)
                out_path = (
                    out_root
                    / f"fold_{fold}"
                    / f"fold_{fold}_layer_{layer_tag_str}{approach_suffix}.png"
                )
                build_layer_grid(
                    panels=panels,
                    fold_results=fold_data[fold]["fold_results"],
                    fig_title=f"{dataset_label} {benchmark} | fold={fold}",
                    out_path=out_path,
                    render_panel=render_layer,
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
                    if approach == "A":
                        coords, arrow, captured = _result_for(fold, layer, approach)
                        panels.append(
                            (
                                fold,
                                layer,
                                coords,
                                fold_data[fold]["is_correct"],
                                fold_data[fold]["is_incorrect"],
                                arrow,
                                captured,
                            )
                        )
                    else:
                        coords = _result_for(fold, layer, approach)
                        panels.append(
                            (
                                fold,
                                layer,
                                coords,
                                fold_data[fold]["is_correct"],
                                fold_data[fold]["is_incorrect"],
                            )
                        )
                layer_tag_str = layer_tag(layer_mode, resolved_layers_per_fold[folds[0]])
                out_path = (
                    out_root
                    / container
                    / f"{container}_layer_{layer_tag_str}{approach_suffix}.png"
                )
                build_fold_grid(
                    panels=panels,
                    fold_results_by_fold=fold_results_by_fold,
                    fig_title=f"{dataset_label} {benchmark} | folds={folds}",
                    out_path=out_path,
                    render_panel=render_fold,
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
                        if approach == "A":
                            coords, arrow, captured = _result_for(fold, layer, approach)
                            panels.append(
                                (
                                    layer,
                                    coords,
                                    fold_data[fold]["is_correct"],
                                    fold_data[fold]["is_incorrect"],
                                    arrow,
                                    captured,
                                )
                            )
                        else:
                            coords = _result_for(fold, layer, approach)
                            panels.append(
                                (
                                    layer,
                                    coords,
                                    fold_data[fold]["is_correct"],
                                    fold_data[fold]["is_incorrect"],
                                )
                            )
                    layer_tag_str = layer_tag(layer_mode, layers)
                    out_path = (
                        out_root
                        / container
                        / f"fold_{fold}_layer_{layer_tag_str}{approach_suffix}.png"
                    )
                    build_layer_grid(
                        panels=panels,
                        fold_results=fold_data[fold]["fold_results"],
                        fig_title=f"{dataset_label} {benchmark} | fold={fold}",
                        out_path=out_path,
                        render_panel=render_layer,
                    )
                    outputs.append(str(out_path))

    for approach in approaches:
        _emit_outputs_for_approach(approach)

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
        "approach": list(approaches),
        "normalize_vector": normalize_vector,
        "implementation_markers": implementation_markers,
        "projection_source": projection_source,
        "vectors_root": str(vectors_root),
        "seed": seed,
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
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Enable per-task mode: use 'all' or explicit task_id list.")
    parser.add_argument(
        "--approach",
        choices=["A", "B", "both"],
        default="both",
        help="PCA visualization approach: A=standard PCA with reading-vector arrow; "
             "B=reading-vector-spanned plane.",
    )
    parser.add_argument(
        "--normalize-vector",
        action="store_true",
        help="Unit-normalize the reading vector for projection scaling.",
    )
    parser.add_argument(
        "--implementation-markers",
        action="store_true",
        help="In --tasks mode, use distinct markers per implementation id.",
    )
    parser.add_argument(
        "--projection-source",
        choices=["recompute", "saved"],
        default="recompute",
        help="Approach B x-axis: recompute activations @ reading vector, or load "
             "per-sample scores from results/.../detailed.json (approach A always "
             "recomputes 2D PCA on activations).",
    )
    parser.add_argument("--output-dir", default="visualizations/pca")
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
    approaches = parse_approach_selector(args.approach)
    if args.implementation_markers and not task_mode:
        raise ValueError("--implementation-markers requires --tasks")

    summary = run_dynamic(
        hidden_states_root=Path(args.hidden_states_root),
        results_root=Path(args.results_root),
        vectors_root=Path(args.vectors_root),
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
        output_dir=Path(args.output_dir),
        folds_is_all=folds_is_all,
        task_mode=task_mode,
        explicit_task_ids=explicit_task_ids,
        approaches=approaches,
        normalize_vector=args.normalize_vector,
        implementation_markers=args.implementation_markers,
        projection_source=parse_projection_source(args.projection_source),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
