"""Shared helpers for cache-only hidden-state visualization scripts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.hidden_states.cache import bundle_output_dir, safe_model_dir, vector_common_path
from src.vectors.direction import Direction

DEFAULT_ICSE26_DATA_ROOT = Path("dataset/icse26")
DEFAULT_MUTATION_PHASE2_ROOT = Path("dataset/bigcodebench/phase2")

RenderPanelFn = Callable[[Any, Any, int], None]
RenderFoldPanelFn = Callable[[Any, Any], None]


def splits_dir(
    *,
    split_family: str,
    benchmark: str,
    data_root: Path,
) -> Path:
    """Directory containing ``manifest.json`` and fold_* subdirs for a split family."""
    if split_family == "icse26":
        return data_root / benchmark

    if (data_root / "manifest.json").is_file():
        return data_root
    nested = data_root / split_family
    if (nested / "manifest.json").is_file():
        return nested
    default = DEFAULT_MUTATION_PHASE2_ROOT / split_family
    if (default / "manifest.json").is_file():
        return default
    raise FileNotFoundError(
        f"could not locate splits for {split_family!r}; "
        f"tried {data_root}, {nested}, and {default}"
    )


def read_selected_layer(
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


def resolve_all_folds(
    *,
    split_family: str,
    data_root: Path,
    benchmark: str,
) -> List[int]:
    manifest_path = splits_dir(
        split_family=split_family, benchmark=benchmark, data_root=data_root
    ) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest for --folds all: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    num_folds = int(manifest["num_folds"])
    return list(range(num_folds))


def parse_fold_selector(
    folds_arg: Sequence[str],
    *,
    split_family: str,
    data_root: Path,
    benchmark: str,
) -> List[int]:
    if len(folds_arg) == 1 and folds_arg[0] == "all":
        return resolve_all_folds(
            split_family=split_family, data_root=data_root, benchmark=benchmark
        )
    try:
        parsed = sorted({int(x) for x in folds_arg})
    except ValueError as exc:
        raise ValueError("--folds must be integers or 'all'") from exc
    if not parsed:
        raise ValueError("--folds resolved to an empty set")
    if any(f < 0 for f in parsed):
        raise ValueError("--folds must be non-negative")
    return parsed


def parse_layer_selector(layers_arg: Sequence[str]) -> Tuple[str, List[int] | None]:
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


def parse_task_selector(tasks_arg: Sequence[str] | None) -> Tuple[bool, List[str] | None]:
    if not tasks_arg:
        return False, None
    if len(tasks_arg) == 1 and tasks_arg[0] == "all":
        return True, None
    task_ids = sorted({x for x in tasks_arg if x})
    if not task_ids:
        raise ValueError("--tasks resolved to an empty set")
    return True, task_ids


def safe_task_id(task_id: str) -> str:
    leaf = task_id.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", leaf)


def viz_output_path_parts(split_family: str) -> Tuple[str, ...]:
    """Path segments between ``<model_safe>/`` and ``<benchmark>/`` in output dirs."""
    if split_family == "icse26":
        return ("icse26",)
    if split_family.startswith("ood_"):
        return ("ood", split_family)
    return ("mutation_ds", split_family)


def viz_dataset_label(split_family: str) -> str:
    if split_family == "icse26":
        return "ICSE26"
    if split_family.startswith("ood_"):
        return f"OOD ({split_family})"
    return f"Mutation_DS ({split_family})"


def dataset_results_path(
    *,
    split_family: str,
    benchmark: str,
    data_root: Path,
) -> Path:
    """Per-task monitoring outcomes JSONL (ICSE26 or Mutation_DS phase2 layout)."""
    splits = splits_dir(
        split_family=split_family, benchmark=benchmark, data_root=data_root
    )
    if split_family == "icse26":
        filename = f"icse26_{benchmark}.jsonl"
    else:
        filename = f"{split_family}_{benchmark}.jsonl"
    return splits / filename


def load_task_correctness_index(
    *,
    split_family: str,
    data_root: Path,
    benchmark: str,
    model: str,
    method: str,
    fit_framing: str,
    representation: str,
    val_framing: str,
    test_framing: str,
) -> Dict[Tuple[int, str], bool]:
    results_path = dataset_results_path(
        split_family=split_family, benchmark=benchmark, data_root=data_root
    )
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


def load_test_bundle(
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
    return torch.load(bundle_path, map_location="cpu")


def load_direction_vectors(
    *,
    vectors_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    method: str,
    fit_framing: str,
    representation: str,
    fold: int,
    distribution: str = "id",
) -> Dict[int, np.ndarray]:
    """Load per-layer reading vectors (incorrect -> correct) from ``direction.pt``."""
    common = vector_common_path(
        model,
        split_family,
        benchmark,
        distribution,
        method,
        fit_framing,
        representation,
    )
    direction_path = vectors_root / common / f"fold_{fold}" / "direction.pt"
    if not direction_path.is_file():
        raise FileNotFoundError(f"missing direction vector file: {direction_path}")

    direction = Direction.load(direction_path)
    if direction.representation and direction.representation != representation:
        raise ValueError(
            f"direction representation mismatch in {direction_path}: "
            f"expected {representation!r}, got {direction.representation!r}"
        )

    vectors: Dict[int, np.ndarray] = {}
    for layer, vec in direction.vectors.items():
        signed = direction.signs[layer] * vec.detach().cpu().numpy()
        vectors[int(layer)] = signed.astype(np.float64)
    return vectors


def classify_labels_strict(
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


def scatter_correctness(
    *,
    ax,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
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


def grid_shape(n: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def save_figure(fig, out_path: Path) -> None:
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_single_scatter(
    *,
    plot_axes: Callable[[Any], None],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_axes(ax)
    ax.set_title(title)
    save_figure(fig, out_path)


def build_layer_grid(
    *,
    panels: Sequence[Any],
    selected_layer: int,
    fig_title: str,
    out_path: Path,
    render_panel: RenderPanelFn,
) -> None:
    rows, cols = grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_1d = np.array(axes).reshape(-1)
    for i, panel in enumerate(panels):
        render_panel(axes_1d[i], panel, selected_layer)
    for j in range(len(panels), len(axes_1d)):
        axes_1d[j].axis("off")
    fig.suptitle(fig_title, fontsize=14)
    save_figure(fig, out_path)


def build_fold_grid(
    *,
    panels: Sequence[Any],
    fig_title: str,
    out_path: Path,
    render_panel: RenderFoldPanelFn,
) -> None:
    rows, cols = grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_1d = np.array(axes).reshape(-1)
    for i, panel in enumerate(panels):
        render_panel(axes_1d[i], panel)
    for j in range(len(panels), len(axes_1d)):
        axes_1d[j].axis("off")
    fig.suptitle(fig_title, fontsize=14)
    save_figure(fig, out_path)


def resolve_layers_for_fold(
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


def layer_tag(layer_mode: str, layers: List[int]) -> str:
    if layer_mode == "selected":
        return str(layers[0])
    if layer_mode == "all":
        return "all"
    return "_".join(str(x) for x in layers)


def cross_fold_dirname(
    *,
    folds: Sequence[int],
    folds_is_all: bool,
    layer_mode: str,
) -> str:
    if layer_mode == "explicit":
        return "custom"
    if folds_is_all:
        return "fold_all"
    if len(folds) > 1:
        return "custom"
    return f"fold_{folds[0]}"


def approach_filename_suffix(approach: str, normalize_vector: bool) -> str:
    return f"_approach{approach}_norm{normalize_vector}"


def parse_approach_selector(approach_arg: str) -> List[str]:
    if approach_arg == "both":
        return ["A", "B"]
    if approach_arg in {"A", "B"}:
        return [approach_arg]
    raise ValueError("--approach must be A, B, or both")
