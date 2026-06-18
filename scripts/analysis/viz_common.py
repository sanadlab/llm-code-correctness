"""Shared helpers for cache-only hidden-state visualization scripts."""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from src.hidden_states.cache import bundle_output_dir, safe_model_dir, vector_common_path
from src.vectors.direction import Direction

DEFAULT_ICSE26_DATA_ROOT = Path("dataset/icse26")
DEFAULT_MUTATION_PHASE2_ROOT = Path("dataset/bigcodebench/phase2")

VAL_TOP_LAYER_ACC_FRACTION = 0.9

LAYER_SELECTED_BG = "#e8f5e9"
LAYER_TOP_VAL_BG = "#fff9c4"
LAYER_TEST_BEST_TITLE_BG = "#a5d6a7"

RenderPanelFn = Callable[[Any, Any, "LayerHighlight"], None]
RenderFoldPanelFn = Callable[[Any, Any, "LayerHighlight"], None]


@dataclass(frozen=True)
class LayerHighlight:
    selected_bg: bool
    top_val_bg: bool
    test_best_title: bool


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


def fold_results_path(
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
) -> Path:
    return (
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


def load_fold_results(
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
) -> dict:
    results_path = fold_results_path(
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
    if not results_path.is_file():
        raise FileNotFoundError(f"missing fold results file: {results_path}")
    with open(results_path) as f:
        payload = json.load(f)
    for key in ("selected_layer", "val_per_layer", "test_theoretical_best_layer"):
        if key not in payload:
            raise KeyError(f"`{key}` missing in {results_path}")
    if "layer" not in payload["test_theoretical_best_layer"]:
        raise KeyError(f"`test_theoretical_best_layer.layer` missing in {results_path}")
    return payload


def fold_detailed_path(
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
) -> Path:
    return fold_results_path(
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
    ).parent / "detailed.json"


def load_fold_detailed(
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
) -> dict:
    detailed_path = fold_detailed_path(
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
    if not detailed_path.is_file():
        raise FileNotFoundError(f"missing fold detailed file: {detailed_path}")
    with open(detailed_path) as f:
        payload = json.load(f)
    for key in ("layers", "test"):
        if key not in payload:
            raise KeyError(f"`{key}` missing in {detailed_path}")
    test = payload["test"]
    for key in ("sample_ids", "scores"):
        if key not in test:
            raise KeyError(f"`test.{key}` missing in {detailed_path}")
    return payload


def test_projection_scores(
    detailed: dict,
    *,
    layer: int,
    sample_ids: Sequence[str],
) -> np.ndarray:
    """Per-sample correctness-direction scores from ``detailed.json`` test split."""
    layers = [int(x) for x in detailed["layers"]]
    if layer not in layers:
        raise KeyError(f"layer {layer} not in detailed.json layers {layers}")
    col = layers.index(layer)
    test = detailed["test"]
    id_to_score = {
        sid: float(test["scores"][i][col]) for i, sid in enumerate(test["sample_ids"])
    }
    missing = [sid for sid in sample_ids if sid not in id_to_score]
    if missing:
        raise KeyError(
            f"{len(missing)} sample_id(s) missing from detailed.json test scores; "
            f"first missing: {missing[:5]}"
        )
    return np.array([id_to_score[sid] for sid in sample_ids], dtype=np.float64)


def parse_projection_source(projection_source_arg: str) -> str:
    if projection_source_arg in {"recompute", "saved"}:
        return projection_source_arg
    raise ValueError("--projection-source must be 'recompute' or 'saved'")


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
    return int(
        load_fold_results(
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
        )["selected_layer"]
    )


def top_validation_layers(results: dict) -> set[int]:
    """Layers whose validation pairwise_acc is within the top fraction of the best layer."""
    val_per_layer = results["val_per_layer"]
    accs = {int(layer): float(val_per_layer[layer]["pairwise_acc"]) for layer in val_per_layer}
    best_val = max(accs.values())
    threshold = VAL_TOP_LAYER_ACC_FRACTION * best_val
    return {layer for layer, acc in accs.items() if acc >= threshold}


def relevant_layers_for_fold(results: dict, *, num_layers: int) -> List[int]:
    """Union of top-val, validation-selected, and test-best layers (deduped, sorted)."""
    layers = set(top_validation_layers(results))
    layers.add(int(results["selected_layer"]))
    layers.add(int(results["test_theoretical_best_layer"]["layer"]))
    bad = [layer for layer in layers if layer >= num_layers]
    if bad:
        raise ValueError(
            f"relevant layer(s) {bad} not present; bundle has layers [0, {num_layers - 1}]"
        )
    return sorted(layers)


def layer_highlight_state(results: dict, layer: int) -> LayerHighlight:
    selected = int(results["selected_layer"])
    test_best = int(results["test_theoretical_best_layer"]["layer"])
    in_top = layer in top_validation_layers(results)
    return LayerHighlight(
        selected_bg=(layer == selected),
        top_val_bg=in_top and (layer != selected),
        test_best_title=(layer == test_best),
    )


def apply_layer_axes_style(ax, highlight: LayerHighlight) -> None:
    if highlight.selected_bg:
        ax.set_facecolor(LAYER_SELECTED_BG)
    elif highlight.top_val_bg:
        ax.set_facecolor(LAYER_TOP_VAL_BG)


def format_layer_subplot_title(layer: int) -> str:
    return f"layer={layer}"


def highlighted_subplot_title_kwargs(highlight: LayerHighlight) -> Dict[str, object]:
    """Green background highlight behind subplot title text (not green text)."""
    if not highlight.test_best_title:
        return {}
    return {
        "bbox": {
            "facecolor": LAYER_TEST_BEST_TITLE_BG,
            "edgecolor": "none",
            "pad": 3.0,
            "boxstyle": "round,pad=0.3",
        }
    }


def set_layer_subplot_title(ax, layer: int, highlight: LayerHighlight, *, extra: str = "") -> None:
    title = format_layer_subplot_title(layer)
    if extra:
        title = f"{title}\n{extra}"
    ax.set_title(title, **highlighted_subplot_title_kwargs(highlight))


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


def is_ood_split_family(split_family: str) -> bool:
    return split_family.startswith("ood_")


def ood_source_from_split_family(split_family: str) -> str:
    if not is_ood_split_family(split_family):
        raise ValueError(f"not an OOD split family: {split_family!r}")
    return split_family.removeprefix("ood_")


def resolve_ood_outer_folds(*, data_root: Path, benchmark: str) -> List[int]:
    manifest_path = data_root / benchmark / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing OOD outer-fold manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return list(range(int(manifest["num_folds"])))


def resolve_ood_inner_folds(*, data_root: Path, ood_source: str) -> List[int]:
    manifest_path = data_root / ood_source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing OOD inner-fold manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return list(range(int(manifest["num_folds"])))


def parse_ood_fold_selector(
    folds_arg: Sequence[str],
    *,
    data_root: Path,
    benchmark: str | None = None,
    ood_source: str | None = None,
    fold_kind: str,
) -> List[int]:
    if fold_kind not in {"outer", "inner"}:
        raise ValueError(f"fold_kind must be 'outer' or 'inner', got {fold_kind!r}")
    if len(folds_arg) == 1 and folds_arg[0] == "all":
        if fold_kind == "outer":
            if benchmark is None:
                raise ValueError("benchmark required to resolve outer folds")
            return resolve_ood_outer_folds(data_root=data_root, benchmark=benchmark)
        if ood_source is None:
            raise ValueError("ood_source required to resolve inner folds")
        return resolve_ood_inner_folds(data_root=data_root, ood_source=ood_source)
    try:
        parsed = sorted({int(x) for x in folds_arg})
    except ValueError as exc:
        raise ValueError(f"--{fold_kind}-folds must be integers or 'all'") from exc
    if not parsed:
        raise ValueError(f"--{fold_kind}-folds resolved to an empty set")
    if any(f < 0 for f in parsed):
        raise ValueError(f"--{fold_kind}-folds must be non-negative")
    return parsed


def parse_layer_selector(layers_arg: Sequence[str]) -> Tuple[str, List[int] | None]:
    if len(layers_arg) == 1 and layers_arg[0] in {"selected", "all", "relevant"}:
        return layers_arg[0], None
    try:
        parsed = sorted({int(x) for x in layers_arg})
    except ValueError as exc:
        raise ValueError(
            "--layers must be integers, 'selected', 'relevant', or 'all'"
        ) from exc
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
    fold_prefix: str = "fold",
) -> Dict[str, object]:
    return load_cached_bundle(
        hidden_states_root=hidden_states_root,
        model=model,
        split_family=split_family,
        benchmark=benchmark,
        fold=fold,
        data_split="test",
        framing=test_framing,
        fold_prefix=fold_prefix,
    )


def load_validate_bundle(
    *,
    hidden_states_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    fold: int,
    val_framing: str,
    fold_prefix: str = "fold",
) -> Dict[str, object]:
    return load_cached_bundle(
        hidden_states_root=hidden_states_root,
        model=model,
        split_family=split_family,
        benchmark=benchmark,
        fold=fold,
        data_split="validate",
        framing=val_framing,
        fold_prefix=fold_prefix,
    )


def load_cached_bundle(
    *,
    hidden_states_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    fold: int,
    data_split: str,
    framing: str,
    fold_prefix: str = "fold",
) -> Dict[str, object]:
    out_dir = bundle_output_dir(
        output_root=hidden_states_root,
        model_name=model,
        split_family=split_family,
        benchmark=benchmark,
        distribution="id",
        data_split=data_split,
        framing=framing,
        fold=fold,
        fold_prefix=fold_prefix,
    )
    bundle_path = out_dir / "bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"missing cached {data_split} bundle: {bundle_path}"
        )
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
    fold_prefix: str = "fold",
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
    direction_path = vectors_root / common / f"{fold_prefix}_{fold}" / "direction.pt"
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


TASK_IMPLEMENTATION_MARKERS = ("o", "s", "^", "D", "v", "p", "X", "P")

CORRECT_FACE = "skyblue"
INCORRECT_FACE = "salmon"
SCATTER_EDGE_WIDTH = 0.6
SCATTER_POINT_SIZE = 20


@dataclass(frozen=True)
class _LayoutPolicy:
    """Typography and margin knobs. Sizes interpolate between min/max via layout scale."""

    # Reference canvas: 4×3 in at 9 panels → scale 1.0
    ref_area_in2: float = 12.0
    ref_panel_count: int = 9

    suptitle_font_pt: Tuple[float, float] = (9.0, 20.0)
    legend_font_pt: Tuple[float, float] = (8.0, 13.0)
    subplot_title_fraction: float = 0.72

    chars_per_inch_wrap: float = 14.0
    min_title_wrap_chars: int = 20
    pt_per_inch: float = 72.0
    title_line_leading: float = 1.3
    legend_row_leading: float = 1.5
    margin_padding_in: float = 0.10

    bottom_margin_min: float = 0.08
    bottom_margin_max: float = 0.24
    top_margin_min: float = 0.86
    top_margin_max: float = 0.96
    top_margin_base: float = 0.88
    tight_layout_top_inset: float = 0.02


_LAYOUT = _LayoutPolicy()


@dataclass(frozen=True)
class FigureLayout:
    suptitle_fontsize: float
    legend_fontsize: float
    subplot_title_fontsize: float
    top_pad: float
    bottom_pad: float
    legend_ncol: int


@dataclass(frozen=True)
class FigureGeometry:
    """Canvas size + content counts; all layout sizes derive from layout_scale."""

    fig_width_in: float
    fig_height_in: float
    n_panels: int
    ncols: int
    title_line_count: int = 1
    n_legend_entries: int = 0

    @classmethod
    def from_grid(cls, *, n_panels: int, ncols: int, fig_width_in: float, fig_height_in: float) -> FigureGeometry:
        return cls(
            fig_width_in=fig_width_in,
            fig_height_in=fig_height_in,
            n_panels=n_panels,
            ncols=ncols,
        )

    def with_content(self, *, title_line_count: int, n_legend_entries: int) -> FigureGeometry:
        return replace(
            self,
            title_line_count=title_line_count,
            n_legend_entries=n_legend_entries,
        )

    def layout_scale(self, policy: _LayoutPolicy = _LAYOUT) -> float:
        area_ratio = math.sqrt(self.fig_width_in * self.fig_height_in / policy.ref_area_in2)
        panel_ratio = math.log2(max(self.n_panels, 1)) / math.log2(policy.ref_panel_count)
        return _clamp(0.4 * area_ratio + 0.6 * panel_ratio, 0.0, 1.0)

    def wrap_title(self, title: str, policy: _LayoutPolicy = _LAYOUT) -> Tuple[str, int]:
        wrap_chars = max(
            policy.min_title_wrap_chars,
            int(policy.chars_per_inch_wrap * self.fig_width_in / max(self.ncols, 1)),
        )
        wrapped = "\n".join(textwrap.wrap(title, width=wrap_chars))
        return wrapped, wrapped.count("\n") + 1

    def to_layout(self, policy: _LayoutPolicy = _LAYOUT) -> FigureLayout:
        scale = self.layout_scale(policy)
        suptitle_fontsize = _lerp(scale, *policy.suptitle_font_pt)
        suptitle_fontsize = min(
            suptitle_fontsize,
            max(policy.suptitle_font_pt[0], self.fig_height_in * 2.4),
        )
        legend_fontsize = _lerp(scale, *policy.legend_font_pt)
        subplot_title_fontsize = _clamp(
            suptitle_fontsize * policy.subplot_title_fraction,
            8.5,
            12.0,
        )
        legend_ncol = self._legend_ncol(policy)
        return FigureLayout(
            suptitle_fontsize=suptitle_fontsize,
            legend_fontsize=legend_fontsize,
            subplot_title_fontsize=subplot_title_fontsize,
            top_pad=self._top_margin(suptitle_fontsize, policy),
            bottom_pad=self._bottom_margin(legend_fontsize, legend_ncol, policy),
            legend_ncol=legend_ncol,
        )

    def _legend_ncol(self, policy: _LayoutPolicy) -> int:
        n = self.n_legend_entries
        if n <= 0:
            return 1
        if n <= 3:
            return n
        return min(n, max(2, min(4, self.ncols)))

    def _text_height_in(self, lines: int, fontsize_pt: float, leading: float, policy: _LayoutPolicy) -> float:
        return lines * (fontsize_pt / policy.pt_per_inch) * leading + policy.margin_padding_in

    def _top_margin(self, suptitle_fontsize: float, policy: _LayoutPolicy) -> float:
        title_height_in = self._text_height_in(
            self.title_line_count, suptitle_fontsize, policy.title_line_leading, policy
        )
        return _clamp(
            policy.top_margin_base + title_height_in / self.fig_height_in * 0.5,
            policy.top_margin_min,
            policy.top_margin_max,
        )

    def _bottom_margin(self, legend_fontsize: float, legend_ncol: int, policy: _LayoutPolicy) -> float:
        if self.n_legend_entries <= 0:
            return policy.bottom_margin_min
        legend_rows = math.ceil(self.n_legend_entries / legend_ncol)
        legend_height_in = self._text_height_in(
            legend_rows, legend_fontsize, policy.legend_row_leading, policy
        )
        return _clamp(
            legend_height_in / self.fig_height_in,
            policy.bottom_margin_min,
            policy.bottom_margin_max,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(t: float, low: float, high: float) -> float:
    return low + t * (high - low)


def darker_edge(facecolor: str, *, factor: float = 0.55) -> str:
    """Return a darker edge color for a scatter face color."""
    r, g, b = mcolors.to_rgb(facecolor)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _scatter_face_edge(facecolor: str) -> Tuple[str, str]:
    return facecolor, darker_edge(facecolor)


def _legend_marker_handle(
    *,
    label: str,
    facecolor: str,
    marker: str = "o",
    alpha: float = 1.0,
    markersize: float = 8,
) -> Line2D:
    edge = darker_edge(facecolor) if alpha >= 1.0 else "gray"
    return Line2D(
        [0],
        [0],
        marker=marker,
        linestyle="None",
        markerfacecolor=facecolor,
        markeredgecolor=edge,
        markeredgewidth=SCATTER_EDGE_WIDTH,
        alpha=alpha,
        markersize=markersize,
        label=label,
    )


def implementation_marker_map(impl_ids: Sequence[str]) -> Dict[str, str]:
    """Deterministic marker assignment from stable implementation-id ordering."""
    unique = sorted(set(impl_ids))
    return {
        impl_id: TASK_IMPLEMENTATION_MARKERS[i % len(TASK_IMPLEMENTATION_MARKERS)]
        for i, impl_id in enumerate(unique)
    }


def correctness_legend_handles() -> List[Line2D]:
    return [
        _legend_marker_handle(label="correct (label=1)", facecolor=CORRECT_FACE),
        _legend_marker_handle(label="incorrect (label=0)", facecolor=INCORRECT_FACE),
    ]


def implementation_marker_legend_handles(impl_ids: Sequence[str]) -> List[Line2D]:
    marker_by_impl = implementation_marker_map(impl_ids)
    return [
        _legend_marker_handle(
            label=impl_id,
            facecolor="#d9d9d9",
            marker=marker_by_impl[impl_id],
        )
        for impl_id in sorted(marker_by_impl)
    ]


def task_implementation_legend_handles(
    sample_ids: Sequence[str],
    *,
    has_background: bool = False,
) -> List[Line2D]:
    handles: List[Line2D] = []
    if has_background:
        handles.append(
            _legend_marker_handle(
                label="fold background",
                facecolor="lightgray",
                alpha=0.35,
                markersize=6,
            )
        )
    handles.extend(correctness_legend_handles())
    handles.extend(implementation_marker_legend_handles(sample_ids))
    return handles


def dedupe_legend_handles(handles, labels):
    seen: Dict[str, object] = {}
    out_handles: List[object] = []
    out_labels: List[str] = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen[label] = handle
        out_handles.append(handle)
        out_labels.append(label)
    return out_handles, out_labels


def scatter_task_implementations(
    *,
    ax,
    coords: np.ndarray,
    sample_ids: Sequence[str],
    is_correct: np.ndarray,
    background_coords: np.ndarray | None = None,
    add_legend: bool = True,
    legend_fontsize: float | None = None,
) -> None:
    """Scatter per-task points with distinct markers per implementation id."""
    if coords.shape[0] != len(sample_ids):
        raise ValueError("coords/sample_ids length mismatch")
    if is_correct.shape[0] != len(sample_ids):
        raise ValueError("is_correct/sample_ids length mismatch")

    has_background = background_coords is not None and background_coords.shape[0] > 0
    if has_background:
        bg_face, bg_edge = _scatter_face_edge("lightgray")
        ax.scatter(
            background_coords[:, 0],
            background_coords[:, 1],
            s=6,
            alpha=0.2,
            facecolors=bg_face,
            edgecolors=bg_edge,
            linewidths=SCATTER_EDGE_WIDTH,
            zorder=1,
        )

    marker_by_impl = implementation_marker_map(sample_ids)
    for i, impl_id in enumerate(sample_ids):
        face = CORRECT_FACE if is_correct[i] else INCORRECT_FACE
        facecolor, edgecolor = _scatter_face_edge(face)
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            s=SCATTER_POINT_SIZE,
            alpha=1.0,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=SCATTER_EDGE_WIDTH,
            marker=marker_by_impl[impl_id],
            zorder=2,
        )

    legend_handles = task_implementation_legend_handles(
        sample_ids, has_background=has_background
    )
    ax._viz_legend_handles = legend_handles  # noqa: SLF001 — read by _attach_figure_legend
    if add_legend:
        ax.legend(
            handles=legend_handles,
            loc="best",
            fontsize=legend_fontsize if legend_fontsize is not None else _LAYOUT.legend_font_pt[1],
        )


def scatter_correctness(
    *,
    ax,
    coords: np.ndarray,
    is_correct: np.ndarray,
    is_incorrect: np.ndarray,
    background_coords: np.ndarray | None = None,
    add_legend: bool = True,
    legend_fontsize: float | None = None,
) -> None:
    has_background = background_coords is not None and background_coords.shape[0] > 0
    if has_background:
        bg_face, bg_edge = _scatter_face_edge("lightgray")
        ax.scatter(
            background_coords[:, 0],
            background_coords[:, 1],
            s=6,
            alpha=0.2,
            facecolors=bg_face,
            edgecolors=bg_edge,
            linewidths=SCATTER_EDGE_WIDTH,
            zorder=1,
        )
    inc_face, inc_edge = _scatter_face_edge(INCORRECT_FACE)
    ax.scatter(
        coords[is_incorrect, 0],
        coords[is_incorrect, 1],
        s=SCATTER_POINT_SIZE,
        alpha=1.0,
        facecolors=inc_face,
        edgecolors=inc_edge,
        linewidths=SCATTER_EDGE_WIDTH,
        zorder=2,
    )
    cor_face, cor_edge = _scatter_face_edge(CORRECT_FACE)
    ax.scatter(
        coords[is_correct, 0],
        coords[is_correct, 1],
        s=SCATTER_POINT_SIZE,
        alpha=1.0,
        facecolors=cor_face,
        edgecolors=cor_edge,
        linewidths=SCATTER_EDGE_WIDTH,
        zorder=2,
    )
    legend_handles = correctness_legend_handles()
    if has_background:
        legend_handles = [
            _legend_marker_handle(
                label="fold background",
                facecolor="lightgray",
                alpha=0.35,
                markersize=6,
            ),
            *legend_handles,
        ]
    ax._viz_legend_handles = legend_handles  # noqa: SLF001
    if add_legend:
        ax.legend(
            handles=legend_handles,
            loc="best",
            fontsize=legend_fontsize if legend_fontsize is not None else _LAYOUT.legend_font_pt[1],
        )


def grid_shape(n: int) -> Tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _finalize_figure(
    fig,
    *,
    title: str,
    geometry: FigureGeometry,
    legend_handles: Sequence[object],
    legend_labels: Sequence[str],
    subplot_axes: Sequence[Any],
) -> FigureLayout:
    wrapped_title, title_lines = geometry.wrap_title(title)
    geometry = geometry.with_content(
        title_line_count=title_lines,
        n_legend_entries=len(legend_labels),
    )
    layout = geometry.to_layout()
    fig.suptitle(wrapped_title, fontsize=layout.suptitle_fontsize, y=layout.top_pad)
    for ax in subplot_axes:
        if ax.title is not None:
            ax.title.set_fontsize(layout.subplot_title_fontsize)
    _attach_figure_legend(
        fig, handles=legend_handles, labels=legend_labels, layout=layout
    )
    return layout


def _panel_layer_from_data(panel: Any) -> int:
    return int(panel[0])


def _legend_handles_from_axis(ax) -> Tuple[List[object], List[str]]:
    stored = getattr(ax, "_viz_legend_handles", None)
    if stored:
        labels = [h.get_label() for h in stored]
        return list(stored), labels
    handles, labels = ax.get_legend_handles_labels()
    return dedupe_legend_handles(handles, labels)


def _attach_figure_legend(
    fig,
    *,
    handles: Sequence[object],
    labels: Sequence[str],
    layout: FigureLayout,
) -> None:
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, layout.bottom_pad / 2),
        ncol=layout.legend_ncol,
        fontsize=layout.legend_fontsize,
        frameon=True,
    )


def save_figure(
    fig,
    out_path: Path,
    *,
    top_pad: float | None = None,
    bottom_pad: float = 0.08,
) -> None:
    if top_pad is not None:
        inset = _LAYOUT.tight_layout_top_inset
        fig.tight_layout(rect=[0, bottom_pad, 1, top_pad - inset])
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_single_scatter(
    *,
    plot_axes: Callable[[Any], None],
    title: str,
    out_path: Path,
    highlight: LayerHighlight | None = None,
) -> None:
    fig_w, fig_h = 8.0, 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if highlight is not None:
        apply_layer_axes_style(ax, highlight)
    plot_axes(ax)
    axis_legend = ax.get_legend()
    if axis_legend is not None:
        axis_legend.remove()
    handles, labels = _legend_handles_from_axis(ax)
    geometry = FigureGeometry.from_grid(
        n_panels=1, ncols=1, fig_width_in=fig_w, fig_height_in=fig_h
    )
    layout = _finalize_figure(
        fig,
        title=title,
        geometry=geometry,
        legend_handles=handles,
        legend_labels=labels,
        subplot_axes=[],
    )
    save_figure(fig, out_path, top_pad=layout.top_pad, bottom_pad=layout.bottom_pad)


def build_layer_grid(
    *,
    panels: Sequence[Any],
    fold_results: dict,
    fig_title: str,
    out_path: Path,
    render_panel: RenderPanelFn,
) -> None:
    n_panels = len(panels)
    rows, cols = grid_shape(n_panels)
    fig_w, fig_h = 5 * cols, 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_1d = np.array(axes).reshape(-1)
    for i, panel in enumerate(panels):
        layer = _panel_layer_from_data(panel)
        highlight = layer_highlight_state(fold_results, layer)
        render_panel(axes_1d[i], panel, highlight, add_legend=False)
    for j in range(n_panels, len(axes_1d)):
        axes_1d[j].axis("off")
    handles, labels = _legend_handles_from_axis(axes_1d[0])
    geometry = FigureGeometry.from_grid(
        n_panels=n_panels, ncols=cols, fig_width_in=fig_w, fig_height_in=fig_h
    )
    layout = _finalize_figure(
        fig,
        title=fig_title,
        geometry=geometry,
        legend_handles=handles,
        legend_labels=labels,
        subplot_axes=list(axes_1d[:n_panels]),
    )
    save_figure(fig, out_path, top_pad=layout.top_pad, bottom_pad=layout.bottom_pad)


def build_fold_grid(
    *,
    panels: Sequence[Any],
    fold_results_by_fold: Dict[int, dict],
    fig_title: str,
    out_path: Path,
    render_panel: RenderFoldPanelFn,
) -> None:
    n_panels = len(panels)
    rows, cols = grid_shape(n_panels)
    fig_w, fig_h = 5 * cols, 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_1d = np.array(axes).reshape(-1)
    for i, panel in enumerate(panels):
        fold = int(panel[0])
        layer = int(panel[1])
        highlight = layer_highlight_state(fold_results_by_fold[fold], layer)
        render_panel(axes_1d[i], panel, highlight, add_legend=False)
    for j in range(n_panels, len(axes_1d)):
        axes_1d[j].axis("off")
    handles, labels = _legend_handles_from_axis(axes_1d[0])
    geometry = FigureGeometry.from_grid(
        n_panels=n_panels, ncols=cols, fig_width_in=fig_w, fig_height_in=fig_h
    )
    layout = _finalize_figure(
        fig,
        title=fig_title,
        geometry=geometry,
        legend_handles=handles,
        legend_labels=labels,
        subplot_axes=list(axes_1d[:n_panels]),
    )
    save_figure(fig, out_path, top_pad=layout.top_pad, bottom_pad=layout.bottom_pad)


def resolve_layers_for_fold(
    *,
    layer_mode: str,
    explicit_layers: List[int] | None,
    selected_layer: int,
    num_layers: int,
    fold_results: dict | None = None,
) -> List[int]:
    if layer_mode == "selected":
        return [selected_layer]
    if layer_mode == "relevant":
        if fold_results is None:
            raise ValueError("fold_results required for layer_mode='relevant'")
        return relevant_layers_for_fold(fold_results, num_layers=num_layers)
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
    if layer_mode == "relevant":
        return "relevant"
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


def pca_approach_dirname(approach: str) -> str:
    return f"approach{approach}"


def approach_norm_filename_suffix(normalize_vector: bool) -> str:
    return f"_norm{normalize_vector}"


def approach_filename_suffix(approach: str, normalize_vector: bool) -> str:
    return f"_approach{approach}_norm{normalize_vector}"


def parse_approach_selector(approach_arg: str) -> List[str]:
    if approach_arg == "both":
        return ["A", "B"]
    if approach_arg in {"A", "B"}:
        return [approach_arg]
    raise ValueError("--approach must be A, B, or both")
