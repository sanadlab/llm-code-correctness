"""Cache-only per-layer accuracy plot (ICSE26, Mutation_DS, OOD).

For each model, computes per-layer accuracy on cached hidden states using
per-fold reading vectors, macro-averages across folds, and plots layer index
vs. accuracy with one line per model.

Use ``--data-split test`` (default) for strict 1-of-4 test accuracy (ties wrong),
or ``--data-split validate`` for validation pairwise accuracy.

For OOD split families (``ood_mbpp``, ``ood_synthetic``), directions are fit on
the OOD source (inner folds) and evaluated on the ID test benchmark (outer
folds). Test implementation hidden states are loaded from the matching
``icse26`` caches (same benchmark test jsonl as the OOD experiment). Test
curves macro-average per-layer accuracy over all selected (outer, inner) cells;
validation curves macro-average over inner folds only.

Reads only precomputed artifacts:
  - hidden states from ``hidden_states/.../bundle.pt`` (test or validate split)
  - per-layer reading vectors from ``vectors/.../direction.pt``
  - fold count from ``manifest.json`` (via ``--folds all``)

Layer 0 (embedding output) is excluded from per-layer accuracy by default;
pass ``--include-embedding`` to include it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analysis.layer_quarter_accuracy import (
    layer_indices_for_bundle,
    task_layer_correct,
)
from scripts.analysis.viz_common import (
    is_ood_split_family,
    load_direction_vectors,
    load_test_bundle,
    load_validate_bundle,
    ood_source_from_split_family,
    parse_fold_selector,
    parse_ood_fold_selector,
    viz_output_path_parts,
)
from scripts.experiments._common import pairwise_accuracy
from src.hidden_states.cache import safe_model_dir

DATA_SPLIT_CHOICES = ("test", "validate")
OOD_TEST_BUNDLE_SPLIT_FAMILY = "icse26"

MODEL_PLOT_ORDER = (
    "mistralai/Mistral-7B-Instruct-v0.3",
    "codellama/CodeLlama-7b-Instruct-hf",
    "infly__OpenCoder-8B-Instruct",
    "Qwen__Qwen2.5-Coder-7B-Instruct",
)

MODEL_COLOR_MAP = {
    "mistralai/Mistral-7B-Instruct-v0.3": "C0",
    "codellama/CodeLlama-7b-Instruct-hf": "C1",
    "infly__OpenCoder-8B-Instruct": "C2",
    "Qwen__Qwen2.5-Coder-7B-Instruct": "C3",
}

logger = logging.getLogger(__name__)


def model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return (MODEL_PLOT_ORDER.index(model_name), model_name)
    except ValueError:
        return (len(MODEL_PLOT_ORDER), model_name)


def ordered_model_summaries(
    model_summaries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return sorted(model_summaries, key=lambda summary: model_sort_key(summary["model"]))


def per_fold_layer_accuracy(correct: np.ndarray) -> np.ndarray:
    """Per-layer strict accuracy for one fold.

    Parameters
    ----------
    correct:
        Boolean array ``[num_tasks, num_layers]`` indicating whether each
        (task, layer) pair strictly ranked the canonical implementation highest.
    """
    if correct.ndim != 2:
        raise ValueError(f"expected correct.ndim == 2, got shape={correct.shape}")
    if correct.size == 0:
        # No valid tasks – return NaNs so they do not influence means.
        num_layers = correct.shape[1]
        return np.full(num_layers, np.nan, dtype=np.float64)
    return correct.mean(axis=0).astype(np.float64)


def per_layer_pairwise_accuracy(
    acts: np.ndarray,
    labels: Sequence[int],
    task_ids: Sequence[str],
    direction_vectors: Dict[int, np.ndarray],
    layer_indices: Sequence[int],
) -> np.ndarray:
    """Per-layer validation pairwise accuracy for one fold."""
    layer_indices = list(layer_indices)
    accs = np.empty(len(layer_indices), dtype=np.float64)
    labels_t = torch.tensor(labels, dtype=torch.long)
    task_ids_list = list(task_ids)
    for col, layer in enumerate(layer_indices):
        if layer not in direction_vectors:
            raise KeyError(f"layer {layer} missing in direction vectors")
        vec = direction_vectors[layer]
        scores = torch.tensor(acts[:, layer] @ vec, dtype=torch.float64)
        accs[col] = pairwise_accuracy(scores, labels_t, task_ids_list)
    return accs


def metric_label(data_split: str) -> str:
    if data_split == "test":
        return "1-of-4 test accuracy"
    if data_split == "validate":
        return "validation pairwise accuracy"
    raise ValueError(f"unsupported data_split: {data_split!r}")


def output_stem(data_split: str) -> str:
    if data_split == "test":
        return "layer_accuracy"
    if data_split == "validate":
        return "layer_pairwise_accuracy"
    raise ValueError(f"unsupported data_split: {data_split!r}")


def aggregate_layers_across_cells(
    per_cell: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Macro-average per-layer accuracy across cells for one model.

    Each cell (ID fold or OOD outer×inner pair) receives equal weight.
    """
    if not per_cell:
        raise ValueError("aggregate_layers_across_cells: per_cell is empty")

    cell_ids = sorted(per_cell)
    first = per_cell[cell_ids[0]]
    num_layers = int(first.shape[0])
    stacked = np.stack([per_cell[c] for c in cell_ids], axis=0)  # [C, L]
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0, ddof=0)
    layers = list(range(num_layers))
    return mean, std, layers


def aggregate_layers_across_folds(
    per_fold: Mapping[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Macro-average per-layer accuracy across folds for one model."""
    return aggregate_layers_across_cells(
        {str(fold): values for fold, values in per_fold.items()}
    )


def _layer_accuracy_from_bundle(
    *,
    acts: np.ndarray,
    labels: Sequence[int],
    task_ids: Sequence[str],
    direction_vectors: Dict[int, np.ndarray],
    layer_indices: Sequence[int],
    data_split: str,
    model: str,
    cell_label: str,
) -> np.ndarray:
    if data_split == "test":
        correct, kept_task_ids = task_layer_correct(
            acts,
            labels,
            task_ids,
            direction_vectors,
            layer_indices,
        )
        if not kept_task_ids:
            logger.warning(
                "model=%s %s: no tasks with exactly 4 implementations and "
                "1 correct / 3 incorrect; layer accuracies will be NaN.",
                model,
                cell_label,
            )
        return per_fold_layer_accuracy(correct)
    return per_layer_pairwise_accuracy(
        acts,
        labels,
        task_ids,
        direction_vectors,
        layer_indices,
    )


def _load_bundle_for_cell(
    *,
    hidden_states_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    data_split: str,
    test_framing: str,
    val_framing: str,
    outer_fold: int | None,
    inner_fold: int,
) -> Dict[str, object]:
    if data_split == "test":
        if outer_fold is None:
            raise ValueError("outer_fold required for OOD test evaluation")
        bundle_split_family = (
            OOD_TEST_BUNDLE_SPLIT_FAMILY
            if is_ood_split_family(split_family)
            else split_family
        )
        return load_test_bundle(
            hidden_states_root=hidden_states_root,
            model=model,
            split_family=bundle_split_family,
            benchmark=benchmark,
            fold=outer_fold,
            test_framing=test_framing,
            fold_prefix="fold",
        )
    return load_validate_bundle(
        hidden_states_root=hidden_states_root,
        model=model,
        split_family=split_family,
        benchmark=benchmark,
        fold=inner_fold,
        val_framing=val_framing,
        fold_prefix="inner",
    )


def _build_output_dir(output_root: Path, split_family: str, benchmark: str) -> Path:
    origin_parts = viz_output_path_parts(split_family)
    out_dir = output_root / Path(*origin_parts) / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_for_model(
    *,
    hidden_states_root: Path,
    vectors_root: Path,
    benchmark: str,
    model: str,
    folds: Sequence[int],
    outer_folds: Sequence[int] | None,
    inner_folds: Sequence[int] | None,
    split_family: str,
    method: str,
    fit_framing: str,
    representation: str,
    data_split: str,
    test_framing: str,
    val_framing: str,
    include_embedding: bool,
) -> Dict[str, Any]:
    """Compute per-layer accuracy for one model on test or validation split."""
    if is_ood_split_family(split_family):
        return run_for_model_ood(
            hidden_states_root=hidden_states_root,
            vectors_root=vectors_root,
            benchmark=benchmark,
            model=model,
            outer_folds=outer_folds or [],
            inner_folds=inner_folds or [],
            split_family=split_family,
            method=method,
            fit_framing=fit_framing,
            representation=representation,
            data_split=data_split,
            test_framing=test_framing,
            val_framing=val_framing,
            include_embedding=include_embedding,
        )

    per_fold_layer_acc: Dict[int, np.ndarray] = {}
    layer_indices_ref: List[int] | None = None

    for fold in folds:
        if data_split == "test":
            bundle_blob = load_test_bundle(
                hidden_states_root=hidden_states_root,
                model=model,
                split_family=split_family,
                benchmark=benchmark,
                fold=fold,
                test_framing=test_framing,
            )
        elif data_split == "validate":
            bundle_blob = load_validate_bundle(
                hidden_states_root=hidden_states_root,
                model=model,
                split_family=split_family,
                benchmark=benchmark,
                fold=fold,
                val_framing=val_framing,
            )
        else:
            raise ValueError(f"unsupported data_split: {data_split!r}")
        acts_map = bundle_blob["acts"]
        if representation not in acts_map:
            raise KeyError(
                f"fold {fold}: representation {representation!r} missing in bundle acts; "
                f"available={list(acts_map.keys())}"
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

        layer_acc = _layer_accuracy_from_bundle(
            acts=acts,
            labels=labels,
            task_ids=task_ids,
            direction_vectors=direction_vectors,
            layer_indices=layer_indices,
            data_split=data_split,
            model=model,
            cell_label=f"fold={fold}",
        )

        if layer_indices_ref is None:
            layer_indices_ref = layer_indices
        elif layer_indices_ref != layer_indices:
            raise ValueError(
                f"layer indices mismatch across folds for model={model!r}: "
                f"{layer_indices_ref} vs {layer_indices}"
            )

        per_fold_layer_acc[fold] = layer_acc

    if layer_indices_ref is None:
        raise ValueError(f"model={model!r}: no folds processed")

    mean_per_layer, std_per_layer, rel_layers = aggregate_layers_across_folds(per_fold_layer_acc)
    if len(rel_layers) != len(layer_indices_ref):
        raise ValueError(
            "aggregate_layers_across_folds returned unexpected length: "
            f"{len(rel_layers)} vs {len(layer_indices_ref)}"
        )

    return {
        "model": model,
        "data_split": data_split,
        "folds": list(sorted(per_fold_layer_acc)),
        "layer_indices": layer_indices_ref,
        "per_fold_layer_accuracy": {
            str(fold): per_fold_layer_acc[fold].tolist() for fold in sorted(per_fold_layer_acc)
        },
        "mean_per_layer": mean_per_layer.tolist(),
        "std_per_layer": std_per_layer.tolist(),
    }


def run_for_model_ood(
    *,
    hidden_states_root: Path,
    vectors_root: Path,
    benchmark: str,
    model: str,
    outer_folds: Sequence[int],
    inner_folds: Sequence[int],
    split_family: str,
    method: str,
    fit_framing: str,
    representation: str,
    data_split: str,
    test_framing: str,
    val_framing: str,
    include_embedding: bool,
) -> Dict[str, Any]:
    """Compute per-layer accuracy for one OOD model (nested outer×inner CV)."""
    if not inner_folds:
        raise ValueError("OOD evaluation requires at least one inner fold")

    per_cell_layer_acc: Dict[str, np.ndarray] = {}
    layer_indices_ref: List[int] | None = None
    direction_cache: Dict[int, Dict[int, np.ndarray]] = {}

    if data_split == "test":
        if not outer_folds:
            raise ValueError("OOD test evaluation requires at least one outer fold")
        cells = [
            (outer, inner)
            for outer in sorted(set(outer_folds))
            for inner in sorted(set(inner_folds))
        ]
    else:
        cells = [(None, inner) for inner in sorted(set(inner_folds))]

    for outer_fold, inner_fold in cells:
        bundle_blob = _load_bundle_for_cell(
            hidden_states_root=hidden_states_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            data_split=data_split,
            test_framing=test_framing,
            val_framing=val_framing,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        )
        acts_map = bundle_blob["acts"]
        if representation not in acts_map:
            cell = (
                f"outer_{outer_fold}_inner_{inner_fold}"
                if outer_fold is not None
                else f"inner_{inner_fold}"
            )
            raise KeyError(
                f"{cell}: representation {representation!r} missing in bundle acts; "
                f"available={list(acts_map.keys())}"
            )
        acts = acts_map[representation].detach().cpu().numpy()
        task_ids = list(bundle_blob["task_ids"])
        labels = bundle_blob["labels"].detach().cpu().tolist()

        if inner_fold not in direction_cache:
            direction_cache[inner_fold] = load_direction_vectors(
                vectors_root=vectors_root,
                model=model,
                split_family=split_family,
                benchmark=benchmark,
                method=method,
                fit_framing=fit_framing,
                representation=representation,
                fold=inner_fold,
                fold_prefix="inner",
            )
        direction_vectors = direction_cache[inner_fold]

        layer_indices = layer_indices_for_bundle(
            acts.shape[1],
            include_embedding=include_embedding,
        )
        missing_layers = [layer for layer in layer_indices if layer not in direction_vectors]
        if missing_layers:
            cell = (
                f"outer_{outer_fold}_inner_{inner_fold}"
                if outer_fold is not None
                else f"inner_{inner_fold}"
            )
            raise KeyError(
                f"{cell}: direction vectors missing layer(s) {missing_layers}"
            )

        cell_label = (
            f"outer={outer_fold} inner={inner_fold}"
            if outer_fold is not None
            else f"inner={inner_fold}"
        )
        cell_key = (
            f"outer_{outer_fold}_inner_{inner_fold}"
            if outer_fold is not None
            else f"inner_{inner_fold}"
        )
        layer_acc = _layer_accuracy_from_bundle(
            acts=acts,
            labels=labels,
            task_ids=task_ids,
            direction_vectors=direction_vectors,
            layer_indices=layer_indices,
            data_split=data_split,
            model=model,
            cell_label=cell_label,
        )

        if layer_indices_ref is None:
            layer_indices_ref = layer_indices
        elif layer_indices_ref != layer_indices:
            raise ValueError(
                f"layer indices mismatch across OOD cells for model={model!r}: "
                f"{layer_indices_ref} vs {layer_indices} ({cell_label})"
            )

        per_cell_layer_acc[cell_key] = layer_acc

    if layer_indices_ref is None:
        raise ValueError(f"model={model!r}: no OOD cells processed")

    mean_per_layer, std_per_layer, rel_layers = aggregate_layers_across_cells(per_cell_layer_acc)
    if len(rel_layers) != len(layer_indices_ref):
        raise ValueError(
            "aggregate_layers_across_cells returned unexpected length: "
            f"{len(rel_layers)} vs {len(layer_indices_ref)}"
        )

    result: Dict[str, Any] = {
        "model": model,
        "data_split": data_split,
        "ood_source": ood_source_from_split_family(split_family),
        "inner_folds": list(sorted(set(inner_folds))),
        "per_cell_layer_accuracy": {
            cell: per_cell_layer_acc[cell].tolist()
            for cell in sorted(per_cell_layer_acc)
        },
        "layer_indices": layer_indices_ref,
        "mean_per_layer": mean_per_layer.tolist(),
        "std_per_layer": std_per_layer.tolist(),
    }
    if data_split == "test":
        result["outer_folds"] = list(sorted(set(outer_folds)))
        result["folds"] = {
            "outer": result["outer_folds"],
            "inner": result["inner_folds"],
        }
    else:
        result["folds"] = result["inner_folds"]
    return result


def plot_models(
    *,
    model_summaries: Sequence[Dict[str, Any]],
    split_family: str,
    benchmark: str,
    data_split: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Render and save the multi-model layer-accuracy plot and numeric dump."""
    if not model_summaries:
        raise ValueError("plot_models: no model summaries provided")

    out_dir = _build_output_dir(output_dir, split_family=split_family, benchmark=benchmark)
    stem = output_stem(data_split)
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    max_layer_index = max(
        int(max(summary["layer_indices"])) for summary in model_summaries
    )
    min_layer_index = min(
        int(min(summary["layer_indices"])) for summary in model_summaries
    )

    ordered_summaries = ordered_model_summaries(model_summaries)

    plt.figure(figsize=(6, 4))
    for summary in ordered_summaries:
        layer_indices = summary["layer_indices"]
        model_name = summary["model"]
        mean_per_layer = np.asarray(summary["mean_per_layer"], dtype=np.float64)
        plt.plot(
            layer_indices,
            mean_per_layer,
            marker="none",
            label=safe_model_dir(model_name),
            color=MODEL_COLOR_MAP.get(model_name),
        )

    plt.xlabel("Layer index")
    plt.ylabel(metric_label(data_split))
    plt.ylim(0.0, 1.0)
    plt.xlim(min_layer_index, max_layer_index)
    plt.grid(alpha=0.3)
    plt.legend(title="Model", fontsize="small")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    payload: Dict[str, Any] = {
        "split_family": split_family,
        "benchmark": benchmark,
        "data_split": data_split,
        "metric": metric_label(data_split),
        "x_axis_layer_range": [min_layer_index, max_layer_index],
        "models": {
            summary["model"]: {
                k: v
                for k, v in summary.items()
                if k
                not in {
                    "model",
                    "mean_per_layer",
                    "std_per_layer",
                    "layer_indices",
                }
            }
            | {
                "layer_indices": summary["layer_indices"],
                "mean_per_layer": summary["mean_per_layer"],
                "std_per_layer": summary["std_per_layer"],
            }
            for summary in ordered_summaries
        },
        "output_paths": {
            "png": str(png_path),
            "json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer accuracy plot from cached bundles (test 1-of-4 or validation pairwise).",
    )
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument(
        "--data-split",
        choices=DATA_SPLIT_CHOICES,
        default="test",
        help="Which cached split to evaluate: test (1-of-4) or validate (pairwise).",
    )
    parser.add_argument(
        "--data-root",
        default="dataset/icse26",
        help="ICSE26 root (dataset/icse26) or Mutation_DS split/phase2 dir; "
        "used for --folds all manifests.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more HuggingFace model ids.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="ID folds only: explicit indices (e.g. 0 3) or 'all'. "
        "Ignored for OOD split families.",
    )
    parser.add_argument(
        "--outer-folds",
        nargs="+",
        default=["all"],
        help="OOD only: outer (ID test) fold indices or 'all' (default).",
    )
    parser.add_argument(
        "--inner-folds",
        nargs="+",
        default=["all"],
        help="OOD only: inner (OOD fit/val) fold indices or 'all' (default).",
    )
    parser.add_argument(
        "--split-family",
        default="icse26",
        help="Dataset origin: icse26, mutation_ds split name, ood_mbpp, or ood_synthetic.",
    )
    parser.add_argument("--hidden-states-root", default="hidden_states")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Include layer 0 (embedding output) in per-layer accuracy (excluded by default).",
    )
    parser.add_argument("--output-dir", default="analysis/layer_accuracy_plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = Path(args.data_root)
    if is_ood_split_family(args.split_family):
        ood_source = ood_source_from_split_family(args.split_family)
        outer_folds = parse_ood_fold_selector(
            args.outer_folds,
            data_root=data_root,
            benchmark=args.benchmark,
            fold_kind="outer",
        )
        inner_folds = parse_ood_fold_selector(
            args.inner_folds,
            data_root=data_root,
            ood_source=ood_source,
            fold_kind="inner",
        )
        folds: List[int] = []
        logger.info(
            "OOD split family %s: outer_folds=%s inner_folds=%s",
            args.split_family,
            outer_folds,
            inner_folds,
        )
    else:
        outer_folds = None
        inner_folds = None
        folds = parse_fold_selector(
            args.folds,
            split_family=args.split_family,
            data_root=data_root,
            benchmark=args.benchmark,
        )

    model_summaries: List[Dict[str, Any]] = []
    for model in args.models:
        logger.info("Computing per-layer accuracy for model=%s", model)
        summary = run_for_model(
            hidden_states_root=Path(args.hidden_states_root),
            vectors_root=Path(args.vectors_root),
            benchmark=args.benchmark,
            model=model,
            folds=folds,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            split_family=args.split_family,
            method=args.method,
            fit_framing=args.fit_framing,
            representation=args.representation,
            data_split=args.data_split,
            test_framing=args.test_framing,
            val_framing=args.val_framing,
            include_embedding=args.include_embedding,
        )
        model_summaries.append(summary)

    png_path, json_path = plot_models(
        model_summaries=model_summaries,
        split_family=args.split_family,
        benchmark=args.benchmark,
        data_split=args.data_split,
        output_dir=Path(args.output_dir),
    )

    summary_payload = {
        "benchmark": args.benchmark,
        "split_family": args.split_family,
        "data_split": args.data_split,
        "metric": metric_label(args.data_split),
        "models": [s["model"] for s in model_summaries],
        "folds": (
            {"outer": outer_folds, "inner": inner_folds}
            if is_ood_split_family(args.split_family)
            else folds
        ),
        "include_embedding": args.include_embedding,
        "output_paths": {
            "png": str(png_path),
            "json": str(json_path),
        },
    }
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()

