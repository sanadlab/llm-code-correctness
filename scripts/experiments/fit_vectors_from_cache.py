"""Fit correctness directions from cached fit bundles, then score validate/test.

Workflow:
  1. Load cached **fit** hidden-state bundles (e.g. function framing).
  2. Fit a per-layer direction (``mean_diff`` or ``pca``) and write ``vectors/``.
  3. Project cached **validate** (and optionally **test**) bundles.
  4. Pick the monitoring layer on validation (default: ``one_of_4``).
  5. Write per-fold ``results.json``, ``detailed.json``, and ``metadata.json``.

Typical usage for ``icse26_val_1_4`` (after caching fit bundles):

    python scripts/experiments/fit_vectors_from_cache.py \\
        --model codellama/CodeLlama-7b-Instruct-hf \\
        --benchmark bigcodebench \\
        --split-family icse26_val_1_4 \\
        --hidden-states-root /scratch/re2230/hidden_states_re2230

Use ``--layer-selection-metric pairwise_acc`` for the original ICSE26 protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments._common import (
    METHODS,
    REPRESENTATIONS,
    per_layer_split_metrics,
    select_layer_by_metric,
)
from src.hidden_states import HiddenStateBundle, bundle_output_dir, git_commit, safe_model_dir
from src.vectors.direction import Direction
from src.vectors.projection import project_all_layers

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN_STATES_ROOT = _REPO_ROOT.parent / "hidden_states_re2230"
DEFAULT_VECTORS_ROOT = _REPO_ROOT / "vectors"
DEFAULT_RESULTS_ROOT = _REPO_ROOT / "results_icse26_val_1_4"
EXPERIMENT_SCRIPT = "scripts/experiments/fit_vectors_from_cache.py"


def _num_folds(splits_dir: Path) -> int:
    manifest = json.loads((splits_dir / "manifest.json").read_text())
    return manifest["num_folds"]


def _load_bundle(
    hidden_states_root: Path,
    model_name: str,
    split_family: str,
    benchmark: str,
    distribution: str,
    data_split: str,
    framing: str,
    fold: int,
    fold_prefix: str = "fold",
) -> Tuple[HiddenStateBundle, Path]:
    out_dir = bundle_output_dir(
        output_root=hidden_states_root,
        model_name=model_name,
        split_family=split_family,
        benchmark=benchmark,
        distribution=distribution,
        data_split=data_split,
        framing=framing,
        fold=fold,
        fold_prefix=fold_prefix,
    )
    bundle_path = out_dir / "bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"missing cached bundle: {bundle_path} "
            "(run scripts/extract/cache_hidden_states.py first)"
        )
    return HiddenStateBundle.load(bundle_path), out_dir


def _assert_representation(
    bundle: HiddenStateBundle, bundle_dir: Path, representation
) -> None:
    try:
        representation.select(bundle)
    except KeyError as exc:
        meta_path = bundle_dir / "metadata.json"
        meta_reps: List[str] = []
        if meta_path.is_file():
            meta_reps = json.loads(meta_path.read_text()).get("representations", [])
        raise KeyError(
            f"{exc} at {bundle_dir / 'bundle.pt'}"
            + (f"; metadata.json lists {meta_reps!r}" if meta_reps else "")
        ) from exc


def _fit_direction(
    bundle_fit: HiddenStateBundle,
    representation,
    method,
) -> Direction:
    acts_fit = representation.select(bundle_fit)
    direction = method.fit(acts_fit, bundle_fit.labels, bundle_fit.task_ids)
    direction.representation = representation.name
    return direction


def _score_bundle(
    direction: Direction,
    representation,
    bundle: HiddenStateBundle,
) -> Tuple[Sequence[int], torch.Tensor, Dict[int, Dict[str, float]]]:
    acts = representation.select(bundle)
    layers, scores_all = project_all_layers(direction, acts)
    per_layer = per_layer_split_metrics(
        scores_all, layers, bundle.labels, bundle.task_ids
    )
    return layers, scores_all, per_layer


def _common_path(
    model: str,
    split_family: str,
    benchmark: str,
    distribution: str,
    method: str,
    fit_framing: str,
    representation: str,
) -> str:
    return (
        f"{safe_model_dir(model)}/{split_family}/{benchmark}/{distribution}/"
        f"{method}/{fit_framing}/{representation}"
    )


def _persist_vectors(
    vectors_root: Path,
    common_path: str,
    fold: int,
    direction: Direction,
    *,
    model: str,
    benchmark: str,
    split_family: str,
    distribution: str,
    method: str,
    fit_framing: str,
    representation: str,
    n_fit: int,
) -> Path:
    vec_fold = vectors_root / common_path / f"fold_{fold}"
    vec_fold.mkdir(parents=True, exist_ok=True)
    direction_path = vec_fold / "direction.pt"
    direction.save(direction_path)
    vec_meta = {
        "model": model,
        "dataset": benchmark,
        "split_family": split_family,
        "distribution": distribution,
        "fold": fold,
        "method": method,
        "representation": representation,
        "fit_framing": fit_framing,
        "chat_template": True,
        "dtype": "float16",
        "n_fit": n_fit,
        "experiment_script": EXPERIMENT_SCRIPT,
        "git_commit": git_commit(),
    }
    with open(vec_fold / "metadata.json", "w") as f:
        json.dump(vec_meta, f, indent=2)
    return direction_path


def _persist_results(
    results_root: Path,
    common_path: str,
    eval_segment: str,
    fold: int,
    *,
    direction_path: Path,
    model: str,
    benchmark: str,
    split_family: str,
    distribution: str,
    method: str,
    fit_framing: str,
    val_framing: str,
    test_framing: str,
    representation: str,
    layer_selection_metric: str,
    selected_layer: int,
    layers: Sequence[int],
    bundle_val: HiddenStateBundle,
    bundle_test: Optional[HiddenStateBundle],
    val_scores_all: torch.Tensor,
    test_scores_all: Optional[torch.Tensor],
    val_per_layer: Dict[int, Dict[str, float]],
    test_per_layer: Optional[Dict[int, Dict[str, float]]],
    n_fit: int,
) -> None:
    res_fold = results_root / common_path / eval_segment / f"fold_{fold}"
    res_fold.mkdir(parents=True, exist_ok=True)

    sel_val = val_per_layer[selected_layer]
    detailed: dict = {
        "fold": fold,
        "layers": list(layers),
        "val": {
            "sample_ids": list(bundle_val.sample_ids),
            "task_ids": list(bundle_val.task_ids),
            "labels": bundle_val.labels.tolist(),
            "scores": val_scores_all.float().tolist(),
        },
    }
    if bundle_test is not None and test_scores_all is not None:
        detailed["test"] = {
            "sample_ids": list(bundle_test.sample_ids),
            "task_ids": list(bundle_test.task_ids),
            "labels": bundle_test.labels.tolist(),
            "scores": test_scores_all.float().tolist(),
        }
    with open(res_fold / "detailed.json", "w") as f:
        json.dump(detailed, f)

    results: dict = {
        "fold": fold,
        "selected_layer": selected_layer,
        "layer_selection_metric": layer_selection_metric,
        "val_per_layer": {str(k): v for k, v in val_per_layer.items()},
        "val_at_selected_layer": dict(sel_val),
        "val_pairwise_acc": sel_val["pairwise_acc"],
        "val_one_of_4": sel_val["one_of_4"],
    }
    if test_per_layer is not None:
        sel_test = test_per_layer[selected_layer]
        best_layer = select_layer_by_metric(test_per_layer, "one_of_4")
        results.update(
            {
                "test_per_layer": {str(k): v for k, v in test_per_layer.items()},
                "test_at_selected_layer": dict(sel_test),
                "test_theoretical_best_layer": {
                    "layer": best_layer,
                    **test_per_layer[best_layer],
                },
                "test_one_of_4": sel_test["one_of_4"],
                "test_pairwise": sel_test["pairwise_acc"],
            }
        )
    with open(res_fold / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(res_fold / "metadata.json", "w") as f:
        json.dump(
            {
                "vector_path": str(direction_path),
                "model": model,
                "dataset": benchmark,
                "split_family": split_family,
                "distribution": distribution,
                "fold": fold,
                "method": method,
                "representation": representation,
                "fit_framing": fit_framing,
                "val_framing": val_framing,
                "test_framing": test_framing,
                "n_fit": n_fit,
                "n_val": bundle_val.num_samples,
                "n_test": bundle_test.num_samples if bundle_test is not None else None,
                "experiment_script": EXPERIMENT_SCRIPT,
                "git_commit": git_commit(),
            },
            f,
            indent=2,
        )


def process_fold(
    *,
    fold: int,
    representation,
    method,
    hidden_states_root: Path,
    vectors_root: Path,
    results_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    distribution: str,
    fit_framing: str,
    val_framing: str,
    test_framing: str,
    layer_selection_metric: str,
    score_test: bool,
) -> Dict[str, object]:
    bundle_fit, fit_dir = _load_bundle(
        hidden_states_root,
        model,
        split_family,
        benchmark,
        distribution,
        "fit",
        fit_framing,
        fold,
    )
    _assert_representation(bundle_fit, fit_dir, representation)

    bundle_val, val_dir = _load_bundle(
        hidden_states_root,
        model,
        split_family,
        benchmark,
        distribution,
        "validate",
        val_framing,
        fold,
    )
    _assert_representation(bundle_val, val_dir, representation)

    direction = _fit_direction(bundle_fit, representation, method)

    common_path = _common_path(
        model,
        split_family,
        benchmark,
        distribution,
        method.name,
        fit_framing,
        representation.name,
    )
    direction_path = _persist_vectors(
        vectors_root,
        common_path,
        fold,
        direction,
        model=model,
        benchmark=benchmark,
        split_family=split_family,
        distribution=distribution,
        method=method.name,
        fit_framing=fit_framing,
        representation=representation.name,
        n_fit=bundle_fit.num_samples,
    )

    layers, val_scores_all, val_per_layer = _score_bundle(direction, representation, bundle_val)
    selected_layer = select_layer_by_metric(val_per_layer, layer_selection_metric)
    sel_val = val_per_layer[selected_layer]

    bundle_test: Optional[HiddenStateBundle] = None
    test_scores_all: Optional[torch.Tensor] = None
    test_per_layer: Optional[Dict[int, Dict[str, float]]] = None
    sel_test = None
    if score_test:
        bundle_test, test_dir = _load_bundle(
            hidden_states_root,
            model,
            split_family,
            benchmark,
            distribution,
            "test",
            test_framing,
            fold,
        )
        _assert_representation(bundle_test, test_dir, representation)
        _, test_scores_all, test_per_layer = _score_bundle(
            direction, representation, bundle_test
        )
        sel_test = test_per_layer[selected_layer]

    print(
        f"[fold {fold}] selected layer = {selected_layer} "
        f"(val {layer_selection_metric} = {sel_val[layer_selection_metric]:.3f}, "
        f"val one_of_4 = {sel_val['one_of_4']:.3f})"
    )
    if sel_test is not None:
        print(
            f"[fold {fold}] test 1-of-4 = {sel_test['one_of_4']:.3f}  "
            f"test pairwise = {sel_test['pairwise_acc']:.3f}"
        )

    eval_segment = f"val={val_framing}__test={test_framing}"
    _persist_results(
        results_root,
        common_path,
        eval_segment,
        fold,
        direction_path=direction_path,
        model=model,
        benchmark=benchmark,
        split_family=split_family,
        distribution=distribution,
        method=method.name,
        fit_framing=fit_framing,
        val_framing=val_framing,
        test_framing=test_framing,
        representation=representation.name,
        layer_selection_metric=layer_selection_metric,
        selected_layer=selected_layer,
        layers=layers,
        bundle_val=bundle_val,
        bundle_test=bundle_test,
        val_scores_all=val_scores_all,
        test_scores_all=test_scores_all,
        val_per_layer=val_per_layer,
        test_per_layer=test_per_layer,
        n_fit=bundle_fit.num_samples,
    )

    return {
        "fold": fold,
        "selected_layer": selected_layer,
        "val_one_of_4": sel_val["one_of_4"],
        "val_pairwise_acc": sel_val["pairwise_acc"],
        "test_one_of_4": sel_test["one_of_4"] if sel_test is not None else None,
        "test_pairwise": sel_test["pairwise_acc"] if sel_test is not None else None,
    }


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "benchmark",
        "split_family",
        "fold",
        "selected_layer",
        "val_one_of_4",
        "val_pairwise_acc",
        "test_one_of_4",
        "test_pairwise_acc",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit vectors from cached fit bundles and score validate/test."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument(
        "--split-family",
        default="icse26_val_1_4",
        help="Split family for cached bundles, vectors, and results (default: icse26_val_1_4).",
    )
    parser.add_argument("--distribution", default="id")
    parser.add_argument(
        "--hidden-states-root",
        type=Path,
        default=DEFAULT_HIDDEN_STATES_ROOT,
        help=f"Root with cached bundle.pt files (default: {DEFAULT_HIDDEN_STATES_ROOT}).",
    )
    parser.add_argument(
        "--vectors-root",
        type=Path,
        default=DEFAULT_VECTORS_ROOT,
        help=f"Output root for direction.pt files (default: {DEFAULT_VECTORS_ROOT}).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Output root for results.json files (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory with manifest.json for fold count (default: infer from --split-family).",
    )
    parser.add_argument("--representation", default="response_avg", choices=list(REPRESENTATIONS))
    parser.add_argument("--method", default="mean_diff", choices=list(METHODS))
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--layer-selection-metric",
        default="one_of_4",
        choices=["one_of_4", "pairwise_acc"],
        help="Metric used to pick the monitoring layer on validation (default: one_of_4).",
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only score validation bundles (skip test when caches are unavailable).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional per-fold summary CSV (default: <results-root>/summary.csv).",
    )
    return parser.parse_args()


def resolve_folds(args: argparse.Namespace) -> List[int]:
    if args.folds is not None:
        return list(args.folds)
    splits_dir = args.splits_dir
    if splits_dir is None:
        candidate = _REPO_ROOT.parent / args.split_family
        if (candidate / "manifest.json").is_file():
            splits_dir = candidate
        else:
            candidate = _REPO_ROOT / "dataset" / "icse26" / args.benchmark
            if (candidate / "manifest.json").is_file():
                splits_dir = candidate
    if splits_dir is None or not (splits_dir / "manifest.json").is_file():
        return list(range(10))
    return list(range(_num_folds(splits_dir)))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()
    representation = REPRESENTATIONS[args.representation]
    method = METHODS[args.method]
    folds = resolve_folds(args)
    summary_csv = args.summary_csv or (args.results_root / "summary.csv")

    logger.info("Hidden states root: %s", args.hidden_states_root.resolve())
    logger.info("Vectors root: %s", args.vectors_root.resolve())
    logger.info("Results root: %s", args.results_root.resolve())
    logger.info("Split family: %s", args.split_family)
    logger.info("Layer selection metric: %s", args.layer_selection_metric)
    logger.info("Folds: %s", folds)

    per_fold: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for fold in folds:
        fold_result = process_fold(
            fold=fold,
            representation=representation,
            method=method,
            hidden_states_root=args.hidden_states_root,
            vectors_root=args.vectors_root,
            results_root=args.results_root,
            model=args.model,
            split_family=args.split_family,
            benchmark=args.benchmark,
            distribution=args.distribution,
            fit_framing=args.fit_framing,
            val_framing=args.val_framing,
            test_framing=args.test_framing,
            layer_selection_metric=args.layer_selection_metric,
            score_test=not args.skip_test,
        )
        per_fold.append(fold_result)
        summary_rows.append(
            {
                "model": args.model,
                "benchmark": args.benchmark,
                "split_family": args.split_family,
                "fold": fold,
                "selected_layer": fold_result["selected_layer"],
                "val_one_of_4": fold_result["val_one_of_4"],
                "val_pairwise_acc": fold_result["val_pairwise_acc"],
                "test_one_of_4": fold_result["test_one_of_4"],
                "test_pairwise_acc": fold_result["test_pairwise"],
            }
        )

    write_summary_csv(summary_csv, summary_rows)

    mean_val_1of4 = sum(float(x["val_one_of_4"]) for x in per_fold) / len(per_fold)
    mean_val_pair = sum(float(x["val_pairwise_acc"]) for x in per_fold) / len(per_fold)
    print(
        f"\n[summary] mean val 1-of-4 = {mean_val_1of4:.3f}  "
        f"mean val pairwise = {mean_val_pair:.3f}  (n_folds={len(per_fold)})"
    )
    if per_fold[0]["test_one_of_4"] is not None:
        mean_test_1of4 = sum(float(x["test_one_of_4"]) for x in per_fold) / len(per_fold)
        mean_test_pair = sum(float(x["test_pairwise"]) for x in per_fold) / len(per_fold)
        print(
            f"[summary] mean test 1-of-4 = {mean_test_1of4:.3f}  "
            f"mean test pairwise = {mean_test_pair:.3f}"
        )
    print(f"[summary] wrote per-fold table to {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
