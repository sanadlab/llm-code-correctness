"""Score cached hidden-state bundles without re-running the model.

Use this when you have already extracted ``bundle.pt`` files (e.g. for a new
validation split with 4 candidates per task) and want per-layer accuracy
results written in the same ``results.json`` format as the live experiments.

Typical workflow for ``icse26_val_1_4``:
  1. Reuse existing fit directions from ``vectors/`` (fit split unchanged).
  2. Load the new 4-option validation bundles from ``hidden_states/``.
  3. Optionally load the unchanged test bundles.
  4. Write ``results.json`` with both ``one_of_4`` and ``pairwise_acc`` on val.

Usage:
    python scripts/experiments/evaluate_from_cache.py \\
        --model codellama/CodeLlama-7b-Instruct-hf \\
        --benchmark bigcodebench \\
        --split-family icse26_val_1_4 \\
        --vectors-split-family icse26 \\
        --hidden-states-root /scratch/re2230/hidden_states_re2230 \\
        --vectors-root vectors \\
        --results-dir results_val_1_4 \\
        --representation response_avg \\
        --method mean_diff \\
        --fit-framing function \\
        --val-framing neutral \\
        --test-framing neutral \\
        --layer-selection-metric one_of_4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from scripts.experiments._common import (
    REPRESENTATIONS,
    per_layer_split_metrics,
    select_layer_by_metric,
)
from src.hidden_states import HiddenStateBundle, bundle_output_dir
from src.hidden_states.cache import git_commit, vector_common_path
from src.vectors.direction import Direction
from src.vectors.projection import project_all_layers


def load_bundle(
    *,
    hidden_states_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    distribution: str,
    data_split: str,
    framing: str,
    fold: int,
) -> HiddenStateBundle:
    bundle_dir = bundle_output_dir(
        hidden_states_root,
        model,
        split_family,
        benchmark,
        distribution,
        data_split,
        framing,
        fold,
    )
    bundle_path = bundle_dir / "bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"missing cached bundle: {bundle_path}")
    return HiddenStateBundle.load(bundle_path)


def load_direction(
    *,
    vectors_root: Path,
    model: str,
    vectors_split_family: str,
    benchmark: str,
    distribution: str,
    method: str,
    fit_framing: str,
    representation: str,
    fold: int,
) -> Direction:
    common = vector_common_path(
        model,
        vectors_split_family,
        benchmark,
        distribution,
        method,
        fit_framing,
        representation,
    )
    direction_path = vectors_root / common / f"fold_{fold}" / "direction.pt"
    if not direction_path.is_file():
        raise FileNotFoundError(f"missing direction vector: {direction_path}")
    return Direction.load(direction_path)


def evaluate_fold(
    *,
    fold: int,
    representation,
    hidden_states_root: Path,
    vectors_root: Path,
    results_dir: Path,
    model: str,
    split_family: str,
    vectors_split_family: str,
    benchmark: str,
    distribution: str,
    method: str,
    fit_framing: str,
    val_framing: str,
    test_framing: str,
    layer_selection_metric: str,
    score_test: bool,
    experiment_script: str,
) -> Dict[str, object]:
    direction = load_direction(
        vectors_root=vectors_root,
        model=model,
        vectors_split_family=vectors_split_family,
        benchmark=benchmark,
        distribution=distribution,
        method=method,
        fit_framing=fit_framing,
        representation=representation.name,
        fold=fold,
    )
    bundle_val = load_bundle(
        hidden_states_root=hidden_states_root,
        model=model,
        split_family=split_family,
        benchmark=benchmark,
        distribution=distribution,
        data_split="validate",
        framing=val_framing,
        fold=fold,
    )
    acts_val = representation.select(bundle_val)
    layers, val_scores_all = project_all_layers(direction, acts_val)
    val_per_layer = per_layer_split_metrics(
        val_scores_all, layers, bundle_val.labels, bundle_val.task_ids
    )
    selected_layer = select_layer_by_metric(val_per_layer, layer_selection_metric)
    sel_val = val_per_layer[selected_layer]

    test_per_layer: Optional[Dict[int, Dict[str, float]]] = None
    test_scores_all = None
    bundle_test = None
    sel_test = None
    best_layer = None
    if score_test:
        bundle_test = load_bundle(
            hidden_states_root=hidden_states_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            distribution=distribution,
            data_split="test",
            framing=test_framing,
            fold=fold,
        )
        acts_test = representation.select(bundle_test)
        _, test_scores_all = project_all_layers(direction, acts_test)
        test_per_layer = per_layer_split_metrics(
            test_scores_all, layers, bundle_test.labels, bundle_test.task_ids
        )
        sel_test = test_per_layer[selected_layer]
        best_layer = select_layer_by_metric(test_per_layer, "one_of_4")

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

    common = vector_common_path(
        model,
        split_family,
        benchmark,
        distribution,
        method,
        fit_framing,
        representation.name,
    )
    eval_segment = f"val={val_framing}__test={test_framing}"
    res_fold = results_dir / common / eval_segment / f"fold_{fold}"
    res_fold.mkdir(parents=True, exist_ok=True)

    detailed = {
        "fold": fold,
        "layers": list(layers),
        "val": {
            "sample_ids": list(bundle_val.sample_ids),
            "task_ids": list(bundle_val.task_ids),
            "labels": bundle_val.labels.tolist(),
            "scores": val_scores_all.float().tolist(),
        },
    }
    if score_test and bundle_test is not None and test_scores_all is not None:
        detailed["test"] = {
            "sample_ids": list(bundle_test.sample_ids),
            "task_ids": list(bundle_test.task_ids),
            "labels": bundle_test.labels.tolist(),
            "scores": test_scores_all.float().tolist(),
        }
    with open(res_fold / "detailed.json", "w") as f:
        json.dump(detailed, f)

    results = {
        "fold": fold,
        "selected_layer": selected_layer,
        "layer_selection_metric": layer_selection_metric,
        "val_per_layer": {str(k): v for k, v in val_per_layer.items()},
        "val_at_selected_layer": dict(sel_val),
    }
    if test_per_layer is not None and sel_test is not None and best_layer is not None:
        results.update(
            {
                "test_per_layer": {str(k): v for k, v in test_per_layer.items()},
                "test_at_selected_layer": dict(sel_test),
                "test_theoretical_best_layer": {
                    "layer": best_layer,
                    **test_per_layer[best_layer],
                },
            }
        )
    with open(res_fold / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(res_fold / "metadata.json", "w") as f:
        json.dump(
            {
                "vector_path": str(
                    vectors_root
                    / vector_common_path(
                        model,
                        vectors_split_family,
                        benchmark,
                        distribution,
                        method,
                        fit_framing,
                        representation.name,
                    )
                    / f"fold_{fold}"
                    / "direction.pt"
                ),
                "model": model,
                "dataset": benchmark,
                "split_family": split_family,
                "vectors_split_family": vectors_split_family,
                "distribution": distribution,
                "fold": fold,
                "method": method,
                "representation": representation.name,
                "fit_framing": fit_framing,
                "val_framing": val_framing,
                "test_framing": test_framing,
                "n_val": bundle_val.num_samples,
                "n_test": bundle_test.num_samples if bundle_test is not None else None,
                "experiment_script": experiment_script,
                "git_commit": git_commit(),
            },
            f,
            indent=2,
        )

    return {
        "fold": fold,
        "selected_layer": selected_layer,
        "val_one_of_4": sel_val["one_of_4"],
        "val_pairwise_acc": sel_val["pairwise_acc"],
        "test_one_of_4": sel_test["one_of_4"] if sel_test is not None else None,
        "test_pairwise": sel_test["pairwise_acc"] if sel_test is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument("--split-family", required=True,
                        help="Split family for cached bundles and results output.")
    parser.add_argument("--vectors-split-family", default=None,
                        help="Split family for direction vectors (default: same as --split-family).")
    parser.add_argument("--distribution", default="id")
    parser.add_argument("--hidden-states-root", required=True)
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--representation", default="response_avg", choices=list(REPRESENTATIONS))
    parser.add_argument("--method", default="mean_diff", choices=["mean_diff", "pca"])
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--layer-selection-metric",
        default="one_of_4",
        choices=["one_of_4", "pairwise_acc"],
        help="Metric used to pick the monitoring layer on validation.",
    )
    parser.add_argument("--folds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only score validation bundles (useful when test caches are not extracted yet).",
    )
    args = parser.parse_args()

    representation = REPRESENTATIONS[args.representation]
    vectors_split_family = args.vectors_split_family or args.split_family

    per_fold: List[Dict[str, object]] = []
    for fold in args.folds:
        per_fold.append(
            evaluate_fold(
                fold=fold,
                representation=representation,
                hidden_states_root=Path(args.hidden_states_root),
                vectors_root=Path(args.vectors_root),
                results_dir=Path(args.results_dir),
                model=args.model,
                split_family=args.split_family,
                vectors_split_family=vectors_split_family,
                benchmark=args.benchmark,
                distribution=args.distribution,
                method=args.method,
                fit_framing=args.fit_framing,
                val_framing=args.val_framing,
                test_framing=args.test_framing,
                layer_selection_metric=args.layer_selection_metric,
                score_test=not args.skip_test,
                experiment_script="scripts/experiments/evaluate_from_cache.py",
            )
        )

    mean_val_1of4 = sum(float(x["val_one_of_4"]) for x in per_fold) / len(per_fold)
    print(f"\n[summary] mean val 1-of-4 = {mean_val_1of4:.3f}  (n_folds={len(per_fold)})")
    if per_fold[0]["test_one_of_4"] is not None:
        mean_test_1of4 = sum(float(x["test_one_of_4"]) for x in per_fold) / len(per_fold)
        print(f"[summary] mean test 1-of-4 = {mean_test_1of4:.3f}")


if __name__ == "__main__":
    main()
