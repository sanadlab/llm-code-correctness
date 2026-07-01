"""Fit a direction and score it against cached hidden states.

This is the cheap half of the large-scale ablation: `cache_hidden_states.py`
does the expensive part (model forward passes) once per (model,
split_family, dataset, fold, data_split, framing, representation) cell, and
this script reuses those cached bundles to fit+score every
(method, representation, fit_framing, val_framing, test_framing, fit_source,
test_dataset) combination without touching the model.

Two fit sources:
  in-domain: fit/validate/test all come from the same dataset's folds
             (split_family="icse26", fold_prefix="fold").
  mbpp | synthetic: nested CV. fit/validate come from the OOD source's
             inner folds; test comes from the target dataset's outer folds.
             (split_family=f"ood_{source}", fold_prefix="inner" for
             fit/validate, "fold" for test.)

Usage:
    python scripts/experiments/fit_and_score_from_cache.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --representation response_avg --method mean_diff \\
        --fit-framing function --val-framing neutral --test-framing neutral \\
        --fit-source in-domain --test-dataset bigcodebench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from scripts.experiments._common import (
    METHODS,
    REPRESENTATIONS,
    per_layer_split_metrics,
    select_layer_by_metric,
)
from src.hidden_states import HiddenStateBundle, bundle_output_dir, git_commit, safe_model_dir
from src.vectors.projection import project_all_layers


def _num_folds(data_root: Path, dataset: str) -> int:
    manifest = json.loads((data_root / dataset / "manifest.json").read_text())
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
    fold_prefix: str,
) -> HiddenStateBundle:
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


def _assert_representation(bundle: HiddenStateBundle, bundle_dir: Path, representation) -> None:
    """Verify the requested representation exists inside bundle.pt."""
    try:
        representation.select(bundle)
        return
    except KeyError as exc:
        meta_path = bundle_dir / "metadata.json"
        meta_reps: List[str] = []
        if meta_path.is_file():
            meta_reps = json.loads(meta_path.read_text()).get("representations", [])
        raise KeyError(
            f"{exc} at {bundle_dir / 'bundle.pt'}"
            + (f"; metadata.json lists {meta_reps!r}" if meta_reps else "")
        ) from exc


def _fit_and_select(
    bundle_fit: HiddenStateBundle,
    bundle_val: HiddenStateBundle,
    representation,
    method,
):
    """Fit a direction on `bundle_fit`, select the best layer on `bundle_val`.

    Mirrors the selection logic in `_common.run_cross_validation`.
    """
    acts_fit = representation.select(bundle_fit)
    direction = method.fit(acts_fit, bundle_fit.labels, bundle_fit.task_ids)
    direction.representation = representation.name

    acts_val = representation.select(bundle_val)
    layers, val_scores_all = project_all_layers(direction, acts_val)
    val_per_layer = per_layer_split_metrics(
        val_scores_all, layers, bundle_val.labels, bundle_val.task_ids
    )
    selected_layer = select_layer_by_metric(val_per_layer, "pairwise_acc")
    return direction, layers, val_scores_all, val_per_layer, selected_layer


def _score_test(direction, layers, representation, bundle_test: HiddenStateBundle):
    acts_test = representation.select(bundle_test)
    _, test_scores_all = project_all_layers(direction, acts_test)
    test_per_layer = per_layer_split_metrics(
        test_scores_all, layers, bundle_test.labels, bundle_test.task_ids
    )
    return test_scores_all, test_per_layer


def _stringify_per_layer(per_layer: Dict[int, Dict[str, float]]) -> dict:
    return {str(k): v for k, v in per_layer.items()}


def _build_results_payload(
    *,
    cell_ids: dict,
    selected_layer: int,
    val_per_layer: Dict[int, Dict[str, float]],
    test_per_layer: Dict[int, Dict[str, float]],
    metadata: dict,
) -> dict:
    sel_val = val_per_layer[selected_layer]
    sel_test = test_per_layer[selected_layer]
    best_layer = select_layer_by_metric(test_per_layer, "one_of_4")
    return {
        **cell_ids,
        "selected_layer": selected_layer,
        "layer_selection_metric": "pairwise_acc",
        "val_per_layer": _stringify_per_layer(val_per_layer),
        "test_per_layer": _stringify_per_layer(test_per_layer),
        "val_at_selected_layer": dict(sel_val),
        "test_at_selected_layer": dict(sel_test),
        "test_theoretical_best_layer": {
            "layer": best_layer,
            **test_per_layer[best_layer],
        },
        "val_pairwise_acc": sel_val["pairwise_acc"],
        "test_one_of_4": sel_test["one_of_4"],
        "test_pairwise": sel_test["pairwise_acc"],
        **metadata,
    }


def _cell_dir(results_dir: Path, common_path: str, eval_segment: str, cell_name: str) -> Path:
    res_dir = Path(results_dir) / common_path / eval_segment / cell_name
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def _write_results(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_name: str,
    payload: dict,
) -> None:
    res_dir = _cell_dir(results_dir, common_path, eval_segment, cell_name)
    with open(res_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_detailed(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_name: str,
    *,
    cell_ids: dict,
    layers: Sequence[int],
    bundle_val: HiddenStateBundle,
    bundle_test: HiddenStateBundle,
    val_scores_all: torch.Tensor,
    test_scores_all: torch.Tensor,
) -> None:
    res_dir = _cell_dir(results_dir, common_path, eval_segment, cell_name)
    detailed = {
        **cell_ids,
        "layers": list(layers),
        "val": {
            "sample_ids": list(bundle_val.sample_ids),
            "task_ids": list(bundle_val.task_ids),
            "labels": bundle_val.labels.tolist(),
            "scores": val_scores_all.float().tolist(),
        },
        "test": {
            "sample_ids": list(bundle_test.sample_ids),
            "task_ids": list(bundle_test.task_ids),
            "labels": bundle_test.labels.tolist(),
            "scores": test_scores_all.float().tolist(),
        },
    }
    with open(res_dir / "detailed.json", "w") as f:
        json.dump(detailed, f)


def _metadata_payload(args, representation, method, fit_source: str) -> dict:
    return {
        "model": args.model,
        "test_dataset": args.test_dataset,
        "fit_source": fit_source,
        "method": method.name,
        "representation": representation.name,
        "fit_framing": args.fit_framing,
        "val_framing": args.val_framing,
        "test_framing": args.test_framing,
        "git_commit": git_commit(),
        "experiment_script": "scripts/experiments/fit_and_score_from_cache.py",
    }


def _persist_cell(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_name: str,
    *,
    cell_ids: dict,
    layers: Sequence[int],
    bundle_val: HiddenStateBundle,
    bundle_test: HiddenStateBundle,
    val_scores_all: torch.Tensor,
    test_scores_all: torch.Tensor,
    selected_layer: int,
    val_per_layer: Dict[int, Dict[str, float]],
    test_per_layer: Dict[int, Dict[str, float]],
    metadata: dict,
) -> dict:
    payload = _build_results_payload(
        cell_ids=cell_ids,
        selected_layer=selected_layer,
        val_per_layer=val_per_layer,
        test_per_layer=test_per_layer,
        metadata=metadata,
    )
    _write_results(results_dir, common_path, eval_segment, cell_name, payload)
    _write_detailed(
        results_dir,
        common_path,
        eval_segment,
        cell_name,
        cell_ids=cell_ids,
        layers=layers,
        bundle_val=bundle_val,
        bundle_test=bundle_test,
        val_scores_all=val_scores_all,
        test_scores_all=test_scores_all,
    )
    return payload


def run_in_domain(args, representation, method) -> List[dict]:
    split_family = "icse26"
    common_path = (
        f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
        f"{method.name}/{args.fit_framing}/{representation.name}"
    )
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
    metadata = _metadata_payload(args, representation, method, "in-domain")

    folds = args.folds or list(range(_num_folds(Path(args.data_root), args.test_dataset)))
    per_fold = []
    for fold in folds:
        bundle_fit, fit_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "fit", args.fit_framing, fold, "fold",
        )
        _assert_representation(bundle_fit, fit_dir, representation)
        bundle_val, val_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "validate", args.val_framing, fold, "fold",
        )
        _assert_representation(bundle_val, val_dir, representation)
        bundle_test, test_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "test", args.test_framing, fold, "fold",
        )
        _assert_representation(bundle_test, test_dir, representation)

        direction, layers, val_scores_all, val_per_layer, selected_layer = _fit_and_select(
            bundle_fit, bundle_val, representation, method
        )
        test_scores_all, test_per_layer = _score_test(
            direction, layers, representation, bundle_test
        )
        sel_test = test_per_layer[selected_layer]

        result = {
            "fold": fold,
            "selected_layer": selected_layer,
            "val_pairwise_acc": val_per_layer[selected_layer]["pairwise_acc"],
            "test_one_of_4": sel_test["one_of_4"],
            "test_pairwise": sel_test["pairwise_acc"],
        }
        per_fold.append(result)
        print(f"[fold {fold}] selected_layer={selected_layer} "
              f"test_1of4={sel_test['one_of_4']:.3f} test_pairwise={sel_test['pairwise_acc']:.3f}")

        if args.results_dir:
            _persist_cell(
                Path(args.results_dir),
                common_path,
                eval_segment,
                f"fold_{fold}",
                cell_ids={"fold": fold},
                layers=layers,
                bundle_val=bundle_val,
                bundle_test=bundle_test,
                val_scores_all=val_scores_all,
                test_scores_all=test_scores_all,
                selected_layer=selected_layer,
                val_per_layer=val_per_layer,
                test_per_layer=test_per_layer,
                metadata=metadata,
            )
    return per_fold


def run_ood(args, representation, method) -> List[dict]:
    split_family = f"ood_{args.fit_source}"
    common_path = (
        f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
        f"{method.name}/{args.fit_framing}/{representation.name}"
    )
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
    metadata = _metadata_payload(args, representation, method, args.fit_source)

    inner_folds = args.inner_folds or list(range(_num_folds(Path(args.data_root), args.fit_source)))
    outer_folds = args.outer_folds or list(range(_num_folds(Path(args.data_root), args.test_dataset)))

    # Test bundles depend only on (model, test_dataset, test_framing, outer fold) —
    # reuse across all inner folds.
    test_bundles = {}
    for outer in outer_folds:
        bundle_test, test_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "test", args.test_framing, outer, "fold",
        )
        _assert_representation(bundle_test, test_dir, representation)
        test_bundles[outer] = bundle_test

    per_cell = []
    for inner in inner_folds:
        bundle_fit, fit_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "fit", args.fit_framing, inner, "inner",
        )
        _assert_representation(bundle_fit, fit_dir, representation)
        bundle_val, val_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "validate", args.val_framing, inner, "inner",
        )
        _assert_representation(bundle_val, val_dir, representation)
        direction, layers, val_scores_all, val_per_layer, selected_layer = _fit_and_select(
            bundle_fit, bundle_val, representation, method
        )

        for outer in outer_folds:
            test_scores_all, test_per_layer = _score_test(
                direction, layers, representation, test_bundles[outer]
            )
            sel_test = test_per_layer[selected_layer]

            result = {
                "outer_fold": outer,
                "inner_fold": inner,
                "selected_layer": selected_layer,
                "val_pairwise_acc": val_per_layer[selected_layer]["pairwise_acc"],
                "test_one_of_4": sel_test["one_of_4"],
                "test_pairwise": sel_test["pairwise_acc"],
            }
            per_cell.append(result)
            print(f"[outer {outer} x inner {inner}] selected_layer={selected_layer} "
                  f"test_1of4={sel_test['one_of_4']:.3f} test_pairwise={sel_test['pairwise_acc']:.3f}")

            if args.results_dir:
                _persist_cell(
                    Path(args.results_dir),
                    common_path,
                    eval_segment,
                    f"outer_{outer}_inner_{inner}",
                    cell_ids={"outer_fold": outer, "inner_fold": inner},
                    layers=layers,
                    bundle_val=bundle_val,
                    bundle_test=test_bundles[outer],
                    val_scores_all=val_scores_all,
                    test_scores_all=test_scores_all,
                    selected_layer=selected_layer,
                    val_per_layer=val_per_layer,
                    test_per_layer=test_per_layer,
                    metadata=metadata,
                )
    return per_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--representation", required=True, choices=list(REPRESENTATIONS))
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--fit-framing", required=True, choices=["function", "neutral", "concept"])
    parser.add_argument("--val-framing", required=True, choices=["neutral", "concept"],
                        help="Framing for the validate split (layer selection).")
    parser.add_argument("--test-framing", default=None, choices=["neutral", "concept"],
                        help="Framing for the test split (default: same as --val-framing).")
    parser.add_argument("--fit-source", required=True, choices=["in-domain", "mbpp", "synthetic"])
    parser.add_argument("--test-dataset", required=True, choices=["bigcodebench", "humaneval"])
    parser.add_argument("--data-root", default="dataset/icse26")
    parser.add_argument("--hidden-states-root", default="hidden_states")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="In-domain only: subset of folds (default: all).")
    parser.add_argument("--inner-folds", nargs="*", type=int, default=None,
                        help="OOD only: subset of inner (fit/val) folds (default: all).")
    parser.add_argument("--outer-folds", nargs="*", type=int, default=None,
                        help="OOD only: subset of outer (test) folds (default: all).")
    args = parser.parse_args()
    args.hidden_states_root = Path(args.hidden_states_root)
    if args.test_framing is None:
        args.test_framing = args.val_framing

    representation = REPRESENTATIONS[args.representation]
    method = METHODS[args.method]

    if args.fit_source == "in-domain":
        cells = run_in_domain(args, representation, method)
    else:
        cells = run_ood(args, representation, method)

    if cells:
        mean_1of4 = sum(c["test_one_of_4"] for c in cells) / len(cells)
        mean_pair = sum(c["test_pairwise"] for c in cells) / len(cells)
        print(f"\n[summary] n_cells={len(cells)} mean_1of4={mean_1of4:.3f} mean_pairwise={mean_pair:.3f}")


if __name__ == "__main__":
    main()
