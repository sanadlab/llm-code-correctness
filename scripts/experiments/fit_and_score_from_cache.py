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
from typing import Any, Dict, List

from scripts.experiments._common import (
    METHODS,
    REPRESENTATIONS,
    one_of_n_accuracy,
    pairwise_accuracy,
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
    val_per_layer: Dict[int, float] = {}
    for j, layer in enumerate(layers):
        val_per_layer[layer] = pairwise_accuracy(
            val_scores_all[:, j], bundle_val.labels, bundle_val.task_ids
        )
    # Python's max() returns the first item on ties; iterating from the
    # deepest layer down replicates the paper's tie-break (see
    # _common.run_cross_validation for the full explanation).
    ordered = sorted(val_per_layer.keys(), reverse=True)
    selected_layer = max(ordered, key=lambda l: val_per_layer[l])
    return direction, layers, val_scores_all, val_per_layer, selected_layer


def _score_test(direction, layers, representation, bundle_test: HiddenStateBundle):
    acts_test = representation.select(bundle_test)
    _, test_scores_all = project_all_layers(direction, acts_test)
    test_per_layer: Dict[int, Dict[str, float]] = {}
    for j, layer in enumerate(layers):
        col = test_scores_all[:, j]
        test_per_layer[layer] = {
            "one_of_4": one_of_n_accuracy(col, bundle_test.labels, bundle_test.task_ids),
            "pairwise_acc": pairwise_accuracy(col, bundle_test.labels, bundle_test.task_ids),
        }
    return test_scores_all, test_per_layer


def _load_bundle_metadata(bundle_dir: Path) -> Dict[str, Any]:
    meta_path = bundle_dir / "metadata.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}


def _cell_dir(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_name: str,
) -> Path:
    return results_dir / common_path / eval_segment / cell_name


def _combo_is_complete(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_names: List[str],
    *,
    ood: bool,
) -> bool:
    required = ("results.json", "metadata.json")
    if not ood:
        required = (*required, "detailed.json")
    return all(
        all((_cell_dir(results_dir, common_path, eval_segment, name) / fname).is_file()
            for fname in required)
        for name in cell_names
    )


def _write_in_domain_cell(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_name: str,
    *,
    fold: int,
    layers: List[int],
    val_scores_all,
    test_scores_all,
    bundle_val: HiddenStateBundle,
    bundle_test: HiddenStateBundle,
    val_per_layer: Dict[int, float],
    test_per_layer: Dict[int, Dict[str, float]],
    selected_layer: int,
    sel_test: Dict[str, float],
    best_layer: int,
    meta: Dict[str, Any],
) -> None:
    res_dir = _cell_dir(results_dir, common_path, eval_segment, cell_name)
    res_dir.mkdir(parents=True, exist_ok=True)

    detailed = {
        "fold": fold,
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

    results = {
        "fold": fold,
        "selected_layer": selected_layer,
        "val_per_layer": {str(k): {"pairwise_acc": v} for k, v in val_per_layer.items()},
        "test_per_layer": {str(k): v for k, v in test_per_layer.items()},
        "test_at_selected_layer": dict(sel_test),
        "test_theoretical_best_layer": {
            "layer": best_layer,
            **test_per_layer[best_layer],
        },
    }
    with open(res_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(res_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def _write_ood_cell(
    results_dir: Path,
    common_path: str,
    eval_segment: str,
    cell_name: str,
    *,
    outer_fold: int,
    inner_fold: int,
    val_per_layer: Dict[int, float],
    test_per_layer: Dict[int, Dict[str, float]],
    selected_layer: int,
    sel_test: Dict[str, float],
    best_layer: int,
    meta: Dict[str, Any],
) -> None:
    res_dir = _cell_dir(results_dir, common_path, eval_segment, cell_name)
    res_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "selected_layer": selected_layer,
        "val_per_layer": {str(k): {"pairwise_acc": v} for k, v in val_per_layer.items()},
        "test_per_layer": {str(k): v for k, v in test_per_layer.items()},
        "test_at_selected_layer": dict(sel_test),
        "best_layer_on_test": best_layer,
        "test_at_best_layer": test_per_layer[best_layer],
        "val_pairwise_acc_at_selected": val_per_layer[selected_layer],
    }
    with open(res_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(res_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_in_domain(args, representation, method) -> List[dict]:
    split_family = "icse26"
    common_path = (
        f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
        f"{method.name}/{args.fit_framing}/{representation.name}"
    )
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
    git_sha = git_commit()
    experiment_script = "scripts/experiments/fit_and_score_from_cache.py"

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
        best_layer = max(test_per_layer, key=lambda k: test_per_layer[k]["one_of_4"])

        result = {
            "fold": fold,
            "selected_layer": selected_layer,
            "val_pairwise_acc": val_per_layer[selected_layer],
            "test_one_of_4": sel_test["one_of_4"],
            "test_pairwise": sel_test["pairwise_acc"],
        }
        per_fold.append(result)
        print(f"[fold {fold}] selected_layer={selected_layer} "
              f"test_1of4={sel_test['one_of_4']:.3f} test_pairwise={sel_test['pairwise_acc']:.3f}")

        if args.results_dir:
            _write_in_domain_cell(
                Path(args.results_dir), common_path, eval_segment, f"fold_{fold}",
                fold=fold,
                layers=layers,
                val_scores_all=val_scores_all,
                test_scores_all=test_scores_all,
                bundle_val=bundle_val,
                bundle_test=bundle_test,
                val_per_layer=val_per_layer,
                test_per_layer=test_per_layer,
                selected_layer=selected_layer,
                sel_test=sel_test,
                best_layer=best_layer,
                meta={
                    "vector_path": None,
                    "model": args.model,
                    "dataset": args.test_dataset,
                    "split_family": split_family,
                    "distribution": "id",
                    "fold": fold,
                    "method": method.name,
                    "representation": representation.name,
                    "fit_framing": args.fit_framing,
                    "val_framing": args.val_framing,
                    "test_framing": args.test_framing,
                    "n_val": bundle_val.num_samples,
                    "n_test": bundle_test.num_samples,
                    "experiment_script": experiment_script,
                    "git_commit": git_sha,
                },
            )
    return per_fold


def run_ood(args, representation, method) -> List[dict]:
    split_family = f"ood_{args.fit_source}"
    common_path = (
        f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
        f"{method.name}/{args.fit_framing}/{representation.name}"
    )
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
    git_sha = git_commit()
    experiment_script = "scripts/experiments/fit_and_score_from_cache.py"

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
        bundle_meta = _load_bundle_metadata(fit_dir)

        direction, layers, _val_scores_all, val_per_layer, selected_layer = _fit_and_select(
            bundle_fit, bundle_val, representation, method
        )

        for outer in outer_folds:
            _test_scores_all, test_per_layer = _score_test(
                direction, layers, representation, test_bundles[outer]
            )
            sel_test = test_per_layer[selected_layer]
            best_layer = max(test_per_layer, key=lambda k: test_per_layer[k]["one_of_4"])

            result = {
                "outer_fold": outer,
                "inner_fold": inner,
                "selected_layer": selected_layer,
                "val_pairwise_acc": val_per_layer[selected_layer],
                "test_one_of_4": sel_test["one_of_4"],
                "test_pairwise": sel_test["pairwise_acc"],
            }
            per_cell.append(result)
            print(f"[outer {outer} x inner {inner}] selected_layer={selected_layer} "
                  f"test_1of4={sel_test['one_of_4']:.3f} test_pairwise={sel_test['pairwise_acc']:.3f}")

            if args.results_dir:
                _write_ood_cell(
                    Path(args.results_dir), common_path, eval_segment,
                    f"outer_{outer}_inner_{inner}",
                    outer_fold=outer,
                    inner_fold=inner,
                    val_per_layer=val_per_layer,
                    test_per_layer=test_per_layer,
                    selected_layer=selected_layer,
                    sel_test=sel_test,
                    best_layer=best_layer,
                    meta={
                        "model": args.model,
                        "ood_source": args.fit_source,
                        "test_dataset": args.test_dataset,
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "method": method.name,
                        "representation": representation.name,
                        "fit_framing": args.fit_framing,
                        "val_framing": args.val_framing,
                        "test_framing": args.test_framing,
                        "chat_template": bundle_meta.get("chat_template"),
                        "dtype": bundle_meta.get("dtype"),
                        "max_length": bundle_meta.get("max_length"),
                        "experiment_script": experiment_script,
                        "git_commit": git_sha,
                    },
                )
    return per_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--representation", required=True, choices=list(REPRESENTATIONS))
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--fit-framing", required=True, choices=["function", "neutral", "concept"],
                        help="Must match ABLATION_FIT_FRAMINGS in scripts/slurm/ablation_grid.sh.")
    parser.add_argument("--val-framing", required=True, choices=["neutral", "concept"],
                        help="Must match ABLATION_VAL_FRAMINGS in scripts/slurm/ablation_grid.sh.")
    parser.add_argument("--test-framing", default=None, choices=["neutral", "concept"],
                        help="Must match ABLATION_TEST_FRAMINGS in scripts/slurm/ablation_grid.sh.")
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
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if all expected result artifacts already exist.")
    args = parser.parse_args()
    args.hidden_states_root = Path(args.hidden_states_root)
    if args.test_framing is None:
        args.test_framing = args.val_framing

    representation = REPRESENTATIONS[args.representation]
    method = METHODS[args.method]

    if args.skip_existing and args.results_dir:
        data_root = Path(args.data_root)
        if args.fit_source == "in-domain":
            split_family = "icse26"
            common_path = (
                f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
                f"{method.name}/{args.fit_framing}/{representation.name}"
            )
            eval_segment = f"val={args.val_framing}__test={args.test_framing}"
            folds = args.folds or list(range(_num_folds(data_root, args.test_dataset)))
            cell_names = [f"fold_{fold}" for fold in folds]
        else:
            split_family = f"ood_{args.fit_source}"
            common_path = (
                f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
                f"{method.name}/{args.fit_framing}/{representation.name}"
            )
            eval_segment = f"val={args.val_framing}__test={args.test_framing}"
            inner_folds = args.inner_folds or list(range(_num_folds(data_root, args.fit_source)))
            outer_folds = args.outer_folds or list(range(_num_folds(data_root, args.test_dataset)))
            cell_names = [
                f"outer_{outer}_inner_{inner}"
                for inner in inner_folds
                for outer in outer_folds
            ]
        if _combo_is_complete(
            Path(args.results_dir), common_path, eval_segment, cell_names,
            ood=args.fit_source != "in-domain",
        ):
            print(f"[skip] results already complete for {common_path}/{eval_segment}")
            return

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