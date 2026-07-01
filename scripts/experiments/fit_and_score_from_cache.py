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
  paired | weighted: fit/validate on mutants by default; pass
             --val-dataset to validate on icse26 bigcodebench/humaneval instead.
             Mutant test from mutants_only_<benchmark>/weighted; icse26 test per --test-dataset.

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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments._common import (
    METHODS,
    REPRESENTATIONS,
    one_of_n_accuracy,
    pairwise_accuracy,
)
from src.hidden_states import (
    HiddenStateBundle,
    MUTANTS_BENCHMARKS,
    MUTANTS_CONFOUNDS_STRATEGIES,
    MUTANTS_STRATEGIES,
    bundle_output_dir,
    cache_path_distribution,
    git_commit,
    mutants_confounds_split_family,
    mutants_confounds_test_split_family,
    mutants_split_family,
    mutants_test_split_family,
    safe_model_dir,
)
from src.vectors.projection import project_all_layers

MUTANTS_FIT_SOURCES = MUTANTS_STRATEGIES
MUTANTS_CONFOUNDS_FIT_SOURCES = MUTANTS_CONFOUNDS_STRATEGIES
MUTANT_TEST_DATASETS = frozenset({"mutants_only", "mutants_confounds"})
ICSE26_SPLIT_FAMILY = "icse26"
ICSE26_BENCHMARKS = frozenset({"bigcodebench", "humaneval"})


def _default_mutants_data_root(benchmark: str, *, confounds: bool) -> Path:
    prefix = "mutants_confounds" if confounds else "mutants_only"
    return Path(f"dataset/{prefix}_{benchmark}")


def _num_folds(data_root: Path, dataset: str) -> int:
    manifest = json.loads((data_root / dataset / "manifest.json").read_text())
    return manifest["num_folds"]


def _mutants_num_folds(mutants_data_root: Path, strategy: str) -> int:
    manifest = json.loads((mutants_data_root / strategy / "manifest.json").read_text())
    return manifest["num_folds"]


def _load_icse26_bundle(
    hidden_states_root: Path,
    model_name: str,
    dataset: str,
    data_split: str,
    framing: str,
    fold: int,
):
    return _load_bundle(
        hidden_states_root,
        model_name,
        ICSE26_SPLIT_FAMILY,
        dataset,
        "id",
        data_split,
        framing,
        fold,
        "fold",
    )


def _load_icse26_test_bundle(
    hidden_states_root: Path,
    model_name: str,
    test_dataset: str,
    test_framing: str,
    fold: int,
):
    return _load_icse26_bundle(
        hidden_states_root,
        model_name,
        test_dataset,
        "test",
        test_framing,
        fold,
    )


def _eval_segment(
    val_framing: str,
    test_framing: str,
    *,
    val_dataset: Optional[str] = None,
) -> str:
    """Result path segment for val/test framings; val_ds= when val != fit source."""
    base = f"val={val_framing}__test={test_framing}"
    if val_dataset is not None:
        return f"val_ds={val_dataset}__{base}"
    return base


def _mutants_split_family_for_fit(
    fit_source: str, *, mutants_confounds: bool, benchmark: str
) -> str:
    if mutants_confounds:
        return mutants_confounds_split_family(fit_source, benchmark)
    return mutants_split_family(fit_source, benchmark)


def _mutants_test_split_family(test_dataset: str, *, benchmark: str) -> str:
    if test_dataset == "mutants_confounds":
        return mutants_confounds_test_split_family(benchmark)
    return mutants_test_split_family(benchmark)


def _mutants_common_path(
    model: str,
    fit_source: str,
    test_dataset: str,
    method,
    fit_framing: str,
    representation,
    *,
    mutants_confounds: bool,
    benchmark: str,
) -> str:
    split_family = _mutants_split_family_for_fit(
        fit_source, mutants_confounds=mutants_confounds, benchmark=benchmark,
    )
    return (
        f"{safe_model_dir(model)}/{split_family}/{test_dataset}/id/"
        f"{method.name}/{fit_framing}/{representation.name}"
    )


def _load_mutants_val_bundle(
    args,
    fold: int,
    representation,
) -> tuple[HiddenStateBundle, Path]:
    if args.val_dataset is None:
        split_family = _mutants_split_family_for_fit(
            args.fit_source,
            mutants_confounds=args.mutants_confounds,
            benchmark=args.mutants_benchmark,
        )
        bundle_val, val_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, None,
            cache_path_distribution(split_family),
            "validate", args.val_framing, fold, "fold",
        )
    else:
        bundle_val, val_dir = _load_icse26_bundle(
            args.hidden_states_root,
            args.model,
            args.val_dataset,
            "validate",
            args.val_framing,
            fold,
        )
    _assert_representation(bundle_val, val_dir, representation)
    return bundle_val, val_dir


def _load_bundle(
    hidden_states_root: Path,
    model_name: str,
    split_family: str,
    benchmark: Optional[str],
    distribution: Optional[str],
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
<<<<<<< HEAD
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
=======
    eval_segment = _eval_segment(args.val_framing, args.test_framing)
    git_sha = git_commit()
    experiment_script = "scripts/experiments/fit_and_score_from_cache.py"
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815

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
<<<<<<< HEAD
                    "git_commit": git_commit(),
                    "experiment_script": "scripts/experiments/fit_and_score_from_cache.py",
=======
                    "n_val": bundle_val.num_samples,
                    "n_test": bundle_test.num_samples,
                    "experiment_script": experiment_script,
                    "git_commit": git_sha,
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
                },
            )
    return per_fold


def run_ood(args, representation, method) -> List[dict]:
    split_family = f"ood_{args.fit_source}"
    common_path = (
        f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
        f"{method.name}/{args.fit_framing}/{representation.name}"
    )
<<<<<<< HEAD
    eval_segment = f"val={args.val_framing}__test={args.test_framing}"
=======
    eval_segment = _eval_segment(args.val_framing, args.test_framing)
    git_sha = git_commit()
    experiment_script = "scripts/experiments/fit_and_score_from_cache.py"
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815

    inner_folds = args.inner_folds or list(range(_num_folds(Path(args.data_root), args.fit_source)))
    outer_folds = args.outer_folds or list(range(_num_folds(Path(args.data_root), args.test_dataset)))

    # Test bundles depend only on (model, test_dataset, test_framing, outer fold) —
    # reuse across all inner folds.
    test_bundles = {}
    for outer in outer_folds:
<<<<<<< HEAD
        bundle_test, test_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "test", args.test_framing, outer, "fold",
=======
        bundle_test, test_dir = _load_icse26_test_bundle(
            args.hidden_states_root,
            args.model,
            args.test_dataset,
            args.test_framing,
            outer,
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
        )
        _assert_representation(bundle_test, test_dir, representation)
        test_bundles[outer] = bundle_test

    per_cell = []
    for inner in inner_folds:
        bundle_fit, fit_dir = _load_bundle(
<<<<<<< HEAD
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "fit", args.fit_framing, inner, "inner",
        )
        _assert_representation(bundle_fit, fit_dir, representation)
        bundle_val, val_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, args.test_dataset,
            "id", "validate", args.val_framing, inner, "inner",
        )
        _assert_representation(bundle_val, val_dir, representation)
        direction, layers, val_per_layer, selected_layer = _fit_and_select(
=======
            args.hidden_states_root, args.model, split_family, None,
            cache_path_distribution(split_family),
            "fit", args.fit_framing, inner, "inner",
        )
        _assert_representation(bundle_fit, fit_dir, representation)
        bundle_val, val_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, None,
            cache_path_distribution(split_family),
            "validate", args.val_framing, inner, "inner",
        )
        _assert_representation(bundle_val, val_dir, representation)
        bundle_meta = _load_bundle_metadata(fit_dir)

        direction, layers, _val_scores_all, val_per_layer, selected_layer = _fit_and_select(
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
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
<<<<<<< HEAD
                        "git_commit": git_commit(),
                        "experiment_script": "scripts/experiments/fit_and_score_from_cache.py",
=======
                        "chat_template": bundle_meta.get("chat_template"),
                        "dtype": bundle_meta.get("dtype"),
                        "max_length": bundle_meta.get("max_length"),
                        "experiment_script": experiment_script,
                        "git_commit": git_sha,
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
                    },
                )
    return per_cell


def run_mutants(args, representation, method) -> List[dict]:
    """Fit on mutants; val on mutants or icse26 per --val-dataset; test per test_dataset."""
    split_family = _mutants_split_family_for_fit(
        args.fit_source, mutants_confounds=args.mutants_confounds, benchmark=args.mutants_benchmark,
    )
    common_path = _mutants_common_path(
        args.model, args.fit_source, args.test_dataset, method, args.fit_framing, representation,
        mutants_confounds=args.mutants_confounds, benchmark=args.mutants_benchmark,
    )
    eval_segment = _eval_segment(
        args.val_framing, args.test_framing, val_dataset=args.val_dataset,
    )
    git_sha = git_commit()
    experiment_script = "scripts/experiments/fit_and_score_from_cache.py"

    folds = args.folds or list(range(_mutants_num_folds(args.mutants_data_root, args.fit_source)))
    per_fold = []
    for fold in folds:
        bundle_fit, fit_dir = _load_bundle(
            args.hidden_states_root, args.model, split_family, None,
            cache_path_distribution(split_family),
            "fit", args.fit_framing, fold, "fold",
        )
        _assert_representation(bundle_fit, fit_dir, representation)
        bundle_val, val_dir = _load_mutants_val_bundle(args, fold, representation)
        if args.test_dataset in MUTANT_TEST_DATASETS:
            test_split_family = _mutants_test_split_family(
                args.test_dataset, benchmark=args.mutants_benchmark,
            )
            bundle_test, test_dir = _load_bundle(
                args.hidden_states_root, args.model, test_split_family, None,
                cache_path_distribution(test_split_family),
                "test", args.test_framing, fold, "fold",
            )
        else:
            bundle_test, test_dir = _load_icse26_test_bundle(
                args.hidden_states_root,
                args.model,
                args.test_dataset,
                args.test_framing,
                fold,
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
                    "fit_source": args.fit_source,
                    "mutants_benchmark": args.mutants_benchmark,
                    "test_split_family": (
                        _mutants_test_split_family(args.test_dataset, benchmark=args.mutants_benchmark)
                        if args.test_dataset in MUTANT_TEST_DATASETS
                        else None
                    ),
                    "test_dataset": args.test_dataset,
                    "distribution": "id",
                    "fold": fold,
                    "method": method.name,
                    "representation": representation.name,
                    "fit_framing": args.fit_framing,
                    "val_framing": args.val_framing,
                    "val_dataset": args.val_dataset,
                    "val_split_family": (
                        ICSE26_SPLIT_FAMILY if args.val_dataset is not None else split_family
                    ),
                    "test_framing": args.test_framing,
                    "n_val": bundle_val.num_samples,
                    "n_test": bundle_test.num_samples,
                    "experiment_script": experiment_script,
                    "git_commit": git_sha,
                },
            )
    return per_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--representation", required=True, choices=list(REPRESENTATIONS))
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--fit-framing", required=True, choices=["function", "neutral", "concept", "raw"],
                        help="Must match a framing cached by scripts/extract/cache_hidden_states.py. "
                             "`raw` is the tagless ICSE paper fit stimulus; use sequence_last "
                             "(not response_avg) because raw has no response slice.")
    parser.add_argument("--val-framing", required=True, choices=["neutral", "concept"],
<<<<<<< HEAD
                        help="Framing for the validate split (layer selection).")
    parser.add_argument("--test-framing", default=None, choices=["neutral", "concept"],
                        help="Framing for the test split (default: same as --val-framing).")
    parser.add_argument("--fit-source", required=True, choices=["in-domain", "mbpp", "synthetic"])
    parser.add_argument("--test-dataset", required=True, choices=["bigcodebench", "humaneval"])
=======
                        help="Must match ABLATION_VAL_FRAMINGS in scripts/slurm/ablation_grid.sh.")
    parser.add_argument("--test-framing", default=None, choices=["neutral", "concept"],
                        help="Must match ABLATION_TEST_FRAMINGS in scripts/slurm/ablation_grid.sh.")
    parser.add_argument(
        "--fit-source",
        required=True,
        choices=[
            "in-domain", "mbpp", "synthetic",
            "paired", "weighted",
        ],
    )
    parser.add_argument(
        "--test-dataset",
        required=True,
        choices=["bigcodebench", "humaneval", "mutants_only", "mutants_confounds"],
    )
    parser.add_argument(
        "--val-dataset",
        default=None,
        choices=sorted(ICSE26_BENCHMARKS),
        help="For mutants fit sources only: validate on this icse26 benchmark instead "
             "of the mutants validate split. Omit to validate on the same dataset as fit.",
    )
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
    parser.add_argument("--data-root", default="dataset/icse26")
    parser.add_argument("--mutants-data-root", default=None)
    parser.add_argument(
        "--mutants-benchmark",
        choices=sorted(MUTANTS_BENCHMARKS),
        default="bcb_based",
        help="Which mutants corpus backs the fit/validate (and mutant test) caches.",
    )
    parser.add_argument(
        "--mutants-confounds",
        action="store_true",
        help="Use mutants_confounds dataset (paired/weighted) for fit/validate caches "
             "and default --mutants-data-root to dataset/mutants_confounds_<benchmark>.",
    )
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
<<<<<<< HEAD
    if args.test_framing is None:
        args.test_framing = args.val_framing
=======
    if args.mutants_data_root is None:
        args.mutants_data_root = _default_mutants_data_root(
            args.mutants_benchmark, confounds=args.mutants_confounds,
        )
    else:
        args.mutants_data_root = Path(args.mutants_data_root)
    if args.test_framing is None:
        args.test_framing = args.val_framing

    if args.val_dataset is not None and args.fit_source not in MUTANTS_FIT_SOURCES:
        parser.error(
            f"--val-dataset is only supported for mutants fit sources "
            f"({', '.join(sorted(MUTANTS_FIT_SOURCES))}), got fit_source={args.fit_source!r}"
        )

    if args.mutants_confounds:
        if args.fit_source not in MUTANTS_CONFOUNDS_FIT_SOURCES:
            parser.error(
                f"--mutants-confounds requires a confounds fit source "
                f"({', '.join(sorted(MUTANTS_CONFOUNDS_FIT_SOURCES))}), got {args.fit_source!r}"
            )
        if args.test_dataset == "mutants_only":
            parser.error(
                "--mutants-confounds is incompatible with --test-dataset mutants_only; "
                "use --test-dataset mutants_confounds"
            )

    if args.test_dataset == "mutants_only" and args.fit_source not in MUTANTS_FIT_SOURCES:
        parser.error(
            f"--test-dataset mutants_only requires a mutants fit source "
            f"({', '.join(sorted(MUTANTS_FIT_SOURCES))}), got {args.fit_source!r}"
        )

    if args.test_dataset == "mutants_confounds":
        if args.fit_source not in MUTANTS_CONFOUNDS_FIT_SOURCES:
            parser.error(
                f"--test-dataset mutants_confounds requires a confounds fit source "
                f"({', '.join(sorted(MUTANTS_CONFOUNDS_FIT_SOURCES))}), got {args.fit_source!r}"
            )
        if not args.mutants_confounds:
            parser.error(
                "--test-dataset mutants_confounds requires --mutants-confounds"
            )
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815

    representation = REPRESENTATIONS[args.representation]
    method = METHODS[args.method]

    if args.fit_framing == "raw" and args.representation == "response_avg":
        parser.error(
            "fit-framing raw has no response slice; use --representation sequence_last"
        )

    if args.skip_existing and args.results_dir:
        data_root = Path(args.data_root)
        if args.fit_source == "in-domain":
            split_family = ICSE26_SPLIT_FAMILY
            common_path = (
                f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
                f"{method.name}/{args.fit_framing}/{representation.name}"
            )
            eval_segment = _eval_segment(args.val_framing, args.test_framing)
            folds = args.folds or list(range(_num_folds(data_root, args.test_dataset)))
            cell_names = [f"fold_{fold}" for fold in folds]
            ood = False
        elif args.fit_source in MUTANTS_FIT_SOURCES:
            common_path = _mutants_common_path(
                args.model, args.fit_source, args.test_dataset,
                method, args.fit_framing, representation,
                mutants_confounds=args.mutants_confounds, benchmark=args.mutants_benchmark,
            )
            eval_segment = _eval_segment(
                args.val_framing, args.test_framing, val_dataset=args.val_dataset,
            )
            folds = args.folds or list(range(_mutants_num_folds(args.mutants_data_root, args.fit_source)))
            cell_names = [f"fold_{fold}" for fold in folds]
            ood = False
        else:
            split_family = f"ood_{args.fit_source}"
            common_path = (
                f"{safe_model_dir(args.model)}/{split_family}/{args.test_dataset}/id/"
                f"{method.name}/{args.fit_framing}/{representation.name}"
            )
            eval_segment = _eval_segment(args.val_framing, args.test_framing)
            inner_folds = args.inner_folds or list(range(_num_folds(data_root, args.fit_source)))
            outer_folds = args.outer_folds or list(range(_num_folds(data_root, args.test_dataset)))
            cell_names = [
                f"outer_{outer}_inner_{inner}"
                for inner in inner_folds
                for outer in outer_folds
            ]
            ood = True
        if _combo_is_complete(
            Path(args.results_dir), common_path, eval_segment, cell_names,
            ood=ood,
        ):
            print(f"[skip] results already complete for {common_path}/{eval_segment}")
            return

    if args.fit_source == "in-domain":
        cells = run_in_domain(args, representation, method)
    elif args.fit_source in MUTANTS_FIT_SOURCES:
        cells = run_mutants(args, representation, method)
    else:
        cells = run_ood(args, representation, method)

    if cells:
        mean_1of4 = sum(c["test_one_of_4"] for c in cells) / len(cells)
        mean_pair = sum(c["test_pairwise"] for c in cells) / len(cells)
        print(f"\n[summary] n_cells={len(cells)} mean_1of4={mean_1of4:.3f} mean_pairwise={mean_pair:.3f}")


if __name__ == "__main__":
    main()