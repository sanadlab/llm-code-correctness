"""ICSE26 monitoring — OOD nested cross-validation.

Fit + validate on an OOD source (synthetic or mbpp; 4 inner folds);
test on bigcodebench or humaneval (10 outer folds). For each outer
fold, we pair its test set with each of the 4 inner directions
(layer selected per-inner on inner-val), giving 40 (outer, inner)
result cells per (model, test_dataset, ood_source, protocol).

Two protocols:
  --protocol pca         : raw fit, concept val/test, sequence_last, PCA
  --protocol function    : function± fit (label-dispatched), neutral
                           val/test, response_avg, mean_diff

Persistence layout:
  <results_dir>/<model_safe>/ood_<ood_source>/<test_dataset>/id/
      <method>/<fit_framing>/<rep>/val=<vf>__test=<tf>/
      outer_O_inner_I/results.json
  <vectors_dir>/<model_safe>/ood_<ood_source>/<test_dataset>/id/
      <method>/<fit_framing>/<rep>/inner_I/direction.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import torch

from scripts.experiments._common import (
    METHODS,
    PAPER_TAGS,
    REPRESENTATIONS,
    build_framing,
    extract_bundle,
    load_model_and_tokenizer,
    load_split,
    one_of_n_accuracy,
    pairwise_accuracy,
    _project_all_layers,
    _safe_model_dir,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dataset", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument("--ood-source", choices=["synthetic", "mbpp"], required=True)
    parser.add_argument("--data-root", default="dataset/icse26")
    parser.add_argument("--test-split-family", default="icse26",
                        help="Subdir under <data-root>/<test-dataset>/ containing fold_*/test.jsonl.")
    parser.add_argument("--ood-split-family", default="icse26",
                        help="Subdir under <data-root>/<ood-source>/ containing fold_*/fit.jsonl & validate.jsonl.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", choices=["pca", "function"], required=True)
    parser.add_argument("--outer-folds", nargs="*", type=int, default=None,
                        help="Subset of outer (test) folds. Default: all from manifest.")
    parser.add_argument("--inner-folds", nargs="*", type=int, default=None,
                        help="Subset of inner (fit/val) folds. Default: all from manifest.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", default="float16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vectors-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    # ---- Resolve protocol -> framings, rep, method ----
    if args.protocol == "pca":
        rep = REPRESENTATIONS["sequence_last"]
        method = METHODS["pca"]
        fit_f = build_framing("raw", args.model)
        val_f = build_framing("concept", args.model)
        test_f = build_framing("concept", args.model)
        fit_dispatcher = lambda r: fit_f
        fit_framing_label = "raw"
        val_framing_label = "concept"
        test_framing_label = "concept"
        method_label = "pca"
        chat_template = False
    else:  # function
        rep = REPRESENTATIONS["response_avg"]
        method = METHODS["mean_diff"]
        fit_pos = build_framing("function_pos", args.model)
        fit_neg = build_framing("function_neg", args.model)
        fit_dispatcher = lambda r: fit_pos if r["label"] == 1 else fit_neg
        eval_f = build_framing("neutral", args.model)
        val_f = test_f = eval_f
        fit_framing_label = "function"
        val_framing_label = "neutral"
        test_framing_label = "neutral"
        method_label = "mean_diff"
        chat_template = True

    # ---- Resolve fold lists from manifests ----
    test_splits_dir = Path(args.data_root) / args.test_dataset
    ood_splits_dir = Path(args.data_root) / args.ood_source
    test_manifest = json.loads((test_splits_dir / "manifest.json").read_text())
    ood_manifest = json.loads((ood_splits_dir / "manifest.json").read_text())
    outer_folds = args.outer_folds if args.outer_folds is not None else list(range(test_manifest["num_folds"]))
    inner_folds = args.inner_folds if args.inner_folds is not None else list(range(ood_manifest["num_folds"]))
    print(f"[ood] outer (test) folds: {outer_folds}")
    print(f"[ood] inner (fit/val) folds: {inner_folds}")

    # ---- Load model once ----
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)

    git_sha = _git_commit()
    safe_model = _safe_model_dir(args.model)
    common_path = (
        f"{safe_model}/ood_{args.ood_source}/{args.test_dataset}/id/"
        f"{method_label}/{fit_framing_label}/{rep.name}"
    )
    eval_segment = f"val={val_framing_label}__test={test_framing_label}"

    # ---- Pre-extract test bundles per outer fold (re-used across inner folds) ----
    print("\n[ood] pre-extracting test bundles for outer folds...")
    test_bundles: Dict[int, object] = {}
    for O in outer_folds:
        t0 = time.time()
        test_recs = load_split(test_splits_dir / f"fold_{O}" / "test.jsonl")
        bundle = extract_bundle(
            lambda r: test_f, test_recs, model, tokenizer,
            [rep], args.batch_size, args.max_length,
        )
        test_bundles[O] = bundle
        print(f"[ood] outer fold {O}: test N={len(test_recs)} ({time.time()-t0:.1f}s)")

    # ---- Inner loop: fit + select layer + project test bundles ----
    for I in inner_folds:
        print(f"\n[ood] ===== inner fold {I} =====")
        inner_dir = ood_splits_dir / f"fold_{I}"
        fit_recs = load_split(inner_dir / "fit.jsonl")
        val_recs = load_split(inner_dir / "validate.jsonl")
        print(f"[ood] inner {I}: fit N={len(fit_recs)} val N={len(val_recs)}")

        t0 = time.time()
        bundle_fit = extract_bundle(
            fit_dispatcher, fit_recs, model, tokenizer,
            [rep], args.batch_size, args.max_length,
        )
        bundle_val = extract_bundle(
            lambda r: val_f, val_recs, model, tokenizer,
            [rep], args.batch_size, args.max_length,
        )
        print(f"[ood] inner {I}: fit+val extracted ({time.time()-t0:.1f}s)")

        acts_fit = rep.select(bundle_fit)
        direction = method.fit(acts_fit, bundle_fit.labels, bundle_fit.task_ids)
        direction.representation = rep.name

        # Per-layer val pairwise.
        acts_val = rep.select(bundle_val)
        layers, val_scores_all = _project_all_layers(direction, acts_val)
        val_per_layer = {}
        for j, L in enumerate(layers):
            val_per_layer[L] = pairwise_accuracy(
                val_scores_all[:, j], bundle_val.labels, bundle_val.task_ids
            )
        ordered = sorted(val_per_layer.keys(), reverse=True)
        selected_layer = max(ordered, key=lambda l: val_per_layer[l])
        print(f"[ood] inner {I}: selected layer = {selected_layer} "
              f"(val pairwise = {val_per_layer[selected_layer]:.3f})")

        # ---- Persist direction (one per inner fold) ----
        if args.vectors_dir:
            vec_dir = Path(args.vectors_dir) / common_path / f"inner_{I}"
            vec_dir.mkdir(parents=True, exist_ok=True)
            direction.save(vec_dir / "direction.pt")
            with open(vec_dir / "metadata.json", "w") as f:
                json.dump({
                    "model": args.model,
                    "ood_source": args.ood_source,
                    "test_dataset": args.test_dataset,
                    "ood_split_family": args.ood_split_family,
                    "test_split_family": args.test_split_family,
                    "inner_fold": I,
                    "method": method_label,
                    "representation": rep.name,
                    "fit_framing": fit_framing_label,
                    "chat_template": chat_template,
                    "dtype": args.dtype,
                    "max_length": args.max_length,
                    "n_fit": len(fit_recs),
                    "n_val": len(val_recs),
                    "selected_layer": selected_layer,
                    "val_per_layer": {str(k): v for k, v in val_per_layer.items()},
                    "experiment_script": "scripts/experiments/icse26_ood.py",
                    "git_commit": git_sha,
                }, f, indent=2)

        # ---- Project against each outer test bundle ----
        for O in outer_folds:
            bundle_test = test_bundles[O]
            acts_test = rep.select(bundle_test)
            _, test_scores_all = _project_all_layers(direction, acts_test)
            test_per_layer = {}
            for j, L in enumerate(layers):
                col = test_scores_all[:, j]
                test_per_layer[L] = {
                    "one_of_4": one_of_n_accuracy(col, bundle_test.labels, bundle_test.task_ids),
                    "pairwise_acc": pairwise_accuracy(col, bundle_test.labels, bundle_test.task_ids),
                }
            sel = test_per_layer[selected_layer]
            best_layer = max(test_per_layer, key=lambda k: test_per_layer[k]["one_of_4"])
            print(f"[ood] outer {O} × inner {I}: "
                  f"test 1-of-4 = {sel['one_of_4']:.3f}  pairwise = {sel['pairwise_acc']:.3f}")

            if args.results_dir:
                res_dir = (Path(args.results_dir) / common_path / eval_segment
                           / f"outer_{O}_inner_{I}")
                res_dir.mkdir(parents=True, exist_ok=True)
                results = {
                    "outer_fold": O,
                    "inner_fold": I,
                    "selected_layer": selected_layer,
                    "val_per_layer": {str(k): {"pairwise_acc": v} for k, v in val_per_layer.items()},
                    "test_per_layer": {str(k): v for k, v in test_per_layer.items()},
                    "test_at_selected_layer": sel,
                    "best_layer_on_test": best_layer,
                    "test_at_best_layer": test_per_layer[best_layer],
                    "val_pairwise_acc_at_selected": val_per_layer[selected_layer],
                }
                with open(res_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2)
                with open(res_dir / "metadata.json", "w") as f:
                    json.dump({
                        "model": args.model,
                        "ood_source": args.ood_source,
                        "test_dataset": args.test_dataset,
                        "outer_fold": O,
                        "inner_fold": I,
                        "method": method_label,
                        "representation": rep.name,
                        "fit_framing": fit_framing_label,
                        "val_framing": val_framing_label,
                        "test_framing": test_framing_label,
                        "chat_template": chat_template,
                        "dtype": args.dtype,
                        "max_length": args.max_length,
                        "experiment_script": "scripts/experiments/icse26_ood.py",
                        "git_commit": git_sha,
                    }, f, indent=2)

            # Free per-outer-projection memory.
            del test_scores_all

        del bundle_fit, bundle_val, acts_fit, acts_val, val_scores_all, direction
        torch.cuda.empty_cache()

    print("\n[ood] done.")


if __name__ == "__main__":
    main()
