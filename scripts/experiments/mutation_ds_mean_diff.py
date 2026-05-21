"""Mutation_DS monitoring — function-framing fit, neutral val/test.

Mirrors `scripts/experiments/icse26_function_fit.py` but points at the
Mutation_DS splits under `dataset/bigcodebench/phase2/<split_family>/`
(decompressed from the matching `.tar.zst` archives in that directory).

Per-fold protocol (same as the ICSE26 function-fit variant):
  fit:   Function framing, dispatched per record by label
           label=1  → "Write a correct implementation for the following task."
           label=0  → "Write an incorrect implementation for the following task."
         Chat templates ARE applied (with each model's default system prompt).
  val:   Neutral framing, chat template applied.
  test:  Neutral framing, chat template applied.
  rep:   response_avg
  vec:   mean_diff

Mutation_DS labels: 1 = canonical solution, 0 = mutant. With `incorrect_per_task=3`
in the default `s1_baseline` test split, the `one_of_4` metric in `_common`
applies directly.

Usage:
    python scripts/experiments/mutation_ds_mean_diff.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --split-family s1_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiments._common import (
    METHODS,
    REPRESENTATIONS,
    build_framing,
    load_model_and_tokenizer,
    run_cross_validation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/bigcodebench/phase2")
    parser.add_argument("--split-family", default="s1_baseline",
                        help="Subdirectory under <data-root> (default: s1_baseline).")
    parser.add_argument("--splits-dir", default=None,
                        help="Override splits dir. Default: <data-root>/<split-family>.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--representation", default="response_avg",
                        choices=list(REPRESENTATIONS))
    parser.add_argument("--method", default="mean_diff", choices=list(METHODS))
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="Subset of folds (default: all).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="float16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vectors-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)

    rep = REPRESENTATIONS[args.representation]
    method = METHODS[args.method]

    fit_pos = build_framing("function_pos", args.model)
    fit_neg = build_framing("function_neg", args.model)
    def fit_framing_for(record):
        return fit_pos if record["label"] == 1 else fit_neg

    eval_framing = build_framing("neutral", args.model)

    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(args.data_root) / args.split_family
    manifest = json.loads((splits_dir / "manifest.json").read_text())
    num_folds = manifest["num_folds"]
    fold_indices = args.folds if args.folds else list(range(num_folds))

    run_cross_validation(
        splits_dir=splits_dir,
        fold_indices=fold_indices,
        framing_for_fit=fit_framing_for,
        framing_for_val=lambda r: eval_framing,
        framing_for_test=lambda r: eval_framing,
        representation=rep,
        method=method,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fit_label_for_log="function±",
        val_label_for_log="neutral",
        test_label_for_log="neutral",
        vectors_dir=args.vectors_dir,
        results_dir=args.results_dir,
        model_name=args.model,
        dataset="bigcodebench",
        split_family=args.split_family,
        distribution="id",
        method_label=args.method,
        fit_framing_label="function",
        val_framing_label="neutral",
        test_framing_label="neutral",
        chat_template=True,
        dtype=args.dtype,
        max_length_meta=args.max_length,
        experiment_script="scripts/experiments/mutation_ds_mean_diff.py",
    )


if __name__ == "__main__":
    main()
