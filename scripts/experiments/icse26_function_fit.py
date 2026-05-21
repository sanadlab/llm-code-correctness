"""ICSE26 monitoring — function-framing fit, neutral val/test.

Per-fold protocol:
  fit:   Function framing, dispatched per record by label
           label=1  → "Write a correct implementation for the following task."
           label=0  → "Write an incorrect implementation for the following task."
         Chat templates ARE applied (with each model's default system prompt).
  val:   Neutral framing, chat template applied (same as above). No
         label-suggestive wording, so the monitor is evaluated under a
         clean stimulus.
  test:  Neutral framing, chat template applied.
  rep:   response_avg
  vec:   mean_diff

The chat-template path in `extract_bundle` uses `apply_chat_template` with
`add_generation_prompt=True`, then appends the assistant turn verbatim.
For Qwen/OpenCoder this includes the model's default system prompt; for
CodeLlama/Mistral the chat template does not inject a system message.

Usage:
    python scripts/experiments/icse26_function_fit.py \\
        --dataset bigcodebench \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct
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
    parser.add_argument("--dataset", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument("--data-root", default="dataset/icse26")
    parser.add_argument("--split-family", default="icse26")
    parser.add_argument("--splits-dir", default=None,
                        help="Override splits dir.")
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

    # Fit dispatcher: positive instruction for correct samples, negative
    # for incorrect ones. The framings themselves carry `use_chat_template=True`
    # by default, so `extract_bundle` will route through `apply_chat_template`.
    fit_pos = build_framing("function_pos", args.model)
    fit_neg = build_framing("function_neg", args.model)
    def fit_framing_for(record):
        return fit_pos if record["label"] == 1 else fit_neg

    # Val/test under a clean (no-instruction) stimulus, also chat-templated.
    eval_framing = build_framing("neutral", args.model)

    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(args.data_root) / args.dataset
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
        test_label_for_log="neutral",        vectors_dir=args.vectors_dir,
        results_dir=args.results_dir,
        model_name=args.model,
        dataset=args.dataset,
        split_family=args.split_family,
        distribution="id",
        method_label=args.method,
        fit_framing_label="function",
        val_framing_label="neutral",
        test_framing_label="neutral",
        chat_template=True,
        dtype=args.dtype,
        max_length_meta=args.max_length,
        experiment_script="scripts/experiments/icse26_function_fit.py",    )


if __name__ == "__main__":
    main()
