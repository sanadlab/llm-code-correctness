"""ICSE26 paper replication experiment.

10-fold cross-validation reproducing the paper's LAT monitoring numbers
on bigcodebench / humaneval:

    fit:  Raw framing (paper-exact stimulus, no chat template)
    val:  Concept framing (paper-exact, raw role markers, trailing-space stem)
    test: Concept framing (same)
    rep:  sequence_last
    vec:  PCA (one signed diff per task, aligned_shuffle-replicated)

Usage:
    python scripts/experiments/icse26_replication.py \\
        --dataset bigcodebench \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --framing-fit raw --framing-val concept --framing-test concept \\
        --representation sequence_last --method pca
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
                        help="Override splits dir. Default: <data-root>/<dataset>/<split-family>.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--framing-fit", default="raw")
    parser.add_argument("--framing-val", default="concept")
    parser.add_argument("--framing-test", default="concept")
    parser.add_argument("--representation", default="sequence_last",
                        choices=list(REPRESENTATIONS))
    parser.add_argument("--method", default="pca", choices=list(METHODS))
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="Subset of folds (default: all).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", default="float16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vectors-dir", default=None,
                        help="If set, persist directions under this root.")
    parser.add_argument("--results-dir", default=None,
                        help="If set, persist detailed/results under this root.")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)

    rep = REPRESENTATIONS[args.representation]
    method = METHODS[args.method]

    fit_f = build_framing(args.framing_fit, args.model)
    val_f = build_framing(args.framing_val, args.model)
    test_f = build_framing(args.framing_test, args.model)

    splits_dir = Path(args.splits_dir) if args.splits_dir else Path(args.data_root) / args.dataset
    manifest = json.loads((splits_dir / "manifest.json").read_text())
    num_folds = manifest["num_folds"]
    fold_indices = args.folds if args.folds else list(range(num_folds))

    run_cross_validation(
        splits_dir=splits_dir,
        fold_indices=fold_indices,
        framing_for_fit=lambda r: fit_f,
        framing_for_val=lambda r: val_f,
        framing_for_test=lambda r: test_f,
        representation=rep,
        method=method,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fit_label_for_log=args.framing_fit,
        val_label_for_log=args.framing_val,
        test_label_for_log=args.framing_test,
        vectors_dir=args.vectors_dir,
        results_dir=args.results_dir,
        model_name=args.model,
        dataset=args.dataset,
        split_family=args.split_family,
        distribution="id",
        method_label=args.method,
        fit_framing_label=args.framing_fit,
        val_framing_label=args.framing_val,
        test_framing_label=args.framing_test,
        chat_template=False,
        dtype=args.dtype,
        max_length_meta=args.max_length,
        experiment_script="scripts/experiments/icse26_replication.py",
    )


if __name__ == "__main__":
    main()
