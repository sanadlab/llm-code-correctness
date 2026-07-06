#!/usr/bin/env python3
"""Re-evaluate rejected confound candidates and compare to source mutant failures."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.phase1.config import load_config
from src.phase1.docker_eval import run_docker_eval
from src.phase1.validate_mutants import compute_failure_fingerprint


def _normalize_trace(trace: str) -> str:
    trace = re.sub(r'File "[^"]*"', 'File "<stripped>"', trace)
    trace = re.sub(r", line \d+,", ", line <N>,", trace)
    trace = re.sub(r"\d+\.\d+s", "<time>", trace)
    trace = re.sub(r"0x[0-9a-fA-F]+", "<addr>", trace)
    # Strip local variable renames (single-letter or short names in quotes)
    trace = re.sub(r"NameError: name '[^']+'", "NameError: name '<var>'", trace)
    return trace


def _failing_test_sets(details: dict) -> tuple[frozenset[str], dict[str, str]]:
    normed = {}
    for test_name, trace in details.items():
        normed[test_name] = _normalize_trace(str(trace))
    return frozenset(normed.keys()), normed


def _trace_overlap(a: dict[str, str], b: dict[str, str]) -> tuple[int, int]:
    """Count shared tests with identical normalized traces."""
    shared = set(a) & set(b)
    identical = sum(1 for t in shared if a[t] == b[t])
    return identical, len(shared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1/bigcodebench.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Max rejected to eval (0=all)")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--work-dir", default="")
    args = parser.parse_args()

    os.environ.setdefault("BIGCODEBENCH_TIMEOUT_PER_TASK", "360")
    os.environ.setdefault("BIGCODEBENCH_GT_TIMEOUT_MULTIPLIER", "15")

    cfg = load_config(args.config)
    parallel_override = os.environ.get("VALIDATE_PARALLEL")
    if parallel_override:
        cfg.validate.parallel = int(parallel_override)
    interm = cfg.intermediate_dir

    kept_sources = {
        json.loads(line)["source_mutant_id"]
        for line in open(interm / "validated_confounded.jsonl")
    }
    mutants = {
        m["mutant_id"]: m
        for m in map(json.loads, open(interm / "validated_mutants.jsonl"))
    }
    tasks = {t["task_id"]: t for t in map(json.loads, open(cfg.validated_tasks_path))}

    rejected = []
    for line in open(interm / "confound_candidates.jsonl"):
        c = json.loads(line)
        if c["source_mutant_id"] not in kept_sources:
            rejected.append(c)

    if args.limit:
        rejected = rejected[: args.limit]

    scratch = os.environ.get("SCRATCH", "/scratch/mki4895")
    work_dir = Path(args.work_dir) if args.work_dir else Path(scratch) / "phase1_eval" / "rejected_confound_analysis"
    work_dir.mkdir(parents=True, exist_ok=True)

    categories = Counter()
    examples: dict[str, list] = {k: [] for k in [
        "false_negative_exact_fp",
        "false_negative_same_tests_same_traces",
        "false_negative_same_tests_diff_traces",
        "false_negative_subset_tests",
        "correct_pass",
        "correct_timeout",
        "correct_different_failure",
    ]}

    print(f"Re-evaluating {len(rejected)} rejected confound candidates...")

    for batch_start in range(0, len(rejected), args.batch_size):
        batch = rejected[batch_start : batch_start + args.batch_size]
        batch_idx = batch_start // args.batch_size + 1
        print(f"  Batch {batch_idx}: {len(batch)} candidates")

        samples = []
        for c in batch:
            task = tasks[c["task_id"]]
            samples.append({
                "task_id": c["task_id"],
                "solution": task["complete_prompt"] + c["code"],
            })

        eval_results = run_docker_eval(
            samples,
            cfg,
            work_dir=work_dir,
            label=f"rejected_batch_{batch_idx:04d}",
            batch_timeout_seconds=20 * 60,
        )

        by_task: dict[str, list] = {}
        for i, c in enumerate(batch):
            by_task.setdefault(c["task_id"], []).append((i, c))

        for tid, items in by_task.items():
            results = eval_results.get(tid, [])
            for order_pos, (_, c) in enumerate(items):
                src = mutants[c["source_mutant_id"]]
                source_fp = src.get("failure_fingerprint")
                source_details = src.get("details") or {}

                status = "fail"
                details = {}
                if order_pos < len(results):
                    r = results[order_pos]
                    status = r.get("status", "fail")
                    details = r.get("details") or {}

                if status == "pass":
                    cat = "correct_pass"
                elif status == "timeout":
                    cat = "correct_timeout"
                else:
                    fp = compute_failure_fingerprint(details)
                    if fp == source_fp:
                        cat = "false_negative_exact_fp"
                    else:
                        src_tests, src_norm = _failing_test_sets(source_details)
                        var_tests, var_norm = _failing_test_sets(details)
                        identical, shared = _trace_overlap(src_norm, var_norm)
                        if src_tests == var_tests and identical == len(src_tests) and src_tests:
                            cat = "false_negative_same_tests_same_traces"
                        elif src_tests == var_tests and shared == len(src_tests) and src_tests:
                            cat = "false_negative_same_tests_diff_traces"
                        elif var_tests and var_tests <= src_tests and identical == len(var_tests):
                            cat = "false_negative_subset_tests"
                        else:
                            cat = "correct_different_failure"

                categories[cat] += 1
                if len(examples[cat]) < 3:
                    examples[cat].append({
                        "variant_id": c["variant_id"],
                        "source_mutant_id": c["source_mutant_id"],
                        "status": status,
                        "source_fp": source_fp,
                        "variant_fp": compute_failure_fingerprint(details) if details else None,
                        "source_tests": sorted(source_details.keys())[:5],
                        "variant_tests": sorted(details.keys())[:5],
                    })

    total = sum(categories.values())
    print("\n=== Rejected confound re-eval categories ===")
    for cat, n in categories.most_common():
        print(f"  {cat:45s}: {n:5d} ({100*n/total:.1f}%)")

    false_neg = (
        categories["false_negative_exact_fp"]
        + categories["false_negative_same_tests_same_traces"]
        + categories["false_negative_same_tests_diff_traces"]
        + categories["false_negative_subset_tests"]
    )
    print(f"\nPotential false negatives (same failure, filter missed): {false_neg} ({100*false_neg/total:.1f}%)")
    print(f"  exact fingerprint match: {categories['false_negative_exact_fp']}")
    print(f"  same tests + identical normalized traces: {categories['false_negative_same_tests_same_traces']}")
    print(f"  same tests + different normalized traces: {categories['false_negative_same_tests_diff_traces']}")
    print(f"  variant fails subset of source tests (identical traces): {categories['false_negative_subset_tests']}")

    for cat in examples:
        if examples[cat]:
            print(f"\n--- Examples: {cat} ---")
            for ex in examples[cat]:
                print(json.dumps(ex, indent=2))


if __name__ == "__main__":
    main()
