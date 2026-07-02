#!/usr/bin/env python3
"""Stratified sample of canonical + mutant records, allowing no-confound mutants.

Each record contains one task canonical and one fail mutant. If the selected
mutant has validated confounded variants, the first variant is included;
otherwise the mutant is kept with an empty confounded_variants list.

Outputs:
  - frozen-format JSONL (N tasks, one mutant each)
  - optional flat JSONL (canonical + mutant, plus confound when present)
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import Counter
from pathlib import Path

from scripts.sample_frozen import (
    _task_sort_key,
    allocate_counts,
    reference_operator_counts,
    sample_task_diverse,
)
from src.phase1.io_utils import read_jsonl, write_jsonl


def mutant_candidates(frozen_path: Path) -> list[dict]:
    """Fail mutants, whether or not they have a validated confound."""
    pool: list[dict] = []
    for task in read_jsonl(frozen_path):
        task_id = task["task_id"]
        for mutant in task.get("mutants", []):
            if mutant.get("status") != "fail":
                continue
            variants = mutant.get("confounded_variants") or []
            pool.append({
                "task_id": task_id,
                "mutant_id": mutant["mutant_id"],
                "operator_code": mutant.get("operator_code"),
                "has_confound": bool(variants),
                "task": task,
                "mutant": mutant,
                "variant": variants[0] if variants else None,
            })
    return pool


def stratified_mutant_sample(
    pool: list[dict],
    n: int,
    reference_root: Path,
    seed: int,
    operator_reference: str,
) -> list[dict]:
    ref_ops = reference_operator_counts(reference_root, operator_reference)
    op_targets = allocate_counts(n, ref_ops)

    rng = random.Random(seed)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    task_use: Counter = Counter()

    for op, op_n in op_targets.items():
        if op_n <= 0:
            continue
        candidates = [
            row for row in pool
            if row["mutant_id"] not in selected_ids
            and row.get("operator_code") == op
        ]
        chosen = sample_task_diverse(rng, candidates, op_n, task_use)
        selected.extend(chosen)
        selected_ids.update(row["mutant_id"] for row in chosen)

    if len(selected) < n:
        remaining = [row for row in pool if row["mutant_id"] not in selected_ids]
        need = n - len(selected)
        if len(remaining) < need:
            raise ValueError(
                f"Pool too small: requested {n} records, "
                f"could only collect {len(selected) + len(remaining)}"
            )
        selected.extend(sample_task_diverse(rng, remaining, need, task_use))

    selected.sort(key=lambda row: _task_sort_key(row["task_id"]))
    return selected


def entry_to_frozen_record(entry: dict) -> dict:
    task = entry["task"]
    mutant = entry["mutant"]
    variant = entry["variant"]
    variants = [copy.deepcopy(variant)] if variant else []
    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "prompt_variants": copy.deepcopy(task.get("prompt_variants", {})),
        "entry_point": task.get("entry_point", ""),
        "canonical": task.get("canonical", ""),
        "mutants": [{
            "mutant_id": mutant["mutant_id"],
            "code": mutant["code"],
            "operator": mutant.get("operator", ""),
            "operator_code": mutant.get("operator_code"),
            "site": copy.deepcopy(mutant.get("site", {})),
            "status": mutant.get("status", "fail"),
            "details": copy.deepcopy(mutant.get("details", {})),
            "failure_fingerprint": mutant.get("failure_fingerprint"),
            "confounded_variants": variants,
        }],
    }


def _flatten_record(task: dict) -> list[dict]:
    task_id = task["task_id"]
    question = task.get("task") or task.get("instruct_prompt", "")
    base = {
        "task_id": task_id,
        "question": question,
        "entry_point": task.get("entry_point", ""),
        "canonical": task.get("canonical", ""),
    }
    rows = [{
        **base,
        "sample_id": f"{task_id}__canonical",
        "kind": "canonical",
        "label": 1,
        "code": task.get("canonical", ""),
        "mutant_id": None,
        "variant_id": None,
        "operator_code": None,
        "status": "pass",
    }]

    mutant = task["mutants"][0]
    mid = mutant["mutant_id"]
    op = mutant.get("operator_code")
    rows.append({
        **base,
        "sample_id": mid,
        "kind": "mutant",
        "label": 0,
        "code": mutant["code"],
        "mutant_id": mid,
        "variant_id": None,
        "operator_code": op,
        "status": mutant.get("status", "fail"),
    })

    variants = mutant.get("confounded_variants") or []
    if variants:
        variant = variants[0]
        rows.append({
            **base,
            "sample_id": variant["variant_id"],
            "kind": "confound",
            "label": 0,
            "code": variant["code"],
            "mutant_id": mid,
            "variant_id": variant["variant_id"],
            "operator_code": op,
            "status": variant.get("status", "fail"),
        })
    return rows


def records_to_flat(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        rows.extend(_flatten_record(record))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flat-output", type=Path, default=None)
    parser.add_argument("--n", type=int, default=270)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--operator-reference",
        choices=("mutant", "confound"),
        default="mutant",
        help="Operator distribution to match. Default uses all killed mutants.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("dataset/isolated_bigcodebench"),
        help="Reference Phase 1 output root for operator proportions",
    )
    args = parser.parse_args()

    pool = mutant_candidates(args.frozen)
    chosen = stratified_mutant_sample(
        pool,
        args.n,
        args.reference_root,
        args.seed,
        args.operator_reference,
    )
    records = [entry_to_frozen_record(entry) for entry in chosen]

    write_jsonl(args.output, records)
    if args.flat_output:
        write_jsonl(args.flat_output, records_to_flat(records))

    ref_ops = reference_operator_counts(args.reference_root, args.operator_reference)
    got_ops = Counter(entry["operator_code"] for entry in chosen)
    has_confound = sum(1 for entry in chosen if entry["has_confound"])
    task_ids = {entry["task_id"] for entry in chosen}
    flat_rows = sum(2 + int(entry["has_confound"]) for entry in chosen)

    print(f"Sampled {len(records)} records -> {args.output}")
    if args.flat_output:
        print(f"  flat rows: {flat_rows} -> {args.flat_output}")
    print(f"Pool size (fail mutants): {len(pool)}")
    print(f"Unique tasks: {len(task_ids)}")
    print(f"With confound: {has_confound}")
    print(f"Without confound: {len(chosen) - has_confound}")
    print(f"Operator reference: {args.operator_reference}")
    print(f"Operator counts (top 8): {dict(got_ops.most_common(8))}")
    for op in list(dict(ref_ops.most_common(5)).keys()):
        ref_pct = 100 * ref_ops[op] / sum(ref_ops.values())
        got_pct = 100 * got_ops[op] / len(chosen) if chosen else 0
        print(f"  {op:4s} ref {ref_pct:5.1f}%  sample {got_pct:5.1f}%")


if __name__ == "__main__":
    main()
