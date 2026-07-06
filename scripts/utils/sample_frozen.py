#!/usr/bin/env python3
"""Stratified random sample from a frozen dataset.

Flattens frozen.jsonl into canonical / mutant / confound rows and samples
N rows while preserving:
  - kind proportions (canonical / mutant / confound) from a reference dataset
  - fail vs timeout proportions among mutants
  - mutation-operator proportions within each kind where applicable
  - task diversity (prefer distinct tasks before reusing a task)
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.phase1.io_utils import read_jsonl, write_jsonl


def _task_sort_key(task_id: str) -> tuple:
    parts = task_id.rsplit("/", 1)
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    return (task_id, 0)


def flatten_frozen(frozen_path: Path) -> list[dict]:
    rows: list[dict] = []
    for task in read_jsonl(frozen_path):
        task_id = task["task_id"]
        question = task.get("task") or task.get("instruct_prompt", "")
        base = {
            "task_id": task_id,
            "question": question,
            "entry_point": task.get("entry_point", ""),
            "canonical": task.get("canonical", ""),
        }
        rows.append({
            **base,
            "sample_id": f"{task_id}__canonical",
            "kind": "canonical",
            "label": 1,
            "code": task.get("canonical", ""),
            "mutant_id": None,
            "variant_id": None,
            "operator_code": None,
            "status": "pass",
        })
        for mutant in task.get("mutants", []):
            mid = mutant["mutant_id"]
            op = mutant.get("operator_code")
            status = mutant.get("status", "fail")
            rows.append({
                **base,
                "sample_id": mid,
                "kind": "mutant",
                "label": 0,
                "code": mutant["code"],
                "mutant_id": mid,
                "variant_id": None,
                "operator_code": op,
                "status": status,
            })
            for variant in mutant.get("confounded_variants") or []:
                vid = variant["variant_id"]
                rows.append({
                    **base,
                    "sample_id": vid,
                    "kind": "confound",
                    "label": 0,
                    "code": variant["code"],
                    "mutant_id": mid,
                    "variant_id": vid,
                    "operator_code": op,
                    "status": variant.get("status", "fail"),
                })
    return rows


def reference_kind_counts(reference_root: Path) -> Counter:
    intermediate = reference_root / "intermediate"
    validated_mutants = read_jsonl(intermediate / "validated_mutants.jsonl")
    validated_confounded = read_jsonl(intermediate / "validated_confounded.jsonl")
    validated_tasks = read_jsonl(intermediate / "validated_tasks.jsonl")

    killed_task_ids = {
        m["task_id"] for m in validated_mutants if m.get("status") != "pass"
    }
    counts = Counter()
    counts["canonical"] = sum(1 for t in validated_tasks if t["task_id"] in killed_task_ids)
    counts["mutant"] = sum(1 for m in validated_mutants if m.get("status") != "pass")
    counts["confound"] = len(validated_confounded)
    return counts


def reference_mutant_status_counts(reference_root: Path) -> Counter:
    intermediate = reference_root / "intermediate"
    validated_mutants = read_jsonl(intermediate / "validated_mutants.jsonl")
    killed = [m for m in validated_mutants if m.get("status") != "pass"]
    return Counter(m.get("status", "fail") for m in killed)


def reference_operator_counts(reference_root: Path, kind: str) -> Counter:
    intermediate = reference_root / "intermediate"
    validated_mutants = read_jsonl(intermediate / "validated_mutants.jsonl")
    mutants_by_id = {m["mutant_id"]: m for m in validated_mutants}

    if kind == "mutant":
        killed = [m for m in validated_mutants if m.get("status") != "pass"]
        return Counter(m.get("operator_code") for m in killed)

    validated_confounded = read_jsonl(intermediate / "validated_confounded.jsonl")
    counts = Counter()
    for row in validated_confounded:
        source = mutants_by_id.get(row["source_mutant_id"])
        if source:
            counts[source.get("operator_code")] += 1
    return counts


def allocate_counts(total: int, proportions: Counter) -> dict[str, int]:
    """Largest-remainder allocation of integer counts summing to total."""
    base_total = sum(proportions.values())
    if base_total == 0:
        raise ValueError("empty reference proportions")

    raw = {k: total * proportions[k] / base_total for k in proportions}
    floors = {k: int(raw[k]) for k in proportions}
    remainder = total - sum(floors.values())
    ranked = sorted(
        proportions.keys(),
        key=lambda k: (raw[k] - floors[k], proportions[k]),
        reverse=True,
    )
    for key in ranked[:remainder]:
        floors[key] += 1
    return floors


def sample_task_diverse(
    rng: random.Random,
    candidates: list[dict],
    n: int,
    task_use: Counter,
) -> list[dict]:
    """Sample n rows, preferring tasks with fewer prior selections."""
    if n <= 0:
        return []
    if len(candidates) <= n:
        return list(candidates)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_task[row["task_id"]].append(row)
    for rows in by_task.values():
        rng.shuffle(rows)

    selected: list[dict] = []
    available_tasks = set(by_task.keys())
    while len(selected) < n and available_tasks:
        min_use = min(task_use.get(t, 0) for t in available_tasks)
        tier = [t for t in available_tasks if task_use.get(t, 0) == min_use]
        tid = rng.choice(tier)
        row = by_task[tid].pop()
        selected.append(row)
        task_use[tid] += 1
        if not by_task[tid]:
            available_tasks.remove(tid)

    return selected


def _filter_pool(
    pool: list[dict],
    *,
    kind: str | None = None,
    status: str | None = None,
    operator_code: str | None = None,
    exclude_ids: set[str],
) -> list[dict]:
    rows = []
    for row in pool:
        if row["sample_id"] in exclude_ids:
            continue
        if kind is not None and row["kind"] != kind:
            continue
        if status is not None and row.get("status") != status:
            continue
        if operator_code is not None and row.get("operator_code") != operator_code:
            continue
        rows.append(row)
    return rows


def _sample_kind_bucket(
    pool: list[dict],
    kind: str,
    target: int,
    rng: random.Random,
    selected: list[dict],
    selected_ids: set[str],
    task_use: Counter,
    ref_status: Counter | None,
    ref_operators: Counter | None,
) -> None:
    if target <= 0:
        return

    if kind == "mutant" and ref_status and sum(ref_status.values()) > 0:
        status_targets = allocate_counts(target, ref_status)
        for status, status_n in status_targets.items():
            if status_n <= 0:
                continue
            if status == "fail" and ref_operators and sum(ref_operators.values()) > 0:
                op_targets = allocate_counts(status_n, ref_operators)
                for op, op_n in op_targets.items():
                    if op_n <= 0:
                        continue
                    candidates = _filter_pool(
                        pool,
                        kind=kind,
                        status=status,
                        operator_code=op,
                        exclude_ids=selected_ids,
                    )
                    chosen = sample_task_diverse(rng, candidates, op_n, task_use)
                    selected.extend(chosen)
                    selected_ids.update(row["sample_id"] for row in chosen)
            else:
                candidates = _filter_pool(
                    pool,
                    kind=kind,
                    status=status,
                    exclude_ids=selected_ids,
                )
                chosen = sample_task_diverse(rng, candidates, status_n, task_use)
                selected.extend(chosen)
                selected_ids.update(row["sample_id"] for row in chosen)
        return

    if kind in ("mutant", "confound") and ref_operators and sum(ref_operators.values()) > 0:
        op_targets = allocate_counts(target, ref_operators)
        for op, op_n in op_targets.items():
            if op_n <= 0:
                continue
            candidates = _filter_pool(
                pool,
                kind=kind,
                operator_code=op,
                exclude_ids=selected_ids,
            )
            chosen = sample_task_diverse(rng, candidates, op_n, task_use)
            selected.extend(chosen)
            selected_ids.update(row["sample_id"] for row in chosen)
        return

    candidates = _filter_pool(pool, kind=kind, exclude_ids=selected_ids)
    chosen = sample_task_diverse(rng, candidates, target, task_use)
    selected.extend(chosen)
    selected_ids.update(row["sample_id"] for row in chosen)


def stratified_sample(
    pool: list[dict],
    n: int,
    reference_root: Path,
    seed: int,
) -> list[dict]:
    ref_kind = reference_kind_counts(reference_root)
    ref_mutant_status = reference_mutant_status_counts(reference_root)
    ref_mutant_ops = reference_operator_counts(reference_root, "mutant")
    ref_confound_ops = reference_operator_counts(reference_root, "confound")

    kind_targets = allocate_counts(n, ref_kind)
    rng = random.Random(seed)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    task_use: Counter = Counter()

    for kind in ("canonical", "mutant", "confound"):
        _sample_kind_bucket(
            pool,
            kind,
            kind_targets.get(kind, 0),
            rng,
            selected,
            selected_ids,
            task_use,
            ref_mutant_status if kind == "mutant" else None,
            ref_mutant_ops if kind == "mutant" else ref_confound_ops if kind == "confound" else None,
        )

    if len(selected) < n:
        remaining = [row for row in pool if row["sample_id"] not in selected_ids]
        need = n - len(selected)
        if len(remaining) < need:
            raise ValueError(
                f"Pool too small: requested {n}, could only collect {len(selected) + len(remaining)}"
            )
        top_up = sample_task_diverse(rng, remaining, need, task_use)
        selected.extend(top_up)
        selected_ids.update(row["sample_id"] for row in top_up)

    selected.sort(key=lambda row: (_task_sort_key(row["task_id"]), row["kind"], row["sample_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=270)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("dataset/isolated_bigcodebench"),
        help="Reference Phase 1 output root for stratification proportions",
    )
    args = parser.parse_args()

    pool = flatten_frozen(args.frozen)
    ref_kind = reference_kind_counts(args.reference_root)
    ref_status = reference_mutant_status_counts(args.reference_root)
    sample = stratified_sample(pool, args.n, args.reference_root, args.seed)

    write_jsonl(args.output, sample)

    got_kind = Counter(row["kind"] for row in sample)
    got_status = Counter(
        row.get("status", "?")
        for row in sample
        if row["kind"] == "mutant"
    )
    task_counts = Counter(row["task_id"] for row in sample)

    print(f"Sampled {len(sample)} rows -> {args.output}")
    print(f"Reference kind counts: {dict(ref_kind)}")
    print(f"Sample kind counts:    {dict(got_kind)}")
    for kind in ("canonical", "mutant", "confound"):
        ref_pct = 100 * ref_kind[kind] / sum(ref_kind.values())
        got_pct = 100 * got_kind[kind] / len(sample) if sample else 0
        print(f"  {kind:9s} ref {ref_pct:5.1f}%  sample {got_pct:5.1f}%")

    print(f"Mutant status (sample): {dict(got_status)}")
    ref_fail = ref_status.get("fail", 0)
    ref_timeout = ref_status.get("timeout", 0)
    ref_mutant_total = ref_fail + ref_timeout
    if ref_mutant_total and got_kind["mutant"]:
        print(
            "  fail/timeout ref "
            f"{100 * ref_fail / ref_mutant_total:5.1f}%/{100 * ref_timeout / ref_mutant_total:5.1f}%  "
            f"sample "
            f"{100 * got_status.get('fail', 0) / got_kind['mutant']:5.1f}%/"
            f"{100 * got_status.get('timeout', 0) / got_kind['mutant']:5.1f}%"
        )

    print(f"Unique tasks: {len(task_counts)} / {len(sample)}")
    print(f"Max rows per task: {max(task_counts.values()) if task_counts else 0}")


if __name__ == "__main__":
    main()
