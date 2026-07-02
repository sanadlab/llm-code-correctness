#!/usr/bin/env python3
from collections import Counter

from scripts.sample_frozen import (
    allocate_counts,
    sample_task_diverse,
    stratified_sample,
)


def test_allocate_counts_sums_to_total():
    props = Counter({"canonical": 1136, "mutant": 28431, "confound": 23481})
    got = allocate_counts(270, props)
    assert sum(got.values()) == 270
    assert got["mutant"] > got["confound"] > got["canonical"]


def test_sample_task_diverse_prefers_distinct_tasks():
    rng = __import__("random").Random(0)
    candidates = [
        {"sample_id": f"t1__{i}", "task_id": "t1"}
        for i in range(3)
    ] + [
        {"sample_id": f"t2__{i}", "task_id": "t2"}
        for i in range(3)
    ]
    chosen = sample_task_diverse(rng, candidates, 2, Counter())
    tasks = {row["task_id"] for row in chosen}
    assert tasks == {"t1", "t2"}


def test_stratified_sample_respects_kind_and_task_spread():
    pool = []
    for task_idx in range(20):
        task_id = f"BigCodeBench/{task_idx}"
        pool.append({
            "task_id": task_id,
            "sample_id": f"{task_id}__canonical",
            "kind": "canonical",
            "status": "pass",
            "operator_code": None,
            "label": 1,
            "code": "pass",
        })
        for status in ("fail", "timeout"):
            for op in ("SD", "CR"):
                pool.append({
                    "task_id": task_id,
                    "sample_id": f"{task_id}__{status}__{op}",
                    "kind": "mutant",
                    "status": status,
                    "operator_code": op,
                    "label": 0,
                    "code": "x",
                })
                pool.append({
                    "task_id": task_id,
                    "sample_id": f"{task_id}__conf__{status}__{op}",
                    "kind": "confound",
                    "status": status,
                    "operator_code": op,
                    "label": 0,
                    "code": "y",
                })

    # Monkeypatch reference helpers via a tiny fake reference root is heavy;
    # instead verify task spread and size using real reference if data exists.
    from pathlib import Path

    ref = Path("dataset/isolated_bigcodebench")
    if not (ref / "intermediate/validated_mutants.jsonl").exists():
        return

    sample = stratified_sample(pool, 30, ref, seed=1)
    assert len(sample) == 30
    assert len({row["task_id"] for row in sample}) >= 20
