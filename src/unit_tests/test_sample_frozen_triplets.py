#!/usr/bin/env python3
from collections import Counter
from pathlib import Path

from scripts.utils.sample_frozen_triplets import (
    stratified_triplet_sample,
    triplet_candidates,
    triplet_to_frozen_record,
    triplets_to_flat,
)


def test_triplet_sample_shape_and_flat():
    frozen = Path("dataset/2_isolated_run/frozen.jsonl")
    if not frozen.exists():
        return

    pool = triplet_candidates(frozen)
    chosen = stratified_triplet_sample(
        pool, 10, Path("dataset/isolated_bigcodebench"), seed=0,
    )
    records = [triplet_to_frozen_record(entry) for entry in chosen]

    assert len(records) == 10
    assert len({r["task_id"] for r in records}) == 10
    for record in records:
        assert len(record["mutants"]) == 1
        assert record["mutants"][0]["status"] == "fail"
        assert record["mutants"][0]["confounded_variants"]

    flat = triplets_to_flat(records)
    assert len(flat) == 30
    assert Counter(row["kind"] for row in flat) == Counter(
        {"canonical": 10, "mutant": 10, "confound": 10},
    )
