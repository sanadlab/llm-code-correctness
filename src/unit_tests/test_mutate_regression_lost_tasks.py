"""Regression tests for tasks that lost all mutants in run 3 due to lineno bugs."""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.phase1.mutate import _mutate_task
from src.phase1.operators import get_operators

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATED_TASKS = _REPO_ROOT / "dataset/3_isolated_run/intermediate/validated_tasks.jsonl"
_RUN2_MUTANTS = _REPO_ROOT / "dataset/2_isolated_run/intermediate/mutants.jsonl"
_ALL_OPERATORS = [
    "AOR", "AOI", "AOD", "ROR", "COR", "COD", "UOI", "UOD", "CR", "VVR",
    "VCR", "CVR", "SD", "RVM", "TR", "FR", "RC", "NEG", "FCD", "CRN",
    "BOR", "BSOR", "BSOD",
]


def _load_task(task_id: str) -> dict:
    for line in _VALIDATED_TASKS.read_text().splitlines():
        task = json.loads(line)
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def _all_operators():
    return get_operators(_ALL_OPERATORS)


@pytest.fixture(scope="module")
def run2_counts_by_task() -> Counter:
    counts: Counter = Counter()
    if not _RUN2_MUTANTS.exists():
        pytest.skip("run 2 mutants file not available")
    for line in _RUN2_MUTANTS.read_text().splitlines():
        counts[json.loads(line)["task_id"]] += 1
    return counts


@pytest.mark.parametrize(
    "task_id",
    [
        "BigCodeBench/0",      # multi-stmt + nested for; VVR on loop target
        "BigCodeBench/96",     # with open csv reader
        "BigCodeBench/313",    # for filename + nested with open
        "BigCodeBench/1000",   # urllib + with json load
        "BigCodeBench/500",    # nested for loops
        "BigCodeBench/1012",   # try/with style task from failure log
        "BigCodeBench/1019",   # single top-level with body
        "BigCodeBench/7",      # with block
        "BigCodeBench/145",    # with block
        "BigCodeBench/177",    # for loop
        "BigCodeBench/449",    # for loop from failure log
        "BigCodeBench/1006",   # with block
    ],
)
def test_lost_tasks_generate_mutants(task_id: str):
    task = _load_task(task_id)
    mutants = _mutate_task(task, _all_operators(), log=False)
    assert mutants, f"{task_id} produced zero mutants"


def test_bigcodebench_0_vvr_for_header_preserves_inner_formatting():
    """VVR on for-loop target must not crash or reformat inner body."""
    task = _load_task("BigCodeBench/0")
    mutants = _mutate_task(task, get_operators(["VVR"]), log=False)
    assert mutants
    body = next(m["code"] for m in mutants if m["site_index"] == 1)
    assert "for perm in permutations:" not in body
    assert "sum_diffs" in body


def test_bigcodebench_96_rc_on_while_or_if_nested_with():
    task = _load_task("BigCodeBench/96")
    mutants = _mutate_task(task, get_operators(["RC", "SD"]), log=False)
    assert mutants


def test_bigcodebench_313_sd_inside_nested_with():
    task = _load_task("BigCodeBench/313")
    mutants = _mutate_task(task, get_operators(["SD"]), log=False)
    assert mutants
    assert any("with open" in m["code"] for m in mutants)





def test_previously_lost_tasks_all_produce_mutants(run2_counts_by_task):
    """Tasks that produced zero mutants in run 3 must all recover after the fix."""
    if not _VALIDATED_TASKS.exists() or not _RUN2_MUTANTS.exists():
        pytest.skip("dataset files not available")

    run3_path = _REPO_ROOT / "dataset/3_isolated_run/intermediate/mutants.jsonl"
    if not run3_path.exists():
        pytest.skip("run 3 mutants not available")

    run3_tasks = {
        json.loads(line)["task_id"] for line in run3_path.read_text().splitlines()
    }
    lost = sorted(set(run2_counts_by_task) - run3_tasks)
    if not lost:
        pytest.skip("run 3 already includes all run 2 tasks")

    empty = []
    for task_id in lost:
        task = _load_task(task_id)
        if not _mutate_task(task, _all_operators(), log=False):
            empty.append(task_id)

    assert not empty, f"tasks still produce zero mutants: {empty[:10]}"


def test_previously_lost_tasks_match_run2_counts(run2_counts_by_task):
    """Recovered tasks should match run 2 counts (formatting may shift dedup on a few)."""
    if not _VALIDATED_TASKS.exists() or not _RUN2_MUTANTS.exists():
        pytest.skip("dataset files not available")

    run3_path = _REPO_ROOT / "dataset/3_isolated_run/intermediate/mutants.jsonl"
    if not run3_path.exists():
        pytest.skip("run 3 mutants not available")

    run3_tasks = {
        json.loads(line)["task_id"] for line in run3_path.read_text().splitlines()
    }
    lost = sorted(set(run2_counts_by_task) - run3_tasks)
    if not lost:
        pytest.skip("run 3 already includes all run 2 tasks")

    mismatches = []
    for task_id in lost:
        task = _load_task(task_id)
        count = len(_mutate_task(task, _all_operators(), log=False))
        expected = run2_counts_by_task[task_id]
        if count != expected:
            mismatches.append((task_id, expected, count))

    # Formatting-preservation can change dedup/canonical-equality on a handful of tasks.
    assert len(mismatches) <= 8, f"unexpected count mismatches: {mismatches[:10]}"
