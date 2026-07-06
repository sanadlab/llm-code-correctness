"""Benchmark-specific constants for mutants_only dataset scripts."""

from __future__ import annotations

BENCHMARK_TASK_PREFIX: dict[str, str] = {
    "bigcodebench": "BigCodeBench/",
    "humaneval": "HumanEval/",
}

# Reference icse26 per-fold task counts (upper bound; subset builds may be lower).
BENCHMARK_EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "bigcodebench": {"fit": 45, "validate": 45, "test": 367},
    "humaneval": {"fit": 15, "validate": 15, "test": 121},
}
