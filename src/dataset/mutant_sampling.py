"""Deterministic mutant subsampling helpers."""

from __future__ import annotations

import copy
import hashlib
import math
import warnings
from typing import Any

from src.dataset.operator_weights import operator_frequencies


def rng_for_key(seed: int, *parts: str) -> int:
    payload = "\0".join(parts).encode()
    digest = hashlib.sha256(f"{seed}\0".encode() + payload).digest()
    return int.from_bytes(digest[:8], "big")


def uniform01(seed: int, *parts: str) -> float:
    """Deterministic uniform in (0, 1] suitable for weighted sampling."""
    value = rng_for_key(seed, *parts)
    # Map to (0, 1]; avoid exact 0 for log stability.
    return (value % (2**53 - 1) + 1) / (2**53)


def split_impls(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = None
    mutants: list[dict[str, Any]] = []
    for impl in record.get("implementations", []):
        if impl.get("kind") == "canonical":
            if canonical is None:
                canonical = impl
        elif impl.get("kind") == "mutant":
            mutants.append(impl)
    if canonical is None:
        raise ValueError(f"record {record.get('task_id')} has no canonical implementation")
    return canonical, mutants


def make_record(
    record: dict[str, Any],
    canonical: dict[str, Any],
    mutants: list[dict[str, Any]],
) -> dict[str, Any]:
    out = {k: v for k, v in record.items() if k != "implementations"}
    out["implementations"] = [copy.deepcopy(canonical)] + [copy.deepcopy(m) for m in mutants]
    return out


def mutant_weights(
    mutants: list[dict[str, Any]],
    op_freq: dict[str, float],
) -> list[float]:
    """Per-mutant weight = operator frequency / count of same operator in task."""
    op_counts: dict[str, int] = {}
    for mutant in mutants:
        code = mutant.get("operator_code", "")
        op_counts[code] = op_counts.get(code, 0) + 1

    weights: list[float] = []
    for mutant in mutants:
        code = mutant.get("operator_code", "")
        freq = op_freq.get(code, 0.0)
        if freq <= 0.0:
            warnings.warn(
                f"mutant {mutant.get('implementation_id')} has unknown/zero-weight "
                f"operator {code!r}",
                stacklevel=3,
            )
        count = op_counts.get(code, 1)
        weights.append(freq / count if freq > 0.0 else 0.0)
    return weights


def weighted_sample_indices(
    weights: list[float],
    k: int,
    seed: int,
    key_parts: tuple[str, ...],
) -> list[int]:
    """Efraimidis-Spirakis weighted sampling without replacement."""
    if k > len(weights):
        raise ValueError(f"cannot sample k={k} from {len(weights)} items")
    if k <= 0:
        return []

    scored: list[tuple[float, int]] = []
    for i, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        u = uniform01(seed, *key_parts, str(i))
        key = math.log(u) / weight
        scored.append((key, i))

    if len(scored) < k:
        raise ValueError(
            f"only {len(scored)} positive-weight candidates available, need k={k}"
        )

    scored.sort(reverse=True)
    return [idx for _, idx in scored[:k]]


def subsample_record_uniform(
    record: dict[str, Any],
    n_mutants: int,
    seed: int,
    key_parts: tuple[str, ...],
) -> dict[str, Any] | None:
    canonical, mutants = split_impls(record)
    if len(mutants) < n_mutants:
        return None
    mutants_sorted = sorted(mutants, key=lambda m: m.get("implementation_id", ""))
    rng = rng_for_key(seed, *key_parts)
    indices = list(range(len(mutants_sorted)))
    indices.sort(key=lambda i: rng_for_key(rng, str(i)))
    chosen = [mutants_sorted[i] for i in indices[:n_mutants]]
    return make_record(record, canonical, chosen)


def subsample_record_weighted(
    record: dict[str, Any],
    n_mutants: int,
    op_freq: dict[str, float],
    seed: int,
    key_parts: tuple[str, ...],
) -> dict[str, Any] | None:
    canonical, mutants = split_impls(record)
    if len(mutants) < n_mutants:
        return None
    mutants_sorted = sorted(mutants, key=lambda m: m.get("implementation_id", ""))
    weights = mutant_weights(mutants_sorted, op_freq)
    try:
        indices = weighted_sample_indices(weights, n_mutants, seed, key_parts)
    except ValueError:
        return None
    chosen = [mutants_sorted[i] for i in indices]
    return make_record(record, canonical, chosen)


def frequencies_from_counts(counts: dict[str, int]) -> dict[str, float]:
    from collections import Counter

    return operator_frequencies(Counter(counts))
