"""Normalize mutant task records for icse26-style loading and weighted sampling."""

from __future__ import annotations

import copy
import re
from typing import Any

from src.dataset.operator_weights import operator_code_to_name

_OPERATOR_CODE_RE = re.compile(r"__([A-Z]+)__")


def parse_operator_code(implementation_id: str) -> str | None:
    match = _OPERATOR_CODE_RE.search(implementation_id)
    return match.group(1) if match else None


def normalize_label(label: Any) -> int:
    if isinstance(label, bool):
        raise ValueError(f"unsupported label: {label!r}")
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        if label == "correct":
            return 1
        if label == "incorrect":
            return 0
    raise ValueError(f"unsupported label: {label!r}")


def normalize_canonical_id(task_id: str, implementation_id: str) -> str:
    if implementation_id.endswith("/canonical"):
        return f"{task_id}__canonical"
    return implementation_id


def normalize_impl(impl: dict[str, Any], *, task_id: str) -> dict[str, Any]:
    out = copy.deepcopy(impl)
    out["label"] = normalize_label(out["label"])
    kind = out.get("kind")

    if kind == "canonical":
        out["implementation_id"] = normalize_canonical_id(task_id, out["implementation_id"])
        return out

    if kind in ("mutant", "confound"):
        impl_id = out["implementation_id"]
        if kind == "mutant":
            code = out.get("operator_code") or parse_operator_code(impl_id)
            if code:
                out["operator_code"] = code
                out["mutant_id"] = out.get("mutant_id") or impl_id
                names = operator_code_to_name()
                if code in names:
                    out["operator"] = names[code]
        return out

    return out


def normalize_task_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with normalized labels and mutant metadata."""
    out = copy.deepcopy(record)
    task_id = out["task_id"]
    out["implementations"] = [
        normalize_impl(impl, task_id=task_id) for impl in out.get("implementations", [])
    ]
    return out
