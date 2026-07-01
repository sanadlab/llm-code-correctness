"""Mutation operator counting and frequency helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SPLITS = ("fit", "validate", "test")
SPLIT_TO_JSONL = {
    "fit": "fit.jsonl",
    "validate": "validate.jsonl",
    "test": "test.jsonl",
}

# Mirrors src/phase1/operators.py ALL_OPERATORS names (kept local to avoid astroid).
OPERATOR_NAMES: dict[str, str] = {
    "AOR": "ArithmeticOperatorReplacement",
    "AOI": "ArithmeticOperatorInsertion",
    "AOD": "ArithmeticOperatorDeletion",
    "ROR": "RelationalOperatorReplacement",
    "COR": "ConditionalOperatorReplacement",
    "COD": "ConditionalOperatorDeletion",
    "UOI": "UnaryOperatorInsertion",
    "UOD": "UnaryOperatorDeletion",
    "CR": "ConstantReplacement",
    "VVR": "VarToVarReplacement",
    "VCR": "VarToConsReplacement",
    "CVR": "ConsToVarReplacement",
    "SD": "StatementDeletion",
    "RVM": "ReturnValueMutation",
    "TR": "TrueReturn",
    "FR": "FalseReturn",
    "RC": "RemoveConditional",
    "NEG": "Negation",
    "FCD": "FunctionCallDeletion",
    "CRN": "ConstructorReplacementNone",
    "BOR": "BitwiseOperatorReplacement",
    "BSOR": "BitshiftOperatorReplacement",
    "BSOD": "BitshiftOperatorDeletion",
}


def operator_code_to_name() -> dict[str, str]:
    return dict(OPERATOR_NAMES)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_operators(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for impl in record.get("implementations", []):
            if impl.get("kind") == "mutant":
                code = impl.get("operator_code")
                if code:
                    counts[code] += 1
    return counts


def operator_frequencies(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {code: count / total for code, count in counter.items()}


def count_operators_in_file(path: Path) -> Counter[str]:
    return count_operators(load_jsonl_records(path))


def format_operator_report(
    split: str,
    counts: Counter[str],
    *,
    names: dict[str, str] | None = None,
    include_descriptions: bool = True,
) -> str:
    names = names or operator_code_to_name()
    total = sum(counts.values())
    lines = [f"=== Frequency of Mutation Operators in {split}.jsonl ===", f"Total Mutants: {total}"]
    for code, count in counts.most_common():
        pct = 100.0 * count / total if total else 0.0
        suffix = ""
        if include_descriptions and code in names:
            suffix = f"  - {names[code]}"
        lines.append(f"  {code:4s}: {count:5d} ({pct:4.1f}%){suffix}")
    return "\n".join(lines)


def collect_fold_split_frequencies(
    data_root: Path,
    folds: list[int],
    splits: list[str],
) -> dict[int, dict[str, dict[str, float]]]:
    """Return {fold: {split: {operator_code: fraction}}}."""
    out: dict[int, dict[str, dict[str, float]]] = {}
    for fold in folds:
        out[fold] = {}
        for split in splits:
            path = data_root / f"fold_{fold}" / SPLIT_TO_JSONL[split]
            counts = count_operators_in_file(path)
            out[fold][split] = operator_frequencies(counts)
    return out
