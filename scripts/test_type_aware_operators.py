#!/usr/bin/env python3
"""Smoke-test astroid type-aware mutation operators.

Usage:
    python scripts/test_type_aware_operators.py
    python scripts/test_type_aware_operators.py --task-id BigCodeBench/0
    python scripts/test_type_aware_operators.py --limit 5

Emit mutants.jsonl-compatible records (same schema as mutate step):
    python scripts/test_type_aware_operators.py --task-id BigCodeBench/0 \\
        --emit-mutants /tmp/test_mutants.jsonl

    python scripts/test_type_aware_operators.py --task-id BigCodeBench/0 \\
        --show-mutants 10
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.phase1.mutate import _mutate_task  # noqa: E402
from src.phase1.operators import (  # noqa: E402
    ALL_OPERATORS,
    _RETURN_COORD,
    _build_type_maps,
    get_operators,
)

TYPE_AWARE_OPS = frozenset({"AOR", "AOD", "CR", "VVR", "VCR", "CVR", "TR", "FR", "RVM"})


def _load_tasks(tasks_path: Path, task_id: str | None, limit: int | None) -> list[dict]:
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            task = json.loads(line)
            if task_id is not None and task["task_id"] != task_id:
                continue
            tasks.append(task)
            if task_id is not None:
                break
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def _test_task(task: dict, verbose: bool) -> dict:
    source = task["complete_prompt"] + task["canonical_solution"]
    entry = task["entry_point"]
    coord_to_type, type_to_vars, var_to_type = _build_type_maps(source, entry)
    tree = ast.parse(source)

    counts: dict[str, int] = {}
    examples: list[str] = []

    for code in sorted(TYPE_AWARE_OPS):
        if code not in ALL_OPERATORS:
            continue
        muts = ALL_OPERATORS[code]().enumerate_mutations(tree, entry, source)
        counts[code] = len(muts)
        if verbose and muts:
            m = muts[0]
            examples.append(
                f"  {code}: {m.site.original!r} -> {m.site.replacement!r} "
                f"@ {m.site.line}:{m.site.col}"
            )

    coord_sample = {
        k: v for k, v in sorted(coord_to_type.items(), key=lambda item: (str(item[0][0]), item[0][1]))
        if isinstance(k[0], int)
    }

    return {
        "task_id": task["task_id"],
        "coord_entries": len(coord_to_type),
        "coord_to_type": coord_sample,
        "var_to_type": dict(sorted(var_to_type.items())),
        "type_to_vars": {k: list(v) for k, v in sorted(type_to_vars.items())},
        "return_type": coord_to_type.get(_RETURN_COORD),
        "counts": counts,
        "examples": examples,
    }


def _parse_operators(value: str | None) -> list[str]:
    if not value:
        return sorted(TYPE_AWARE_OPS)
    codes = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [code for code in codes if code not in ALL_OPERATORS]
    if unknown:
        raise ValueError(f"Unknown operator code(s): {', '.join(unknown)}")
    return codes


def _collect_mutants(tasks: list[dict], operator_codes: list[str]) -> list[dict]:
    """Run the real mutate pipeline and return mutants.jsonl-shaped records."""
    operators = get_operators(operator_codes)
    mutants: list[dict] = []
    for task in tasks:
        mutants.extend(_mutate_task(task, operators, log=False))
    mutants.sort(key=lambda row: row["mutant_id"])
    return mutants


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_mutants(rows: list[dict], limit: int | None) -> None:
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test type-aware mutation operators")
    parser.add_argument(
        "--tasks-path",
        default="dataset/bigcodebench/intermediate/validated_tasks.jsonl",
        help="Path to validated_tasks.jsonl",
    )
    parser.add_argument("--task-id", default=None, help="Run on a single task id")
    parser.add_argument("--limit", type=int, default=3, help="Number of tasks (default: 3)")
    parser.add_argument(
        "--operators",
        default=None,
        help="Comma-separated operator codes (default: type-aware set)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show example mutations")
    parser.add_argument(
        "--emit-mutants",
        metavar="PATH",
        default=None,
        help="Write mutants.jsonl-compatible records to PATH",
    )
    parser.add_argument(
        "--show-mutants",
        type=int,
        nargs="?",
        const=5,
        metavar="N",
        help="Print first N mutant records as JSON (default: 5)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip type-map summary; only emit/show mutants",
    )
    args = parser.parse_args()

    tasks = _load_tasks(Path(args.tasks_path), args.task_id, args.limit)
    if not tasks:
        print("No tasks matched.", file=sys.stderr)
        sys.exit(1)

    operator_codes = _parse_operators(args.operators)

    if not args.summary_only:
        print(f"Testing {len(tasks)} task(s) from {args.tasks_path}\n")
        total_by_op: dict[str, int] = {code: 0 for code in sorted(operator_codes)}

        for task in tasks:
            result = _test_task(task, args.verbose)
            print(f"=== {result['task_id']} ===")
            print(f"  coord entries: {result['coord_entries']}")
            print(f"  var_to_type:   {result['var_to_type']}")
            print(f"  type_to_vars:  {result['type_to_vars']}")
            print(f"  coord_to_type: {result['coord_to_type']}")
            print(f"  return type:   {result['return_type']!r}")
            print("  mutation counts:")
            for code in sorted(operator_codes):
                n = result["counts"].get(code, 0)
                total_by_op[code] += n
                print(f"    {code}: {n}")
            if args.verbose and result["examples"]:
                print("  examples:")
                print("\n".join(result["examples"]))
            print()

        print("=== totals ===")
        for code in sorted(operator_codes):
            print(f"  {code}: {total_by_op[code]}")
        print()

    if args.emit_mutants or args.show_mutants is not None:
        mutants = _collect_mutants(tasks, operator_codes)
        if args.emit_mutants:
            out_path = Path(args.emit_mutants)
            _write_jsonl(out_path, mutants)
            print(f"Wrote {len(mutants)} mutants to {out_path}")
        if args.show_mutants is not None:
            if mutants:
                print(f"=== mutant records ({min(args.show_mutants, len(mutants))} of {len(mutants)}) ===")
                _print_mutants(mutants, args.show_mutants)
            else:
                print("No mutants generated.")

    print("Done.")


if __name__ == "__main__":
    main()
