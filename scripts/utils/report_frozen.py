#!/usr/bin/env python3
"""Print a summary report for a Phase 1 frozen dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.phase1.io_utils import read_jsonl


def _analyze_frozen(frozen_path: Path) -> dict:
    rows = read_jsonl(frozen_path)
    operator_counts = Counter()
    mutants_total = 0
    mutants_with_confound = 0
    confounds_total = 0
    status_counts = Counter()
    tasks_with_confound = 0

    for task in rows:
        task_has_confound = False
        for mutant in task.get("mutants", []):
            mutants_total += 1
            operator_counts[mutant.get("operator_code", "?")] += 1
            status_counts[mutant.get("status", "?")] += 1
            variants = mutant.get("confounded_variants") or []
            if variants:
                mutants_with_confound += 1
                task_has_confound = True
            confounds_total += len(variants)
        if task_has_confound:
            tasks_with_confound += 1

    impl_rows = len(rows) + mutants_total + confounds_total
    return {
        "frozen_path": str(frozen_path),
        "num_tasks": len(rows),
        "num_killed_mutants": mutants_total,
        "num_confounded_variants": confounds_total,
        "num_implementation_rows": impl_rows,
        "mutants_with_confound": mutants_with_confound,
        "mutants_without_confound": mutants_total - mutants_with_confound,
        "tasks_with_confound": tasks_with_confound,
        "mutant_status": dict(status_counts),
        "operator_counts": dict(operator_counts.most_common()),
        "kind_proportions": {
            "canonical": len(rows) / impl_rows if impl_rows else 0.0,
            "mutant": mutants_total / impl_rows if impl_rows else 0.0,
            "confound": confounds_total / impl_rows if impl_rows else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = _analyze_frozen(args.frozen)
    if args.metadata and args.metadata.exists():
        with open(args.metadata) as f:
            report["metadata"] = json.load(f)

    print("Frozen dataset report")
    print("=" * 60)
    print(f"Path:                    {report['frozen_path']}")
    print(f"Tasks (with killed):     {report['num_tasks']:,}")
    print(f"Killed mutants:          {report['num_killed_mutants']:,}")
    print(f"Confounded variants:     {report['num_confounded_variants']:,}")
    print(f"Flattened impl rows:     {report['num_implementation_rows']:,}")
    print(f"  canonical:             {report['num_tasks']:,} ({100*report['kind_proportions']['canonical']:.1f}%)")
    print(f"  mutant:                {report['num_killed_mutants']:,} ({100*report['kind_proportions']['mutant']:.1f}%)")
    print(f"  confound:              {report['num_confounded_variants']:,} ({100*report['kind_proportions']['confound']:.1f}%)")
    print(f"Mutants w/ confound:     {report['mutants_with_confound']:,}")
    print(f"Mutants w/o confound:    {report['mutants_without_confound']:,}")
    print(f"Mutant status:           {report['mutant_status']}")
    print("Top operators:")
    for op, count in list(report["operator_counts"].items())[:10]:
        print(f"  {op:6s} {count:6,}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
