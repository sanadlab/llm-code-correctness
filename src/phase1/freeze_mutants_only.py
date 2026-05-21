"""Build a mutation-only frozen dataset from existing Phase 1 intermediates.

This is a convenience path for Phase 2 work before confound validation finishes.
It reads the validated tasks and validated mutants already present under a source
Phase 1 output root and writes a separate frozen dataset under a different output
root so it does not interfere with the future final dataset.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.phase1.io_utils import read_jsonl, write_jsonl


def _task_sort_key(task_id: str):
    parts = task_id.rsplit("/", 1)
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    return (task_id, 0)


def build_mutants_only_dataset(source_root: Path, output_root: Path) -> None:
    intermediate_dir = source_root / "intermediate"
    validated_tasks_path = intermediate_dir / "validated_tasks.jsonl"
    validated_mutants_path = intermediate_dir / "validated_mutants.jsonl"

    if not validated_tasks_path.exists():
        raise FileNotFoundError(f"validated tasks not found: {validated_tasks_path}")
    if not validated_mutants_path.exists():
        raise FileNotFoundError(f"validated mutants not found: {validated_mutants_path}")

    tasks = {t["task_id"]: t for t in read_jsonl(validated_tasks_path)}
    validated_mutants = read_jsonl(validated_mutants_path)

    killed_by_task = defaultdict(list)
    total_killed = 0
    total_generated = len(validated_mutants)
    for mutant in validated_mutants:
        if mutant.get("status") == "pass":
            continue
        killed_by_task[mutant["task_id"]].append(mutant)
        total_killed += 1

    frozen = []
    for task_id in sorted(tasks.keys(), key=_task_sort_key):
        task = tasks[task_id]
        task_mutants = killed_by_task.get(task_id, [])
        if not task_mutants:
            continue
        task_mutants.sort(key=lambda mutant: mutant["mutant_id"])

        mutant_records = []
        for mutant in task_mutants:
            mutant_records.append({
                "mutant_id": mutant["mutant_id"],
                "code": mutant["code"],
                "operator": mutant["operator"],
                "operator_code": mutant["operator_code"],
                "site": mutant.get("site", {}),
                "status": mutant["status"],
                "details": mutant["details"],
                "failure_fingerprint": mutant["failure_fingerprint"],
                "confounded_variants": [],
            })

        frozen.append({
            "task_id": task_id,
            "task": task["instruct_prompt"],
            "prompt_variants": {
                "complete": task["complete_prompt"],
                "instruct": task["instruct_prompt"],
            },
            "entry_point": task["entry_point"],
            "canonical": task["canonical_solution"],
            "mutants": mutant_records,
        })

    frozen_path = output_root / "frozen.jsonl"
    metadata_path = output_root / "metadata.json"
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(frozen_path, frozen)

    metadata = {
        "benchmark": "bigcodebench",
        "subset": "full",
        "source_root": str(source_root),
        "variant": "mutants_only",
        "num_loaded_tasks": len(tasks),
        "num_surviving_tasks": len(frozen),
        "num_generated_mutants": total_generated,
        "num_killed_mutants": total_killed,
        "num_confounded_variants": 0,
        "confounders_enabled": False,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mutation-only frozen dataset: {len(frozen)} tasks, {total_killed} killed mutants")
    print(f"  -> {frozen_path}")
    print(f"  -> {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mutation-only frozen dataset")
    parser.add_argument("--source-root", default="dataset/bigcodebench", help="Existing Phase 1 output root with intermediate files")
    parser.add_argument("--output-root", default="dataset/bigcodebench_mutants_only", help="Destination root for the mutation-only frozen dataset")
    args = parser.parse_args()

    build_mutants_only_dataset(Path(args.source_root), Path(args.output_root))


if __name__ == "__main__":
    main()
