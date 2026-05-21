"""Step 7: Freeze the task-level dataset.

Groups validated killed mutants under their parent task, embeds kept confounded
variants, excludes tasks with zero killed mutants, and writes frozen.jsonl + metadata.json.
"""

import json
from collections import defaultdict
from pathlib import Path

from src.phase1.config import Phase1Config
from src.phase1.io_utils import read_jsonl, write_jsonl


def freeze(cfg: Phase1Config) -> None:
    tasks = {t["task_id"]: t for t in read_jsonl(cfg.validated_tasks_path)}
    validated_mutants = read_jsonl(cfg.validated_mutants_path)
    validated_confounded = read_jsonl(cfg.validated_confounded_path)

    # Index confounded variants by source_mutant_id
    confounded_by_mutant = defaultdict(list)
    for v in validated_confounded:
        confounded_by_mutant[v["source_mutant_id"]].append(v)

    # Group killed mutants by task
    killed_by_task = defaultdict(list)
    total_killed = 0
    total_pass = 0
    total_fail = 0
    total_timeout = 0
    for m in validated_mutants:
        if m["status"] == "pass":
            total_pass += 1
            continue  # exclude equivalent mutants
        killed_by_task[m["task_id"]].append(m)
        total_killed += 1
        if m["status"] == "fail":
            total_fail += 1
        elif m["status"] == "timeout":
            total_timeout += 1

    # Build frozen records
    frozen = []
    for tid in sorted(tasks.keys(), key=_task_sort_key):
        task = tasks[tid]
        task_mutants = killed_by_task.get(tid, [])
        if not task_mutants:
            continue  # exclude tasks with zero killed mutants

        # Sort mutants by mutant_id
        task_mutants.sort(key=lambda m: m["mutant_id"])

        mutant_records = []
        for m in task_mutants:
            # Get confounded variants for this mutant
            variants = confounded_by_mutant.get(m["mutant_id"], [])
            variants.sort(key=lambda v: v["variant_id"])

            confounded_records = []
            for v in variants:
                confounded_records.append({
                    "variant_id": v["variant_id"],
                    "backend": v.get("backend", ""),
                    "code": v["code"],
                    "status": v["status"],
                    "details": v["details"],
                    "failure_fingerprint": v["failure_fingerprint"],
                })

            mutant_records.append({
                "mutant_id": m["mutant_id"],
                "code": m["code"],
                "operator": m["operator"],
                "operator_code": m["operator_code"],
                "site": m.get("site", {}),
                "status": m["status"],
                "details": m["details"],
                "failure_fingerprint": m["failure_fingerprint"],
                "confounded_variants": confounded_records,
            })

        frozen.append({
            "task_id": tid,
            "task": task["instruct_prompt"],
            "prompt_variants": {
                "complete": task["complete_prompt"],
                "instruct": task["instruct_prompt"],
            },
            "entry_point": task["entry_point"],
            "canonical": task["canonical_solution"],
            "mutants": mutant_records,
        })

    write_jsonl(cfg.frozen_path, frozen)

    # Load tasks.jsonl for total loaded count
    all_tasks = read_jsonl(cfg.tasks_path)

    # Write metadata
    metadata = {
        "benchmark": cfg.benchmark,
        "subset": cfg.subset,
        "num_loaded_tasks": len(all_tasks),
        "num_surviving_tasks": len(frozen),
        "num_generated_mutants": len(validated_mutants),
        "num_killed_mutants": total_killed,
        "num_fail_mutants": total_fail,
        "num_timeout_mutants": total_timeout,
        "num_confounded_variants": len(validated_confounded),
        "confounders_enabled": cfg.confound.enabled,
        "docker_image": cfg.validate.docker_image,
        "seed": cfg.seed,
    }

    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Frozen dataset: {len(frozen)} tasks, {total_killed} killed mutants")
    print(f"  → {cfg.frozen_path}")
    print(f"  → {cfg.metadata_path}")


def _task_sort_key(task_id: str):
    parts = task_id.rsplit("/", 1)
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    return (task_id, 0)
