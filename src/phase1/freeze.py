"""Step 7: Freeze the task-level dataset.

Groups validated killed mutants under their parent task, embeds kept confounded
variants, excludes tasks with zero killed mutants, and writes frozen.jsonl + metadata.json.
"""

import json
from collections import Counter, defaultdict
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

    # Group killed mutants by task. Passing mutants are equivalent, and timeout
    # mutants are too ambiguous for the frozen correctness dataset.
    killed_by_task = defaultdict(list)
    total_timeout_dropped = 0
    for m in validated_mutants:
        if m["status"] == "pass":
            continue  # exclude equivalent mutants
        if m["status"] == "timeout":
            total_timeout_dropped += 1
            continue
        killed_by_task[m["task_id"]].append(m)

    duplicate_stats = Counter()

    # Build frozen records
    frozen = []
    for tid in sorted(tasks.keys(), key=_task_sort_key):
        task = tasks[tid]
        task_mutants = killed_by_task.get(tid, [])
        if not task_mutants:
            continue  # exclude tasks with zero killed mutants

        task_mutants = _dedupe_cross_operator_mutants(
            task_mutants, confounded_by_mutant, duplicate_stats
        )

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

    # Load tasks.jsonl for total loaded count; fall back to validated_tasks.
    if cfg.tasks_path.exists():
        all_tasks = read_jsonl(cfg.tasks_path)
    else:
        all_tasks = read_jsonl(cfg.validated_tasks_path)

    # Write metadata
    kept_mutants = [m for task in frozen for m in task["mutants"]]
    kept_confounds = sum(
        len(m.get("confounded_variants") or [])
        for m in kept_mutants
    )

    metadata = {
        "benchmark": cfg.benchmark,
        "subset": cfg.subset,
        "num_loaded_tasks": len(all_tasks),
        "num_surviving_tasks": len(frozen),
        "num_generated_mutants": len(validated_mutants),
        "num_killed_mutants": len(kept_mutants),
        "num_fail_mutants": sum(1 for m in kept_mutants if m["status"] == "fail"),
        "num_timeout_mutants": 0,
        "num_confounded_variants": kept_confounds,
        "confounders_enabled": cfg.confound.enabled,
        "docker_image": cfg.validate.docker_image,
        "seed": cfg.seed,
        "cleanup": {
            "dropped_timeouts": total_timeout_dropped,
            "duplicate_code_cross_operator_groups": duplicate_stats["groups"],
            "dropped_duplicate_mutants": duplicate_stats["dropped"],
            "mixed_confound_duplicate_groups": duplicate_stats["mixed_confound_groups"],
            "rule": (
                "drop timeout mutants; for same-code mutants with different "
                "operator_code within a task, keep one, preferring a mutant "
                "with confounded_variants"
            ),
        },
    }

    cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Frozen dataset: {len(frozen)} tasks, {len(kept_mutants)} killed mutants")
    print(f"  dropped timeouts: {total_timeout_dropped}")
    print(f"  dropped duplicate mutants: {duplicate_stats['dropped']}")
    print(f"  → {cfg.frozen_path}")
    print(f"  → {cfg.metadata_path}")


def _dedupe_cross_operator_mutants(
    mutants: list[dict],
    confounded_by_mutant,
    stats: Counter,
) -> list[dict]:
    """Remove same-code mutants with different operator labels within one task."""
    by_code = defaultdict(list)
    for mutant in mutants:
        by_code[mutant.get("code", "")].append(mutant)

    deduped = []
    for group in by_code.values():
        operator_codes = {m.get("operator_code") for m in group}
        if len(group) <= 1 or len(operator_codes) <= 1:
            deduped.extend(group)
            continue

        stats["groups"] += 1
        stats["members"] += len(group)
        has_confound = [bool(confounded_by_mutant.get(m["mutant_id"])) for m in group]
        if any(has_confound) and not all(has_confound):
            stats["mixed_confound_groups"] += 1

        winner = min(
            group,
            key=lambda m: (
                -int(bool(confounded_by_mutant.get(m["mutant_id"]))),
                -len(confounded_by_mutant.get(m["mutant_id"], [])),
                m.get("mutant_id", ""),
            ),
        )
        deduped.append(winner)
        stats["dropped"] += len(group) - 1

    return deduped


def _task_sort_key(task_id: str):
    parts = task_id.rsplit("/", 1)
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    return (task_id, 0)
