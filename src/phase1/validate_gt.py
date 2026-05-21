"""Step 2: Validate canonical solutions in Docker.

Runs every canonical solution through the BigCodeBench test harness in Docker.
Passing tasks go to validated_tasks.jsonl; failures go to dropped_tasks.jsonl.
"""

from pathlib import Path

from src.phase1.config import Phase1Config
from src.phase1.docker_eval import run_docker_eval
from src.phase1.io_utils import read_jsonl, write_jsonl


def validate_ground_truth(cfg: Phase1Config) -> None:
    tasks = read_jsonl(cfg.tasks_path)
    print(f"Validating {len(tasks)} canonical solutions...")

    # Build samples: solution = complete_prompt + canonical_solution
    samples = []
    for t in tasks:
        samples.append({
            "task_id": t["task_id"],
            "solution": t["complete_prompt"] + t["canonical_solution"],
        })

    work_dir = cfg.intermediate_dir / "gt_eval_workdir"
    eval_results = run_docker_eval(samples, cfg, work_dir=work_dir, label="gt")

    validated = []
    dropped = []

    for t in tasks:
        tid = t["task_id"]
        results = eval_results.get(tid, [])
        if results:
            # Take the first (and only) result for this task
            r = results[0] if isinstance(results, list) else results
            status = r.get("status", "fail")
            details = r.get("details", {})
        else:
            status = "fail"
            details = {"ALL": "No evaluation result returned from Docker"}

        if status == "pass":
            validated.append({
                **t,
                "canonical_status": "pass",
                "canonical_details": details,
            })
        else:
            dropped.append({
                "task_id": tid,
                "reason": "canonical_failed_validation",
                "status": status,
                "details": details,
            })

    write_jsonl(cfg.validated_tasks_path, validated)
    write_jsonl(cfg.dropped_tasks_path, dropped)
    print(f"Validated: {len(validated)} passed, {len(dropped)} dropped")
    print(f"  → {cfg.validated_tasks_path}")
    print(f"  → {cfg.dropped_tasks_path}")
