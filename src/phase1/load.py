"""Step 1: Load canonical BigCodeBench tasks and write tasks.jsonl."""

import ast

from bigcodebench.data import get_bigcodebench

from src.phase1.config import Phase1Config
from src.phase1.io_utils import write_jsonl

KEEP_FIELDS = [
    "task_id", "complete_prompt", "instruct_prompt",
    "canonical_solution", "test", "entry_point",
]


def load_tasks(cfg: Phase1Config) -> None:
    raw = get_bigcodebench(subset=cfg.subset)

    records = []
    for task_id in sorted(raw.keys(), key=_task_sort_key):
        task = raw[task_id]
        rec = {k: task[k] for k in KEEP_FIELDS}

        # Invariant: complete_prompt + canonical_solution must compile
        full_source = rec["complete_prompt"] + rec["canonical_solution"]
        try:
            ast.parse(full_source)
        except SyntaxError as e:
            print(f"WARN: skipping {task_id} — does not parse: {e}")
            continue

        records.append(rec)

    write_jsonl(cfg.tasks_path, records)
    print(f"Wrote {len(records)} tasks to {cfg.tasks_path}")


def _task_sort_key(task_id: str):
    # "BigCodeBench/627" -> try numeric sort on the suffix
    parts = task_id.rsplit("/", 1)
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    return (task_id, 0)
