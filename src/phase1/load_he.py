"""Step 1: Load canonical HumanEval + EvalPlus tasks and write tasks.jsonl."""

from datasets import load_dataset
from src.phase1.config import Phase1Config
from src.phase1.io_utils import write_jsonl

KEEP_FIELDS = [
    "task_id", "complete_prompt", "instruct_prompt",
    "canonical_solution", "test", "entry_point",
]

def load_tasks(cfg: Phase1Config) -> None:
    print("Downloading codeparrot/instructhumaneval...")
    ds_parrot = load_dataset('codeparrot/instructhumaneval', split='test')
    
    print("Downloading evalplus/humanevalplus...")
    ds_plus = load_dataset('evalplus/humanevalplus', split='test')
    
    # Map evalplus tests by task_id
    plus_tests = {}
    for item in ds_plus:
        plus_tests[item["task_id"]] = item["test"]
    
    records = []
    for item in ds_parrot:
        task_id = item["task_id"]
        if task_id not in plus_tests:
            print(f"WARN: skipping {task_id} — not found in evalplus/humanevalplus")
            continue
            
        rec = {
            "task_id": task_id,
            "complete_prompt": item["prompt"],
            "instruct_prompt": item["instruction"],
            "canonical_solution": item["canonical_solution"],
            "test": plus_tests[task_id],
            "entry_point": item["entry_point"],
        }
        records.append(rec)
    
    write_jsonl(cfg.tasks_path, records)
    print(f"Wrote {len(records)} tasks to {cfg.tasks_path}")
