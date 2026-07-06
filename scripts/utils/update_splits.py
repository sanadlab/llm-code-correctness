import json
import shutil
from pathlib import Path

def parse_frozen(path, include_confounds=False):
    """
    Parses frozen.jsonl into a dictionary: task_id -> task_dict
    """
    tasks = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tid = d["task_id"]
            
            impls = [{
                "implementation_id": f"{tid}/canonical",
                "kind": "canonical",
                "label": "correct",
                "code": d["canonical"]
            }]
            
            for m in d.get("mutants", []):
                impls.append({
                    "implementation_id": m["mutant_id"],
                    "kind": "mutant",
                    "label": "incorrect",
                    "code": m["code"],
                    "operator": m.get("operator_code", m.get("operator", "")),
                    "failure_fingerprint": m.get("failure_fingerprint", "")
                })
                if include_confounds:
                    for cv in m.get("confounded_variants", []):
                        impls.append({
                            "implementation_id": cv["variant_id"],
                            "kind": "confound",
                            "label": "incorrect",
                            "code": cv["code"],
                            "source_mutant_id": m["mutant_id"],
                            "failure_fingerprint": cv.get("failure_fingerprint", m.get("failure_fingerprint", ""))
                        })
            
            prompt = d.get("prompt_variants", {}).get("instruct", d.get("task", ""))
            
            tasks[tid] = {
                "task_id": tid,
                "task": prompt,
                "prompt_variants": d.get("prompt_variants", {}),
                "entry_point": d["entry_point"],
                "implementations": impls
            }
    return tasks

def load_tids(path):
    tids = []
    with open(path) as f:
        for line in f:
            tids.append(json.loads(line)["task_id"])
    return tids

def write_split(out_path, tids, task_dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for tid in tids:
            if tid in task_dict:
                f.write(json.dumps(task_dict[tid]) + "\n")

def main():
    base_dir = Path("/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness")
    frozen_path = base_dir / "dataset/4_isolated_run_v3/frozen.jsonl"
    old_base = base_dir / "dataset/deliverables/phase2"
    new_base = base_dir / "dataset/deliverables/phase2_v3"
    
    # 1. Parse frozen data
    print("Parsing frozen dataset...")
    syn_m = parse_frozen(frozen_path, include_confounds=False)
    syn_mc = parse_frozen(frozen_path, include_confounds=True)
    
    # 3. Generate S1-M, S1-MC
    for name, task_dict in [("S1-M", syn_m), ("S1-MC", syn_mc)]:
        print(f"Generating {name}...")
        for i in range(10):
            old_fold = old_base / "S1-MC" / f"fold_{i}"
            new_fold = new_base / name / f"fold_{i}"
            
            fit_tids = load_tids(old_fold / "fit.jsonl")
            val_tids = load_tids(old_fold / "val.jsonl")
            test_tids = load_tids(old_fold / "test.jsonl")
            
            write_split(new_fold / "fit.jsonl", fit_tids, task_dict)
            write_split(new_fold / "val.jsonl", val_tids, task_dict)
            write_split(new_fold / "validate.jsonl", val_tids, task_dict)
            write_split(new_fold / "test.jsonl", test_tids, task_dict)
            
    # 4. Generate S2-M-BCB, S2-M-HE, S2-MC-BCB, S2-MC-HE
    for name, task_dict in [("S2-M-BCB", syn_m), ("S2-M-HE", syn_m), ("S2-MC-BCB", syn_mc), ("S2-MC-HE", syn_mc)]:
        print(f"Generating {name}...")
        for i in range(10):
            old_fold = old_base / name.replace("S2-M-", "S2-MC-") / f"fold_{i}"
            new_fold = new_base / name / f"fold_{i}"
            new_fold.mkdir(parents=True, exist_ok=True)
            
            fit_tids = load_tids(old_fold / "fit.jsonl")
            val_tids = load_tids(old_fold / "val.jsonl")
            
            write_split(new_fold / "fit.jsonl", fit_tids, task_dict)
            write_split(new_fold / "val.jsonl", val_tids, task_dict)
            write_split(new_fold / "validate.jsonl", val_tids, task_dict)

if __name__ == "__main__":
    main()
