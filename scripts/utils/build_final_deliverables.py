import json
import shutil
import random
from pathlib import Path

def parse_frozen(path, include_confounds=False):
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
                    "code": m["code"]
                })
                if include_confounds:
                    for cv in m.get("confounded_variants", []):
                        impls.append({
                            "implementation_id": cv["variant_id"],
                            "kind": "confound",
                            "label": "incorrect",
                            "code": cv["code"]
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
    if not path.exists():
        return tids
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

def partition_tids_into_folds(all_tids, n_folds=10, test_ratio=0.2):
    random.seed(42)
    tids = sorted(all_tids)
    
    folds = []
    for i in range(n_folds):
        random.shuffle(tids)
        n_test = int(len(tids) * test_ratio)
        n_val = int(len(tids) * 0.1) # 10% val
        
        test = tids[:n_test]
        val = tids[n_test:n_test+n_val]
        fit = tids[n_test+n_val:]
        folds.append({"fit": fit, "val": val, "test": test})
    return folds

def main():
    base_dir = Path("/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness")
    out_base = base_dir / "dataset/deliverables/final"
    old_base = base_dir / "dataset/deliverables/phase2"
    
    frozen_bcb = base_dir / "dataset/4_isolated_run_v3/frozen.jsonl"
    frozen_he = base_dir / "dataset/isolated_humaneval/frozen.jsonl"
    
    print("Parsing frozen datasets...")
    syn_m_bcb = parse_frozen(frozen_bcb, include_confounds=False)
    syn_mc_bcb = parse_frozen(frozen_bcb, include_confounds=True)
    syn_m_he = parse_frozen(frozen_he, include_confounds=False)
    syn_mc_he = parse_frozen(frozen_he, include_confounds=True)
    
    out_base.mkdir(parents=True, exist_ok=True)
    
    print("Copying frozen files...")
    shutil.copy(frozen_bcb, out_base / "frozen-BCB.jsonl")
    shutil.copy(frozen_he, out_base / "frozen-HE.jsonl")
    
    print("Generating S0 Baselines...")
    for name in ["S0-BCB", "S0-HE"]:
        if (old_base / name).exists():
            if (out_base / name).exists():
                shutil.rmtree(out_base / name)
            shutil.copytree(old_base / name, out_base / name)
            
    print("Generating S1 (BCB)...")
    for name, task_dict in [("S1-M-BCB", syn_m_bcb), ("S1-MC-BCB", syn_mc_bcb)]:
        for i in range(10):
            old_fold = old_base / "S1-MC" / f"fold_{i}"
            new_fold = out_base / name / f"fold_{i}"
            
            fit_tids = load_tids(old_fold / "fit.jsonl")
            val_tids = load_tids(old_fold / "val.jsonl")
            test_tids = load_tids(old_fold / "test.jsonl")
            
            write_split(new_fold / "fit.jsonl", fit_tids, task_dict)
            write_split(new_fold / "val.jsonl", val_tids, task_dict)
            write_split(new_fold / "validate.jsonl", val_tids, task_dict)
            write_split(new_fold / "test.jsonl", test_tids, task_dict)

    print("Generating S1 (HE)...")
    # For HE, we might not have old reference splits for S1, so let's generate random partitions
    # based on the available HE tasks.
    he_tids = list(syn_m_he.keys())
    he_folds = partition_tids_into_folds(he_tids, 10, 0.2)
    for name, task_dict in [("S1-M-HE", syn_m_he), ("S1-MC-HE", syn_mc_he)]:
        for i in range(10):
            new_fold = out_base / name / f"fold_{i}"
            fit_tids = he_folds[i]["fit"]
            val_tids = he_folds[i]["val"]
            test_tids = he_folds[i]["test"]
            
            write_split(new_fold / "fit.jsonl", fit_tids, task_dict)
            write_split(new_fold / "val.jsonl", val_tids, task_dict)
            write_split(new_fold / "validate.jsonl", val_tids, task_dict)
            write_split(new_fold / "test.jsonl", test_tids, task_dict)

    print("Generating S2 (BCB)...")
    s2_bcb_configs = [
        ("S2-M-BCB-valSYN", syn_m_bcb, "val", "test", "S0-BCB"),
        ("S2-M-BCB-valICSE", syn_m_bcb, "S0-BCB", "S0-BCB", "S0-BCB"),
        ("S2-M-BCB-testHE", syn_m_bcb, "val", "S0-HE", "S0-BCB"),
        ("S2-MC-BCB-valSYN", syn_mc_bcb, "val", "test", "S0-BCB"),
        ("S2-MC-BCB-valICSE", syn_mc_bcb, "S0-BCB", "S0-BCB", "S0-BCB"),
    ]
    for name, fit_dict, val_src, test_src, ref_test_baseline in s2_bcb_configs:
        for i in range(10):
            old_s1 = old_base / "S1-MC" / f"fold_{i}"
            new_fold = out_base / name / f"fold_{i}"
            new_fold.mkdir(parents=True, exist_ok=True)
            
            # fit is always SYN-M-BCB or SYN-MC-BCB
            fit_tids = load_tids(old_s1 / "fit.jsonl")
            write_split(new_fold / "fit.jsonl", fit_tids, fit_dict)
            
            # val
            if val_src == "val":
                val_tids = load_tids(old_s1 / "val.jsonl")
                write_split(new_fold / "val.jsonl", val_tids, fit_dict)
                write_split(new_fold / "validate.jsonl", val_tids, fit_dict)
            else:
                shutil.copy(old_base / val_src / f"fold_{i}" / "val.jsonl", new_fold / "val.jsonl")
                shutil.copy(old_base / val_src / f"fold_{i}" / "val.jsonl", new_fold / "validate.jsonl") # ICSE validation

            # test
            if test_src == "test":
                test_tids = load_tids(old_s1 / "test.jsonl")
                write_split(new_fold / "test.jsonl", test_tids, fit_dict)
            else:
                shutil.copy(old_base / test_src / f"fold_{i}" / "test.jsonl", new_fold / "test.jsonl")

    print("Generating S2 (HE)...")
    s2_he_configs = [
        ("S2-M-HE-valSYN", syn_m_he, "val", "test", "S0-HE"),
        ("S2-M-HE-valICSE", syn_m_he, "S0-HE", "S0-HE", "S0-HE"),
        ("S2-M-HE-testBCB", syn_m_he, "val", "S0-BCB", "S0-HE"),
        ("S2-MC-HE-valSYN", syn_mc_he, "val", "test", "S0-HE"),
        ("S2-MC-HE-valICSE", syn_mc_he, "S0-HE", "S0-HE", "S0-HE"),
    ]
    for name, fit_dict, val_src, test_src, ref_test_baseline in s2_he_configs:
        for i in range(10):
            new_fold = out_base / name / f"fold_{i}"
            new_fold.mkdir(parents=True, exist_ok=True)
            
            # fit is always SYN-M-HE or SYN-MC-HE
            fit_tids = he_folds[i]["fit"]
            write_split(new_fold / "fit.jsonl", fit_tids, fit_dict)
            
            # val
            if val_src == "val":
                val_tids = he_folds[i]["val"]
                write_split(new_fold / "val.jsonl", val_tids, fit_dict)
                write_split(new_fold / "validate.jsonl", val_tids, fit_dict)
            else:
                shutil.copy(old_base / val_src / f"fold_{i}" / "val.jsonl", new_fold / "val.jsonl")
                shutil.copy(old_base / val_src / f"fold_{i}" / "val.jsonl", new_fold / "validate.jsonl")

            # test
            if test_src == "test":
                test_tids = he_folds[i]["test"]
                write_split(new_fold / "test.jsonl", test_tids, fit_dict)
            else:
                shutil.copy(old_base / test_src / f"fold_{i}" / "test.jsonl", new_fold / "test.jsonl")

    print("All splits successfully generated in deliverables/final!")

if __name__ == "__main__":
    main()
