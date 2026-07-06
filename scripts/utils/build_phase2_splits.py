import json
import random
import argparse
from pathlib import Path

def parse_triplets(path, kind_filter):
    """
    kind_filter: set of kinds to include. e.g. {"canonical", "mutant", "confound"}
    Returns list of task dicts formatted for phase2.
    """
    tasks = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tid = d["task_id"]
            
            impls = []
            if "canonical" in kind_filter:
                impls.append({
                    "implementation_id": f"{tid}/canonical",
                    "kind": "canonical",
                    "label": "correct",
                    "code": d["canonical"]
                })
            
            for m in d.get("mutants", []):
                if "mutant" in kind_filter:
                    impls.append({
                        "implementation_id": m["mutant_id"],
                        "kind": "mutant",
                        "label": "incorrect",
                        "code": m["code"]
                    })
                if "confound" in kind_filter:
                    for cv in m.get("confounded_variants", []):
                        impls.append({
                            "implementation_id": cv["variant_id"],
                            "kind": "confound",
                            "label": "incorrect",
                            "code": cv["code"]
                        })
            
            tasks.append({
                "task_id": tid,
                "task": d.get("instruct_prompt", d.get("complete_prompt")),
                "prompt_variants": d.get("prompt_variants", {}),
                "entry_point": d["entry_point"],
                "implementations": impls
            })
    return tasks

def write_split(out_path, tasks):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

def main():
    base_dir = Path("/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness")
    data_dir = base_dir / "dataset/4_isolated_run_v3"
    out_base = base_dir / "dataset/deliverables/phase2_270"
    
    path_syn_m = data_dir / "sample_270_seed0_allow_unconfounded_triplets.jsonl"
    path_syn_mc = data_dir / "sample_270_seed0_triplets.jsonl"
    
    # Pre-parse datasets
    ds_icse_bcb = parse_triplets(path_syn_mc, {"canonical"})
    ds_syn_m = parse_triplets(path_syn_m, {"canonical", "mutant"})
    ds_syn_mc = parse_triplets(path_syn_mc, {"canonical", "mutant", "confound"})
    
    # Keep identical task order
    task_ids = [t["task_id"] for t in ds_icse_bcb]
    random.seed(42)
    shuffled_tids = list(task_ids)
    random.shuffle(shuffled_tids)
    
    folds = [shuffled_tids[i::10] for i in range(10)]
    
    def get_split(ds, tids):
        return [t for t in ds if t["task_id"] in tids]
    
    setups = [
        ("S0-BCB", ds_icse_bcb, ds_icse_bcb, ds_icse_bcb),
        ("S1-M", ds_syn_m, ds_syn_m, ds_syn_m),
        ("S1-MC", ds_syn_mc, ds_syn_mc, ds_syn_mc),
        ("S2-M-BCB", ds_syn_m, ds_syn_m, ds_icse_bcb),
        ("S2-MC-BCB", ds_syn_mc, ds_syn_mc, ds_icse_bcb),
    ]
    
    for name, fit_ds, val_ds, test_ds in setups:
        setup_dir = out_base / name
        print(f"Generating {name}...")
        for fold_idx in range(10):
            fit_tids = folds[fold_idx]
            val_tids = folds[(fold_idx + 1) % 10]
            test_tids = []
            for i in range(10):
                if i != fold_idx and i != (fold_idx + 1) % 10:
                    test_tids.extend(folds[i])
            
            fold_dir = setup_dir / f"fold_{fold_idx}"
            write_split(fold_dir / "fit.jsonl", get_split(fit_ds, fit_tids))
            write_split(fold_dir / "val.jsonl", get_split(val_ds, val_tids))
            # Some scripts expect validate.jsonl
            write_split(fold_dir / "validate.jsonl", get_split(val_ds, val_tids))
            write_split(fold_dir / "test.jsonl", get_split(test_ds, test_tids))
    print("Done generating BCB splits.")

if False:
    main()

def parse_humaneval(path):
    tasks = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tid = d["task_id"]
            
            impls = [{
                "implementation_id": f"{tid}/canonical",
                "kind": "canonical",
                "label": "correct",
                "code": d["canonical_solution"]
            }]
            
            tasks.append({
                "task_id": tid,
                "task": d.get("instruct_prompt", d.get("complete_prompt")),
                "prompt_variants": {},
                "entry_point": d["entry_point"],
                "implementations": impls
            })
    return tasks

def main_he():
    base_dir = Path("/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness")
    data_dir = base_dir / "dataset/4_isolated_run_v3"
    he_path = base_dir / "dataset/isolated_humaneval/intermediate/tasks.jsonl"
    out_base = base_dir / "dataset/deliverables/phase2_270"
    
    path_syn_m = data_dir / "sample_270_seed0_allow_unconfounded_triplets.jsonl"
    path_syn_mc = data_dir / "sample_270_seed0_triplets.jsonl"
    
    ds_syn_m = parse_triplets(path_syn_m, {"canonical", "mutant"})
    ds_syn_mc = parse_triplets(path_syn_mc, {"canonical", "mutant", "confound"})
    ds_icse_he = parse_humaneval(he_path)
    
    # Shuffle HE tasks
    he_task_ids = [t["task_id"] for t in ds_icse_he]
    random.seed(42)
    shuffled_he = list(he_task_ids)
    random.shuffle(shuffled_he)
    he_folds = [shuffled_he[i::10] for i in range(10)]
    
    # Re-use BCB shuffling for train/val
    bcb_tids = [t["task_id"] for t in ds_syn_mc]
    shuffled_bcb = list(bcb_tids)
    random.seed(42)
    random.shuffle(shuffled_bcb)
    bcb_folds = [shuffled_bcb[i::10] for i in range(10)]
    
    def get_split(ds, tids):
        return [t for t in ds if t["task_id"] in tids]
    
    setups = [
        ("S0-HE", ds_icse_he, ds_icse_he, ds_icse_he, he_folds, he_folds, he_folds),
        ("S2-M-HE", ds_syn_m, ds_syn_m, ds_icse_he, bcb_folds, bcb_folds, he_folds),
        ("S2-MC-HE", ds_syn_mc, ds_syn_mc, ds_icse_he, bcb_folds, bcb_folds, he_folds),
    ]
    
    for name, fit_ds, val_ds, test_ds, fit_f, val_f, test_f in setups:
        setup_dir = out_base / name
        print(f"Generating {name}...")
        for fold_idx in range(10):
            fit_tids = fit_f[fold_idx]
            val_tids = val_f[(fold_idx + 1) % 10]
            test_tids = []
            for i in range(10):
                if i != fold_idx and i != (fold_idx + 1) % 10:
                    test_tids.extend(test_f[i])
            
            fold_dir = setup_dir / f"fold_{fold_idx}"
            write_split(fold_dir / "fit.jsonl", get_split(fit_ds, fit_tids))
            write_split(fold_dir / "val.jsonl", get_split(val_ds, val_tids))
            write_split(fold_dir / "validate.jsonl", get_split(val_ds, val_tids))
            write_split(fold_dir / "test.jsonl", get_split(test_ds, test_tids))
    print("Done generating HE splits.")

if False:
    main_he()
if __name__ == "__main__":
    main()
    main_he()
