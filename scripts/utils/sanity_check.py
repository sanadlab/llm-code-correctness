import json
from collections import Counter
from pathlib import Path

def count_kinds(path):
    c = Counter()
    tasks = 0
    with open(path) as f:
        for line in f:
            tasks += 1
            d = json.loads(line)
            for impl in d["implementations"]:
                c[impl["kind"]] += 1
    return tasks, dict(c)

def main():
    base = Path("/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness/dataset/deliverables/phase2_v3")
    
    print("=== Sanity Check: Task and Implementation Counts (Fold 0) ===")
    for split in ["S0-BCB", "S1-M", "S1-MC", "S2-M-BCB", "S2-MC-BCB", "S0-HE", "S2-M-HE", "S2-MC-HE"]:
        if not (base / split).exists():
            continue
        print(f"\n[{split}]")
        for subset in ["fit.jsonl", "val.jsonl", "test.jsonl"]:
            p = base / split / "fold_0" / subset
            if p.exists():
                tasks, kinds = count_kinds(p)
                print(f"  {subset:<10}: {tasks} tasks -> Kinds: {kinds}")
                
if __name__ == "__main__":
    main()
