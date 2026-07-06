import json
from collections import Counter
from pathlib import Path

def compute_frequencies(path):
    operators = Counter()
    fingerprints = Counter()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            for impl in d["implementations"]:
                if impl["kind"] in ("mutant", "confound"):
                    op = impl.get("operator", "UNKNOWN")
                    fp = impl.get("failure_fingerprint", "UNKNOWN")
                    operators[op] += 1
                    fingerprints[fp] += 1
    return operators, fingerprints

def main():
    base = Path("/scratch/mki4895/summer-2026-llm-correctness/llm-code-correctness/dataset/deliverables/phase2_v3")
    
    print("=== Error Frequencies (Fold 0) ===")
    for split in ["S1-M", "S1-MC"]:
        print(f"\n[{split}] fold_0 fit.jsonl")
        p = base / split / "fold_0" / "fit.jsonl"
        if not p.exists():
            print("  Not found!")
            continue
        
        operators, fingerprints = compute_frequencies(p)
        print("  Top 5 Operators:")
        for op, count in operators.most_common(5):
            print(f"    {op}: {count}")
            
        print("  Top 5 Failure Fingerprints:")
        for fp, count in fingerprints.most_common(5):
            print(f"    {fp}: {count}")

if __name__ == "__main__":
    main()
