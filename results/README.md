# Results — Layer Selection & Monitoring

One folder per model, same names as in `../vectors/`. The interesting ones for the summer project are:

- `<model>/icse26_bigcodebench/icse26/` — results on ICSE26_DS (BigCodeBench tasks)
- `<model>/icse26_humaneval/icse26/`   — results on ICSE26_DS (HumanEval tasks)
- `<model>/bigcodebench/<split>/`       — results on Mutation_DS splits

Ignore the `icse26_sentiment*`, `icse26_translations*`, etc. folders.

## Layout

```
<model>/<dataset>/<split_family>/phase3/<method>/<framing>/fold_N/monitoring/
    monitoring_results.json          # per-layer validation scores → used to pick the layer

<model>/<dataset>/<split_family>/phase4/<method>/<framing>/fold_N/monitoring/
    monitoring_test_results.json     # selected layer + test-set metrics
    monitoring_per_task.jsonl        # per-task scores on the test set
```

- `phase3` = layer selection (run on the validate split, scores every layer)
- `phase4` = final evaluation (the selected layer applied to the test split)

`method` and `framing` mean the same thing as in `../vectors/README.md` (`pca` / `mean_diff` / `random`, and `repe` / `function`).

## Where to find the "selected layer"

Open `phase4/<method>/<framing>/fold_N/monitoring/monitoring_test_results.json` and look at the `phase3_selected` field — that's the layer chosen for this model/dataset/method/framing/fold. Example:

```json
"phase3_selected": { "rep_type": "sequence_last", "layer": 1, "pair_idx": null }
```

## What's currently here

- ICSE26_DS (BigCodeBench and HumanEval): `pca` and `mean_diff` with the `repe` framing only. No `function` framing yet.
- Mutation_DS splits: `pca` and `mean_diff` with the `function` framing (and `mean_diff/neutral` for some splits). Only folds 0 and 1.

The same caveats from `../vectors/README.md` apply: the missing combinations are things you may end up generating.

---

Ask if something is unclear.
