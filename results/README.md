# Results — Layer Selection & Monitoring

One folder per model, same names as in `../vectors/`. The top-level directories under each model are split families, matching `../vectors/<model>/`:

- `<model>/icse26/<benchmark>/...`     — results on ICSE26_DS
- `<model>/s*/...`                      — results on Mutation_DS splits
- `<model>/ood_mbpp/...`, `<model>/ood_synthetic/...` — OOD results

## Layout

```
<model>/<split_family>/<dataset>/<distribution>/<method>/<fit_framing>/<representation>/val=<framing>__test=<framing>/fold_N/
    results.json     # selected_layer + per-layer validation scores
    detailed.json    # per-task / per-pair details on the test set
    metadata.json    # model, dataset, split_family, fold, method, framings, etc.
```

- `<method>`, `<fit_framing>`, `<representation>` mean the same as in `../vectors/README.md`.
- `val=<framing>__test=<framing>/` records the framings used when **evaluating** on the validate / test sets (these can differ from `fit_framing`, e.g. fit on `raw`, evaluate on `concept`).

## Where to find the "selected layer"

Open `.../fold_N/results.json`. The top-level `selected_layer` field is the layer chosen on the validate split; `val_per_layer` has the per-layer validation scores it was chosen from. Example:

```json
{
  "fold": 3,
  "selected_layer": 16,
  "val_per_layer": { "2": { "pairwise_acc": 0.65 }, "3": { ... }, ... }
}
```

## What's currently here

Results (contain selected layers during validation) for all 4 models, datasets, and methods.


---

Ask if something is unclear.
