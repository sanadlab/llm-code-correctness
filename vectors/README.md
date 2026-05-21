# Reading Vectors

Pre-computed reading vectors, one folder per model:

- `codellama__CodeLlama-7b-Instruct-hf/`
- `mistralai__Mistral-7B-Instruct-v0.3/`
- `Qwen__Qwen2.5-Coder-7B-Instruct/`
- `infly__OpenCoder-8B-Instruct/`

Inside each model the top-level directories are split families:

- `icse26/` — vectors from **ICSE26_DS** (fits on its 10 folds)
- `s1_baseline/`, `s2_test_confound/`, `s3_holdout_*/`, `s4_confound_fit/`, `s5_mixed/`, `s6_timeout_only/`, `s7_cross_status/`, `s8_single_*/` — vectors from the **Mutation_DS** splits (see `../dataset/bigcodebench/phase2/`)
- `ood_mbpp/`, `ood_synthetic/` — vectors fit on `icse26/mbpp` or `icse26/synthetic` for out-of-distribution experiments

## Layout inside a split family

```
<split_family>/<dataset>/<distribution>/<method>/<fit_framing>/<representation>/fold_N/
    direction.pt       # the vector(s), torch tensors
    metadata.json      # model, fold, method, framing, representation, etc.
```

- **dataset**: target benchmark (`bigcodebench` or `humaneval`).
- **distribution**: `id` for in-distribution fits (currently the only value).
- **method**: how the vector was extracted
  - `pca` — the technique used in the ICSE 2026 paper
  - `mean_diff` — Anthropic-style mean of differences (persona-vectors approach)
- **fit_framing**: how the prompt was set up when collecting hidden states for fitting (`raw`, `function`, `concept`, …).
- **representation**: which token activation was pooled (`sequence_last`, `response_avg`).

Each `fold_N/` corresponds to the fit set of the matching fold in `../dataset/`.

## What's currently here

Vectors for all 4 models, datasets, and methods.

---

Ask if something looks wrong or unclear.
