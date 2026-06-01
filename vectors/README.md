# Reading Vectors

Pre-computed reading vectors, one folder per model:

- `codellama__CodeLlama-7b-Instruct-hf/`
- `mistralai__Mistral-7B-Instruct-v0.3/`
- `Qwen__Qwen2.5-Coder-7B-Instruct/`
- `infly__OpenCoder-8B-Instruct/`

Inside each model you'll only care about the folders matching the two datasets from `../dataset/`:

- `icse26_bigcodebench/`, `icse26_humaneval/` — vectors from **ICSE26_DS**
- `bigcodebench/` — vectors from **Mutation_DS** (the splits under `dataset/bigcodebench/phase2/`)

Ignore `icse26_sentiment*`, `icse26_translations*`, etc.

## Layout inside a dataset folder

```
<dataset>/<split_family>/<method>/<framing>/fold_N/
    code_correctness_response_avg_diff.pt   # the vector(s), torch tensors
    code_correctness_sequence_last_diff.pt
    metadata.json                           # model, fold, method, framing, etc.
```

- **method**: how the vector was extracted
  - `pca` — the technique used in the ICSE 2026 paper
  - `mean_diff` — Anthropic-style mean of differences (persona-vectors approach)
  - `random` — random baseline for sanity checks
- **framing**: how the prompt was set up when collecting hidden states
  - `repe` — neutral framing (what the ICSE paper used)
  - `function` — frames correctness as a behavior/function (the "persona" style)

Each `fold_N/` corresponds to the fit set of the matching fold in `../dataset/`.

## What's currently here

This is the state as of when you're starting. Treat it as "what we have"; filling gaps is part of the work.

**ICSE26_DS — `icse26_bigcodebench/icse26/` and `icse26_humaneval/icse26/`:**

| method      | framing    | bigcodebench       | humaneval         |
|-------------|------------|--------------------|-------------------|
| `pca`       | `repe`     | all 10 folds       | all 10 folds      |
| `mean_diff` | `repe`     | all 10 folds       | all 10 folds      |
| `mean_diff` | `function` | folds 0, 1 only    | **missing**       |
| `random`    | `repe`     | all 10 folds       | all 10 folds      |

**Mutation_DS — `bigcodebench/s*/`:** for each split family (`s1_baseline`, `s2_test_confound`, …):

| method      | framing    | coverage         |
|-------------|------------|------------------|
| `mean_diff` | `function` | folds 0, 1 only  |
| `mean_diff` | `neutral`  | folds 0, 1 only  |
| `pca`       | `function` | folds 0, 1 only  |

So: ICSE data has full coverage for `pca/repe` and `mean_diff/repe`; everything involving the `function` framing or Mutation_DS is partial. Generating the missing combinations is a reasonable warm-up task.

---

Ask if something looks wrong or unclear.
