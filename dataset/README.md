# Datasets

This folder has the two datasets we'll use: **ICSE26_DS** and **Mutation_DS**.

## ICSE26_DS — `icse26/`

The data from the ICSE 2026 paper. The benchmarks you'll work with are:

- `icse26/bigcodebench/`
- `icse26/humaneval/`

(`icse26/mbpp/` and `icse26/synthetic/` are also here and used as out-of-distribution fit sets in some experiments.)

Each benchmark is a flat 10-fold split:

```
icse26/bigcodebench/
└── fold_0/ ... fold_9/
    ├── fit.jsonl        # train (used to learn the reading vector)
    ├── validate.jsonl   # used to pick the layer
    └── test.jsonl       # held-out evaluation
```

One task per line. Each record has an `implementations` list; each entry has a `code` string, a `kind` (`"canonical"` or `"pre-generated"`), and a `label` (`1` = correct, `0` = incorrect).

## Mutation_DS — `bigcodebench/`

The error-specific mutation + refactoring dataset. The two things to know:

- **`bigcodebench/frozen.jsonl`** (or its compressed `.zst`) — the full aggregated dataset. This is what "Mutation_DS" refers to. Each line is one task with its original/correct code, its mutants, and the LLM-refactored confounding variants.
- **`bigcodebench/phase2/sN_<name>/`** — pre-built splits over `frozen.jsonl` (different ways of partitioning the data into fit/validate/test). For Phase 2 (the analysis phase) start with one of these splits rather than the full file. Most relevant ones:
  - `s1_baseline/` — straightforward random split, good starting point.
  - `s2_test_confound/`, `s2b_val_test_confound/` — test sets focused on the confounding variants.
  - `s3_holdout_*/`, `s8_single_*/` — splits centered on one bug/operator type at a time.

### Working on Phase 1 (data generation)?

Look in `bigcodebench/intermediate/`. That's the output of the mutation + refactoring pipeline before it gets aggregated into `frozen.jsonl`:

- `tasks.jsonl` → original tasks
- `dropped_tasks.jsonl` → tasks discarded right away because the canonical/reference solution failed the test cases in our environment
- `mutants.jsonl`, `validated_mutants.jsonl` → mutants before/after running tests
- `confound_candidates.jsonl`, `validated_confounded.jsonl` → refactored variants before/after validation

If you're auditing or editing the pipeline, this is where the evidence lives.

## Mutants-only ablation — `mutants_only_bcb_based/`, `mutants_only_he_based/`

Two benchmark-specific corpora, same internal layout: `mutants_only_bcb_based/`
(BigCodeBench-derived mutants, the original/default corpus) and
`mutants_only_he_based/` (HumanEval-derived mutants). Pass `--mutants-benchmark
bcb_based|he_based` to `cache_hidden_states.py` / `fit_and_score_from_cache.py`
to pick one (default: `bcb_based`). `mutants_confounds_bcb_based/` /
`mutants_confounds_he_based/` follow the same convention.

Strategy subfolders for hidden-state caching / scoring (uniform across both
benchmarks):

- `paired/` — one canonical+mutant row per mutant (large fit/val sets).
- `weighted/` — operator-weighted subsample for fit/val and the **canonical mutant test** split (3 mutants per task).

`paired/` has only `fit.jsonl` and `validate.jsonl`. When `--test-dataset mutants_only`, scoring always loads test from `mutants_only_<benchmark>/weighted/` regardless of fit strategy.

Build via `scripts/dataset/limit_mutants_only_impls.py` (weighted) and `scripts/dataset/build_mutants_only.py`. Operator frequencies: `scripts/analysis/mutation_operator_frequencies.py`.

At score time, `test_dataset` selects the test benchmark:

- `bigcodebench` or `humaneval` — canonical test from `icse26/<benchmark>/id/test/...`
- `mutants_only` — mutant test from `mutants_only_<benchmark>/weighted/` (when fitting on `paired` or `weighted`), where `<benchmark>` is whatever `--mutants-benchmark` selected for fit/validate.

**Cache layout:** `hidden_states/<model>/mutants_only_<benchmark>/<strategy>/{fit,validate,test}/<framing>/fold_<N>/` — no `benchmark/` or `id/` segment. Test caches exist only under `mutants_only_<benchmark>/weighted/`.

**Migrating legacy caches** (manual): `S2-M-pair` fit/val → `mutants_only_bcb_based/paired` fit/val; mutant test caches → `mutants_only_bcb_based/weighted/test` only.

---

If anything is unclear or seems out of date, please ask.
