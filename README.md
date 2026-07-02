# LLM Code Correctness

This repository accompanies our paper on the robustness of LLMs' internal representation of code correctness. It provides the datasets, pre-computed experimental results, and the full pipeline to reproduce every experiment.

## Contents

| Directory | Description |
|-----------|-------------|
| `dataset/` | Datasets used for fitting and evaluation |
| `hidden_states/` | Cached per-layer pooled activations |
| `vectors/` | Fitted correctness direction vectors for a selected subset of configurations |
| `results/` | Validation and test scores for the same subset of configurations as `vectors/` |
| `rq1_ablation_results/` | Comprehensive RQ1 scores: all configurations, in-domain and OOD fit sources |
| `rq2_mutants_only_bcb_based_ablation_results/` | Comprehensive RQ2 scores: mutant fit (BigCodeBench-based corpus) |
| `rq2_mutants_confounds_bcb_based_ablation_results/` | Comprehensive RQ2 scores: confound-augmented mutant fit (BigCodeBench-based corpus) |
| `rq2_mutants_only_he_based_ablation_results/` | Comprehensive RQ2 scores: mutant fit (HumanEval-based corpus) |
| `rq2_mutants_confounds_he_based_ablation_results/` | Comprehensive RQ2 scores: confound-augmented mutant fit (HumanEval-based corpus) |
| `src/` | Core library: hidden state extraction, representations, fitting, and direction projection |
| `scripts/` | Experiment scripts and Slurm launchers |

## Setup

### On an HPC cluster

```bash
make setup-hpc
```

Loads `python/3.11.3` and `cuda/12.2.0`, then creates `$SCRATCH/envs/<repo-name>/` and installs all dependencies including vLLM.

### Locally (e.g. your laptop)

```bash
make setup-env
```

Creates `.venv/` with Python 3.11 and installs `requirements.txt`.

None of the `make` targets below need the venv activated — they invoke the venv's Python directly. Activate it (`source .venv/bin/activate` locally, or `source $SCRATCH/envs/<repo-name>/bin/activate` on HPC) only for ad-hoc Python work.

## Downloading data and results

All datasets and experimental results are hosted on figshare, as they are too large for git. Download the archives you need from **https://figshare.com/s/4dccae927dbe7772d2a8**, copy them to the repo root on HPC, and extract each one with:

```bash
tar -xf <archive.tar.zst>
```

The available archives are described below.

### Datasets — `dataset.tar.zst`

Extracts to `dataset/`. Contains all datasets used for fitting and evaluation:

- `dataset/icse26/{bigcodebench,humaneval,mbpp,synthetic}/` — the primary benchmarks, pre-split into 10 cross-validation folds with `fit`/`validate`/`test` splits. BigCodeBench and HumanEval are the main evaluation targets; MBPP and synthetic are used as out-of-distribution fit sources.
- `dataset/mutants_only_bcb_based/`, `dataset/mutants_only_he_based/` — mutant corpora derived from BigCodeBench and HumanEval respectively, used for RQ2 mutant-fit experiments.
- `dataset/mutants_confounds_bcb_based/`, `dataset/mutants_confounds_he_based/` — same corpora augmented with confounding variants.
- `dataset/bigcodebench/` — mutation dataset splits (not relevant for now).

See [dataset/README.md](dataset/README.md) for a full description.

### Hidden states — `hidden_states.tar.zst`

Extracts to `hidden_states/`. Contains cached per-layer pooled activations for all four models across all dataset splits, representations, and framings. Download this if you want to run Stage 2 (fit and score) without repeating the GPU-heavy Stage 1 extraction.

### Experimental results

The result archives differ in scope. `vectors/` and `results/` cover **a selected subset** of configurations explored during development. The `rq1` and `rq2` archives are **comprehensive** — they contain scores for every configuration reported in the paper.

| Archive | Directory | Scope | Contents |
|---------|-----------|-------|----------|
| `vectors.tar.zst` | `vectors/` | Subset | Fitted correctness directions (the direction vectors themselves, as `.pt` tensors) for a representative set of model/method/framing/fold combinations |
| `results.tar.zst` | `results/` | Subset | Per-layer validation and test accuracy scores corresponding to the same subset of configurations in `vectors/` |
| `rq1_ablation_results.tar.zst` | `rq1_ablation_results/` | Comprehensive | Accuracy scores across all ablation configurations for RQ1: every model × method × representation × framing combination, for both in-domain and OOD fit sources |
| `rq2_mutants_only_bcb_based_ablation_results.tar.zst` | `rq2_mutants_only_bcb_based_ablation_results/` | Comprehensive | All RQ2 scores when fitting and evaluating on the BigCodeBench-based mutant corpus as well as evaluating on held-out benchmarks |
| `rq2_mutants_confounds_bcb_based_ablation_results.tar.zst` | `rq2_mutants_confounds_bcb_based_ablation_results/` | Comprehensive | All RQ2 scores when fitting on the confound-augmented BigCodeBench-based mutant corpus |
| `rq2_mutants_only_he_based_ablation_results.tar.zst` | `rq2_mutants_only_he_based_ablation_results/` | Comprehensive | All RQ2 scores when fitting and evaluating on the HumanEval-based mutant corpus as well as evaluating on held-out benchmarks |
| `rq2_mutants_confounds_he_based_ablation_results.tar.zst` | `rq2_mutants_confounds_he_based_ablation_results/` | Comprehensive | All RQ2 scores when fitting on the confound-augmented HumanEval-based mutant corpus |

## Models

The four code LLMs used in the paper must be downloaded before running any experiment from scratch. This assumes you are on HPC:

```bash
make download-models
```

Models are saved to `$SCRATCH/hf-models` by default. To use a different path:

```bash
make download-models HF_HOME=/path/to/my/hf-cache
```

The four models are: Qwen2.5-Coder-7B-Instruct, OpenCoder-8B-Instruct, CodeLlama-7b-Instruct-hf, and Mistral-7B-Instruct-v0.3.

## Reproducing the experiments

The experiment pipeline has two stages. **Stage 1** (cache hidden states) is GPU-heavy and requires the models to be loaded. **Stage 2** (fit and score) is CPU-only and operates entirely on the Stage 1 outputs — it can run on a login or compute node without a GPU.

### Stage 1 — Cache hidden states

Runs each of the four models over each dataset split, pooling activations at every layer and saving them to `hidden_states/`. Completed bundles are skipped on re-run.

**Single configuration** (useful for testing):

```bash
python scripts/extract/cache_hidden_states.py \
  --models codellama/CodeLlama-7b-Instruct-hf \
  --dataset-source icse26 --benchmark bigcodebench \
  --protocol pca \
  --data-splits fit \
  --representations sequence_last
```

**Full Slurm sweep — one model:**

```bash
MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" \
  OUTPUT_DIR=/path/to/hidden_states \
  GPU_TYPE=a100 GPUS=1 \
  bash scripts/slurm/launchers/ablation_cache_sweep.sh
```

**Full Slurm sweep — all four models:**

```bash
GPU_TYPE=a100 GPUS=1 \
  PARTITION=<your-partition> ACCOUNT=<your-account> \
  bash scripts/slurm/launchers/ablation_cache_sweep.sh
```

Pass `--dry-run` to preview what would be submitted. Collaborators can split the model set with `MODELS="..."` pointing to a shared `OUTPUT_DIR` — per-model subdirectories never overlap, so no merge is needed.

### Stage 2 — Fit and score from cache

Fits a correctness direction for every combination of method (`pca`, `mean_diff`), representation, and framing, then scores it on the held-out test split. Results are written to `RESULTS_DIR` (default: `full_ablation_results/`).

**Single configuration:**

```bash
python scripts/experiments/fit_and_score_from_cache.py \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --representation sequence_last --method pca \
  --fit-framing raw --val-framing concept --test-framing concept \
  --fit-source in-domain --test-dataset bigcodebench \
  --results-dir full_ablation_results
```

**Full Slurm sweep — one model:**

```bash
MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" \
  bash scripts/slurm/launchers/ablation_fit_score_sweep.sh
```

**Full sweep — all models:**

```bash
bash scripts/slurm/launchers/ablation_fit_score_sweep.sh
```

Useful flags:
- `--paper-configs` — restricts to the configurations reported in the paper (fit framings `function`/`neutral`/`concept`, val framing = test framing).
- `--local` — runs inline instead of submitting to Slurm (useful on a single GPU node).
- `--dry-run` — prints commands without executing them.

#### RQ2: fitting on mutants

The RQ2 experiments use a different fit source. Two variants are supported via mode flags:

- `--mutants-only` — fit and validate on the mutants corpus; test on held-out benchmarks or on the mutant test split.
- `--mutants-confounds` — same, but the fit set is augmented with confounding variants (refactored-but-correct code) to test whether the model generalises beyond surface-level syntax errors.

Key environment variables for these modes:

| Variable | Values | Description |
|----------|--------|-------------|
| `MUTANT_STRATEGIES` | `paired weighted` | Which fit strategy to run. `paired` pairs each canonical solution with each of its mutants; `weighted` uses an operator-weighted subsample. |
| `MUTANTS_BENCHMARKS` | `bcb_based he_based` | Which mutant corpus to use — BigCodeBench-based or HumanEval-based. |
| `MUTANT_TEST_DATASETS` | `bigcodebench humaneval mutants_only mutants_confounds` | Which benchmark to score against at test time. |
| `RESULTS_DIR` | any path | Where to write the results (default: `full_ablation_results/`). |

Pass `--val-on-test-dataset` when `MUTANT_TEST_DATASETS` is a standard benchmark (`bigcodebench` or `humaneval`): this uses the in-domain validate split for layer selection instead of the mutant validate split.

**example mutants-only and mutants-confounds sweeps (HumanEval-based corpus, paper configurations):**

```bash
# paired strategy
RESULTS_DIR=rq2_mutants_only_he_based_ablation_results \
  MUTANT_STRATEGIES="paired" MUTANT_TEST_DATASETS=humaneval MUTANTS_BENCHMARKS=he_based \
  PARTITION=<your-partition> GPU_TYPE=<gpu-type> \
  bash scripts/slurm/launchers/ablation_fit_score_sweep.sh \
  --mutants-only --paper-configs --val-on-test-dataset

# weighted strategy
RESULTS_DIR=rq2_mutants_confounds_he_based_ablation_results \
  MUTANT_STRATEGIES="weighted" MUTANTS_BENCHMARKS=he_based \
  PARTITION=<your-partition> ACCOUNT=<your-account> \
  bash scripts/slurm/launchers/ablation_fit_score_sweep.sh \
  --mutants-confounds --paper-configs
```

Replace `bcb_based` for `MUTANTS_BENCHMARKS` to run the BigCodeBench-based corpus instead (the pre-computed results in `rq2_*_bcb_based_ablation_results/` used this corpus).


### Aggregate results into a matrix

After both stages complete, compile all per-fold scores into a single TSV for analysis:

```bash
python scripts/analysis/generate_ablation_matrix.py \
  --results-dir full_ablation_results \
  --rebuild \
  --output matrix.tsv
```

This writes one row per (model, method, representation, framing, fit-source, test-dataset) combination, with fold-averaged accuracy columns. Omit `--rebuild` to reuse the cached scan from a previous run.

## Code review viewer

A browser-based UI for inspecting the JSONL datasets without reading raw JSON lives in [`viewer_platform/`](viewer_platform/). See [viewer_platform/README.md](viewer_platform/README.md) for setup and deployment instructions.

## Data generation (advanced)

The datasets in `dataset/` were produced by a mutation and refactoring pipeline. You only need this section if you want to regenerate or extend the datasets.

**BigCodeBench submodule** (required for the data generation scripts):

```bash
make submodules
```

**BigCodeBench evaluation container** (required to validate mutants and confounds against test cases):

```bash
make pull-eval-image
```

This downloads the `bigcodebench/bigcodebench-evaluate` Docker image as a Singularity `.sif` to `$SCRATCH/containers/`. Override the path with `make pull-eval-image SIF_PATH=/my/path.sif`.
