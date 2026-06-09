# LLM Code Correctness

This is
the starting point for the Summer 2026 project. This repo has:
- Pre-computed correctness vectors
- Per-layer monitoring results
- datasets used to produce vectors and results
- prototype data generation pipeline

## Setup

### HPC

```bash
make setup-hpc
```

Loads the `python/3.11.3` and `cuda/12.2.0` modules, then creates
`$SCRATCH/envs/<repo-name>/` and installs `requirements.txt` plus
`requirements-hpc.txt` (vLLM and friends).

### Local (e.g. your laptop)

```bash
make setup-env
```

Creates `.venv/` with Python 3.11 and installs `requirements.txt`.

None of the `make` targets below need the venv activated — they invoke the
venv's Python directly. Activate it (`source .venv/bin/activate` locally, or
`source $SCRATCH/envs/<repo-name>/bin/activate` on HPC) only for ad-hoc Python
work.

## Initialize submodules

```bash
make submodules
```

Sets up and installs `third_party/bigcodebench` into the active venv (most likely only needed for phase 1, but please do it nonetheless).

## Pull the BigCodeBench evaluation container (phase 1 only)

The phase 1 validation steps (`validate_ground_truth`, `validate_mutants`,
`validate_confounds`) run BigCodeBench inside a Singularity container. Pull
its image before running those steps:

```bash
make pull-eval-image
```

This downloads `docker://bigcodebench/bigcodebench-evaluate:latest` to
`$SCRATCH/containers/bigcodebench-evaluate.sif`, which the path referenced by
`configs/phase1/bigcodebench.yaml` (`validate.sif_path`). If you need to override the
destination, do: `make pull-eval-image SIF_PATH=/my/path.sif` (don't forget to update any configs after).

## Decompress datasets

```bash
make decompress
```

Extracts:

- `dataset/icse26.tar.zst` → `dataset/icse26/{bigcodebench,humaneval,mbpp,synthetic}/`
- `dataset/bigcodebench/phase2.tar.zst` → `dataset/bigcodebench/phase2/<split>/` (the `Mutation_DS` splits)
- `dataset/bigcodebench/intermediate/*.jsonl.zst` → `*.jsonl`

Archives are hidden from `git status` after extraction. Run `make compress` to
repack from the on-disk directories (also un-hides the archives so git sees the
updates).

## Download the four models

This assumes you'll be running models only on the HPC.

```bash
make download-models
```

Default stores to `$SCRATCH/hf-models`. Note that $SCRATCH resolves to your user's scratch path already, so there's no need for you to specify it. If you want to use a different directory:

```bash
make download-models HF_HOME=/path/to/my/hf-cache
```

## Download vectors and results

`vectors/` and `results/` are too large for the GitHub remote, so they are hosted on
figshare. Download them **manually via your browser**, then copy them to HPC:

1. Download both archives from: **https://figshare.com/s/4dccae927dbe7772d2a8**
2. Copy them to the repo root on HPC:
   ```bash
   scp vectors.tar.zst results.tar.zst <net-id>@jubail.abudhabi.nyu.edu:<path-to-repo>/
   ```
3. Unpack:
   ```bash
   make decompress-artifacts
   ```

If you need to repack after regenerating something: `make compress-artifacts`.

## WhAt’s iN ThE BoX?!?!
(title makes sense if you watched the movie [Seven](https://www.imdb.com/title/tt0114369/))

### Correctness vectors → `vectors/`

One directory per model, e.g. `vectors/mistralai__Mistral-7B-Instruct-v0.3/`.

Inside each model:

- `icse26_bigcodebench/`, `icse26_humaneval/` — vectors from **ICSE26_DS**
- `bigcodebench/` — vectors from **Mutation_DS**

Each leaf is `<dataset>/<split>/<method>/<framing>/fold_N/` and contains the
`.pt` tensors plus a `metadata.json`. See [vectors/README.md](vectors/README.md) for more details.

Available vectors: models × {ICSE26 PCA, ICSE26 mean-of-diffs, Mutation_DS PCA,
Mutation_DS mean-of-diffs}.

### Selected layers → `results/`

Same `<model>/<dataset>/<split>/...` layout as `vectors/`. Per-fold output:

- `phase3/<method>/<framing>/fold_N/monitoring/monitoring_results.json` — per-layer validation scores.
- `phase4/<method>/<framing>/fold_N/monitoring/monitoring_test_results.json` — the selected layer (`phase3_selected`) and test-set metrics.

See [results/README.md](results/README.md) for more details.

## Collect hidden states

Extract and cache per-layer hidden states for all four plan models on both
ICSE26 benchmarks (BigCodeBench and HumanEval). Run on a GPU node (interactive
session or Slurm); the full sweep is GPU-heavy and may take many hours.

**Prerequisites:** `make setup` (or `make setup-hpc`), `make decompress`,
`make download-models`, and `vectors/` unpacked (`make decompress-artifacts`) for
`--verify`.

```bash
make collect-hidden-states
```

Writes to `hidden_states/` (override with `HS_OUTPUT_DIR`). Completed bundles
are skipped on re-run; pass `HS_FORCE=1` to re-extract.

Partial sweeps:

```bash
make collect-hidden-states HS_MODELS="mistralai/Mistral-7B-Instruct-v0.3"
make collect-hidden-states HS_BENCHMARKS=humaneval
make collect-hidden-states HF_HOME=/path/to/my/hf-cache
```

See [hidden_states/STRUCTURE.md](hidden_states/STRUCTURE.md) for output layout.


## ICSE2026 tSNE (cache-only)

This script is selector-driven (no hardcoded stage mode). It reads:
- selected layer from `results/.../results.json`
- hidden states from `hidden_states/.../bundle.pt`

It does not run extraction or model inference.

```bash
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers selected \
  --seed 7 \
  --perplexity 30
```

Default output path format (dataset origin from `--split-family`):
- ICSE26: `visualizations/tsne/<model_safe>/icse26/<benchmark>/<bucket>/<image_name>.png`
- Mutation_DS (e.g. `s1_baseline`): `visualizations/tsne/<model_safe>/mutation_ds/<split-family>/<benchmark>/<bucket>/<image_name>.png`
- OOD (e.g. `ood_mbpp`): `visualizations/tsne/<model_safe>/ood/<split-family>/<benchmark>/...`

Where `<bucket>` is:
- `fold_<N>` for single-fold requests
- `fold_all` when `--folds all` and layers are not explicit custom lists
- `custom` for custom multi-fold and/or explicit multi-layer requests

Filenames use a compact pattern with no prefixes before the fold id:
- `fold_0_layer_12.png`
- `fold_0_layer_all.png`
- `fold_0_layer_12_13_15.png`


Per-task mode:
- enable with `--tasks all` or explicit `--tasks <task_id ...>`
- each figure is per fold + per task, with subplots across selected layers
- per-task tSNE uses `--per-task-perplexity` (default `2.0`)
- `--task-view-mode`:
  - `local` (default): fit tSNE on each task's 4 points
  - `global_subset`: fit tSNE on all fold points, then display only task points
- `--show-background`: with `global_subset`, also draw faint fold background points
  behind the 4 highlighted task points (requires `--tasks`)

Examples:

```bash
# one fold + selected layer (single figure)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers selected

# one fold + all layers (single layer-grid figure)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers all

# all folds + selected layer (single fold-grid figure)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds all \
  --layers selected

# all folds + all layers (one layer-grid figure per fold)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds all \
  --layers all

# custom folds + custom layers (stored under custom/)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 3 \
  --layers 12 13 15

# per-task view (all tasks in fold 0, all layers, uses per-task perplexity default=2)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers all \
  --tasks all

# per-task view (specific task ids)
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers 12 13 15 \
  --tasks BigCodeBench/208

# per-task view with global-fit + task-subset mode
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers all \
  --tasks BigCodeBench/208 \
  --task-view-mode global_subset

# same, with faint global background for manifold context
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds 0 \
  --layers all \
  --tasks BigCodeBench/208 \
  --task-view-mode global_subset \
  --show-background

```

Change `--method`, `--fit-framing`, `--representation`, `--val-framing`,
`--test-framing`, `--split-family`, and path roots to target other result/cache cells.

Mutation_DS example (`s1_baseline`):

```bash
python scripts/analysis/tsne.py \
  --benchmark bigcodebench \
  --split-family s1_baseline \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --method mean_diff \
  --fit-framing function \
  --representation response_avg \
  --val-framing neutral \
  --test-framing neutral \
  --folds 0 \
  --layers all
```

Per-task mode (`--tasks`) reads task outcomes from
`<split-dir>/<split-family>_<benchmark>.jsonl` (e.g.
`dataset/bigcodebench/phase2/s8_single_arithmetic/s8_single_arithmetic_bigcodebench.jsonl`),
not from the ICSE26 aggregate file.


## ICSE2026 PCA (cache-only)

This script is selector-driven (no hardcoded stage mode). It reads:
- selected layer from `results/.../results.json`
- hidden states from `hidden_states/.../bundle.pt`
- per-layer reading vectors from `vectors/.../direction.pt` (mean-diff direction, incorrect→correct)

It does not run extraction or model inference.

Two visualization approaches are available via `--approach`:
- **A** — standard 2D PCA on the points, with the reading vector drawn as an arrow from the origin; panel titles include `captured` (fraction of the direction lying in the PC1–PC2 plane).
- **B** — x-axis is projection onto the reading vector (correctness direction); y-axis is the top residual PC orthogonal to it.

Use `--normalize-vector` to unit-normalize the reading vector for projection scaling (encoded in output filenames).

Default output path format (dataset origin from `--split-family`):
- ICSE26: `visualizations/pca/<model_safe>/icse26/<benchmark>/<bucket>/<image_name>.png`
- Mutation_DS (e.g. `s1_baseline`): `visualizations/pca/<model_safe>/mutation_ds/<split-family>/<benchmark>/<bucket>/<image_name>.png`
- OOD (e.g. `ood_mbpp`): `visualizations/pca/<model_safe>/ood/<split-family>/<benchmark>/...`

Filenames append `_approach{A|B}_norm{True|False}` before `.png` so A/B and normalized/raw runs do not overwrite each other (e.g. `fold_0_task_379_layer_all_approachA_normFalse.png`).

Per-task mode (`--tasks`) uses the same `fold_<f>/tasks/<success|fail>/` layout as tSNE. PCA always fits locally on the plotted points (4 per task, or the full fold otherwise).

Example (Mutation_DS per-task, both approaches, all layers):

```bash
python scripts/analysis/pca.py \
  --benchmark bigcodebench \
  --split-family s8_single_arithmetic \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --method mean_diff \
  --fit-framing function \
  --representation response_avg \
  --val-framing neutral \
  --test-framing neutral \
  --folds 0 \
  --layers all \
  --tasks all \
  --approach both
```


## Regenerating the vectors/results (secondary)
You don't need to do this. Just added information. The available `vectors/` and `results/` were produced
by the experiments in `scripts/experiments/`. The code under
`src/{framings, hidden_states, representations, vectors}` implements logic regarding representations / fit
/ monitoring. (Re)Generation is GPU-heavy. It depends on the GPU you request or the HPC assigns but expect at least 10 minutes for a fit + validate + test (monitor) loop.
