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


## Layer quarter accuracy (cache-only)

For a given model and benchmark, reports how often each **quarter of layers**
correctly identifies the canonical implementation among four candidates (strict
1-of-4 ranking: the correct implementation must have the **strictly** highest
projection onto that fold's per-layer reading vector; ties count as wrong).

Reads only precomputed artifacts:
- hidden states from `hidden_states/.../bundle.pt` (`sample_ids`, `task_ids`,
  `labels`, `acts`)
- per-fold reading vectors from `vectors/.../direction.pt`
- fold count from `{splits_dir}/manifest.json` when using `--folds all`

**Layer binning:** layer indices are split into four contiguous quarters with
`np.array_split`. Layer 0 (embedding output) is **excluded** by default; pass
`--include-embedding` to include it.

Per-fold quarter accuracy pools over all (task, layer) pairs in that quarter.
The final table also reports macro mean and std across folds (equal fold weight).

Default output:
- `analysis/layer_quarter_accuracy/<model_safe>/<origin>/<benchmark>/quarter_accuracy.csv`
- `analysis/layer_quarter_accuracy/<model_safe>/<origin>/<benchmark>/quarter_accuracy.json`

Example (all folds, ICSE26 BigCodeBench):

```bash
python scripts/analysis/layer_quarter_accuracy.py \
  --benchmark bigcodebench \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --folds all \
  --method mean_diff \
  --fit-framing function \
  --representation response_avg \
  --test-framing neutral
```

Output table columns: `Q1` … `Q4`. Rows: one per fold, plus `mean-across-folds`
and `std-across-folds`.


## Per-layer strict accuracy plot (cache-only)

For a given benchmark/split-family and a set of models, plots **strict 1-of-4
test accuracy per layer** (ties wrong) with one line per model. Accuracy at a
layer is the mean over tasks of whether the canonical implementation scores
strictly highest when projecting the four implementations onto that layer's
reading vector.

Reads only precomputed artifacts:
- hidden states from `hidden_states/.../bundle.pt` (test split)
- per-fold reading vectors from `vectors/.../direction.pt`
- fold count from `{splits_dir}/manifest.json` when using `--folds all`

Layer 0 (embedding output) is **excluded** from the plot by default; pass
`--include-embedding` to include it.

Per-layer accuracy is first computed per fold, then macro-averaged across
folds for each model (equal fold weight).

Default output:
- `analysis/layer_accuracy_plot/<origin>/<benchmark>/layer_accuracy.png`
- `analysis/layer_accuracy_plot/<origin>/<benchmark>/layer_accuracy.json`

Example (all folds, ICSE26 BigCodeBench, two models):

```bash
python scripts/analysis/layer_accuracy_plot.py \
  --benchmark bigcodebench \
  --models mistralai/Mistral-7B-Instruct-v0.3 codellama/CodeLlama-7b-Instruct-hf \
  --split-family icse26 \
  --folds all \
  --method mean_diff \
  --fit-framing function \
  --representation response_avg \
  --test-framing neutral
```

The JSON dump contains the per-layer means and standard deviations for each
model, plus per-fold per-layer accuracies.


## Residual-PCA correlation (cache-only)

Tests **what the residual-PCA y-axis represents beyond correctness** by
correlating it against surface-level code metrics, per layer. For each task (a
set of `k` implementations, expected `k=4`: 1 correct + 3 incorrect) it builds a
per-task scatter where the x-axis is the fold-specific reading vector
(correctness direction) and the y-axis is the **top principal component of the
residuals** (hidden states with the reading-vector component projected out),
then correlates that y-axis against each surface metric.

Reads only precomputed artifacts (no model inference):
- hidden states from `hidden_states/.../bundle.pt` (`response_avg` acts; layer 0
  embedding is **excluded** by default, pass `--include-embedding` to keep it)
- per-fold reading vectors from `vectors/.../direction.pt`
- validation-selected layer from `results/.../results.json`
- implementation source code from the dataset fold JSONL (`fold_<k>/test.jsonl`)
- fold count from `manifest.json` (via `--folds all`)

**Surface metrics** (computed once per task and cached to `surface_metrics.csv`,
reused across all layers/folds/models; they do not depend on layer/fold/model):
- `levenshtein_to_correct` — Levenshtein distance of the implementation source to
  the task's correct implementation (correct = 0).
- `ast_dist_to_correct` — tree edit distance over `ast.parse` trees
  (`--ast-backend {apted,zss}`, default `apted`; correct = 0). Identifier renames
  count as edits unless `--ast-normalize-identifiers`. Tree edit distance is
  super-quadratic, so `--ast-max-nodes` (default 300) skips pathologically large
  ASTs (recorded as `nan`); unparseable implementations are also `nan`.
- `loc`, `cyclomatic` — source lines of code and McCabe cyclomatic complexity via
  `radon` (absolute per implementation; cyclomatic = max over functions).

**Stages** (`--stage {metrics,correlate,aggregate,plot,all}`, default `all`).
Each stage reads/writes on-disk artifacts so it can run standalone:
`metrics` -> `surface_metrics.csv`; `correlate` -> `correlation_raw.csv`;
`aggregate` -> `correlation_summary.csv/.json` + `correlation_selected_layer.csv`;
`plot` -> `layer_curves*.png`.

**Two-level aggregation (per-fold -> macro).** Because the reading vector is per
fold, a task's `r^2` is only defined relative to a fold's reading vector, so we
aggregate **within fold first, then across folds**:
- Level 1 (within fold): for fold *j*, layer L, mean `r^2` over fold *j*'s
  held-out tasks -> one per-fold-per-layer value.
- Level 2 (across folds): **macro-average** the per-fold means (equal weight per
  fold, because tasks recur unequally across folds) -> `macro_mean_r2 +/- macro_std_r2`.

Raw per-task `r^2` is never pooled across folds into a single average. Only `r^2`
is reported (the sign of the residual component, hence Pearson `r`, is arbitrary).
The null is the **exact permutation null** (`k!` permutations for `k<=7`, random
`--n-perms` sampling otherwise).

**Interpretation note (read against 1/3, not 0).** With `k=4` implementations a
metric with *no* real relationship to the y-axis still averages `r^2 ~= 1/3`
(analytic n=4 chance level). So the curves must be read against the dashed `1/3`
chance line, not against `0`: a layer near `0.33` means **no tracking**, and only
values **clearly above `0.33`** indicate the y-axis tracks that surface metric.

Default output (dataset origin from `--split-family`):
`analysis/residual_pca_correlation/<model_safe>/<origin>/<benchmark>/`
- `surface_metrics.csv` — per `(task, implementation)` metrics cache.
- `correlation_raw.csv` — one row per
  `(dataset, model, fold, layer, task_id, error_type, metric)` with
  `pearson_r, r_squared, null_r2_mean, p_value, v1_explained_var, n_points`.
- `correlation_summary.csv` / `.json` — macro-aggregated, one row per
  `(dataset, model, layer, error_type, metric)` with
  `macro_mean_r2, macro_std_r2, null_baseline (=0.333), n_folds, frac_tasks_above_null`.
- `correlation_selected_layer.csv` — at each fold's validation-selected layer
  (`results.json["selected_layer"]`), per-fold `r^2` per metric macro-averaged
  across folds (each fold contributes its own layer's value).
- `layer_curves.png` — macro-mean `r^2` vs layer, one line per metric, `+/-std`
  band, dashed chance line at `1/3`. For Mutation_DS, also `layer_curves_<error_type>.png`
  per error-type subset (error type derived from the mutation operator in the
  mutant sample ids).

Example (ICSE26 BigCodeBench, all folds):

```bash
python scripts/analysis/residual_pca_correlation.py \
  --benchmark bigcodebench \
  --model codellama/CodeLlama-7b-Instruct-hf \
  --split-family icse26 \
  --data-root dataset/icse26 \
  --folds all \
  --method mean_diff \
  --fit-framing function \
  --representation response_avg \
  --val-framing neutral \
  --test-framing neutral
```

Example (Mutation_DS `s8_single_arithmetic`, all folds):

```bash
python scripts/analysis/residual_pca_correlation.py \
  --benchmark bigcodebench \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --split-family s8_single_arithmetic \
  --data-root dataset/bigcodebench/phase2 \
  --folds all
```

Computing surface metrics (the tree-edit distance in particular) is the slow,
one-time step; it is cached in `surface_metrics.csv` and reused on subsequent
runs (pass `--surface-metrics-path` to reuse an existing cache, e.g. across models).


## Regenerating the vectors/results (secondary)
You don't need to do this. Just added information. The available `vectors/` and `results/` were produced
by the experiments in `scripts/experiments/`. The code under
`src/{framings, hidden_states, representations, vectors}` implements logic regarding representations / fit
/ monitoring. (Re)Generation is GPU-heavy. It depends on the GPU you request or the HPC assigns but expect at least 10 minutes for a fit + validate + test (monitor) loop.
