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

## Regenerating the vectors/results (secondary)
You don't need to do this. Just added information. The available `vectors/` and `results/` were produced
by the experiments in `scripts/experiments/`. The code under
`src/{framings, hidden_states, representations, vectors}` implements logic regarding representations / fit
/ monitoring. (Re)Generation is GPU-heavy. It depends on the GPU you request or the HPC assigns but expect at least 10 minutes for a fit + validate + test (monitor) loop.
