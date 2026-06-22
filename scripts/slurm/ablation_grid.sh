#!/bin/bash
# Shared ablation grid definition for stage 1 (cache) and stage 2 (fit+score).
#
# Stage 1 (ablation_cache.sbatch) caches hidden states for every combination of:
#   model x dataset-source spec x representation x split-framing
# where split-framing is chosen from the framing list appropriate to each data
# split (fit / validate / test).
#
# Stage 2 (ablation_fit_score_sweep.sh) reuses those caches and additionally
# sweeps method (mean_diff, pca). Every fit+score combo only loads bundles
# whose framings were cached in stage 1.
#
# Dataset-source specs (6 per model; same job partition in both stages):
#   - in-domain:  test_dataset in {bigcodebench, humaneval}
#   - OOD:        fit_source in {mbpp, synthetic} x test_dataset in {bigcodebench, humaneval}
#
# Do not edit one launcher without updating this file and the other stage.

ABLATION_MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "infly/OpenCoder-8B-Instruct"
    "codellama/CodeLlama-7b-Instruct-hf"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

ABLATION_BENCHMARKS=("bigcodebench" "humaneval")
ABLATION_OOD_SOURCES=("mbpp" "synthetic")
ABLATION_TEST_DATASETS=("bigcodebench" "humaneval")

# Cached in every bundle.pt (stage 1 --representations).
ABLATION_REPRESENTATIONS=("response_avg" "sequence_last")

# Stage 1 caches fit/validate/test splits with these framings (see ablation_cache.sbatch).
ABLATION_FIT_FRAMINGS=("function" "neutral" "concept")
ABLATION_VAL_FRAMINGS=("neutral" "concept")
ABLATION_TEST_FRAMINGS=("neutral" "concept")

# Stage 2 only (no extra caching needed).
ABLATION_METHODS=("mean_diff" "pca")

# Combos per dataset-source spec in stage 2:
#   |REP| x |METHODS| x |FIT| x |VAL| x |TEST| = 2 x 2 x 3 x 2 x 2 = 48
ABLATION_COMBOS_PER_SPEC=$(( \
    ${#ABLATION_REPRESENTATIONS[@]} \
    * ${#ABLATION_METHODS[@]} \
    * ${#ABLATION_FIT_FRAMINGS[@]} \
    * ${#ABLATION_VAL_FRAMINGS[@]} \
    * ${#ABLATION_TEST_FRAMINGS[@]} \
))
