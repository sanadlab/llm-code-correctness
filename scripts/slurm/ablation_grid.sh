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
# Dataset-source specs (5 per model when mutants included):
#   - in-domain:  test_dataset in {bigcodebench, humaneval}
#   - OOD:        fit_source in {mbpp, synthetic}  (fit/val cached once; no benchmark segment)
#   - mutants:    paired (fit/val) + weighted (fit/val/test), per ABLATION_MUTANTS_BENCHMARKS
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

ABLATION_MUTANTS_STRATEGIES=("paired" "weighted")
ABLATION_MUTANTS_TEST_DATASETS=("bigcodebench" "humaneval" "mutants_only")

# Mutants corpus benchmark axis: bcb_based is the default/legacy corpus;
# he_based is opt-in via MUTANTS_BENCHMARKS in the launchers.
ABLATION_MUTANTS_BENCHMARKS=("bcb_based" "he_based")

# mutants_confounds: paired + weighted only; test target uses mutants_confounds not mutants_only.
ABLATION_MUTANTS_CONFOUNDS_STRATEGIES=("paired" "weighted")
ABLATION_MUTANTS_CONFOUNDS_TEST_DATASETS=("bigcodebench" "humaneval" "mutants_confounds")

# Cached in every bundle.pt (stage 1 --representations).
ABLATION_REPRESENTATIONS=("response_avg" "sequence_last")

# Stage 1 caches fit/validate/test splits with these framings (see ablation_cache.sbatch).
# `raw` is the ICSE paper fit stimulus (no chat-template role markers). Stage 2 sweeps
# raw with all method/val/test combos except response_avg (no response slice on raw fit).
ABLATION_FIT_FRAMINGS=("function" "neutral" "concept" "raw")
ABLATION_VAL_FRAMINGS=("neutral" "concept")
ABLATION_TEST_FRAMINGS=("neutral" "concept")

# Stage 2 only (no extra caching needed).
ABLATION_METHODS=("mean_diff" "pca")

# Combos per dataset-source spec in stage 2 (full Cartesian product before skips):
#   |REP| x |METHODS| x |FIT| x |VAL| x |TEST| = 2 x 2 x 4 x 2 x 2 = 64
ABLATION_COMBOS_PER_SPEC=$(( \
    ${#ABLATION_REPRESENTATIONS[@]} \
    * ${#ABLATION_METHODS[@]} \
    * ${#ABLATION_FIT_FRAMINGS[@]} \
    * ${#ABLATION_VAL_FRAMINGS[@]} \
    * ${#ABLATION_TEST_FRAMINGS[@]} \
))

# Paper ablation subset (stage 2 --paper-configs): no raw fit; val framing must
# match test framing. Fit framing may differ from val/test.
ABLATION_PAPER_FIT_FRAMINGS=("function" "neutral" "concept")

# Stage 2 skips (fit_score sweeps call ablation_skip_irrelevant_combo):
#   - fit=raw with response_avg (raw stimulus has no response slice to pool).
#
# Combos per spec after skips: 64 - 8 (raw x response_avg x 2 methods x 2 val x 2 test) = 56.
# With --paper-configs: 2 x 2 x 3 x 2 (val=test pairs only) = 24.
ablation_skip_irrelevant_combo() {
    local rep="$1" method="$2" fit="$3" val="$4" test="$5"
    if [[ "${fit}" == "raw" && "${rep}" == "response_avg" ]]; then
        return 0
    fi
    return 1
}

ablation_skip_non_paper_combo() {
    local fit="$1" val="$2" test="$3"
    if [[ "${fit}" == "raw" || "${val}" != "${test}" ]]; then
        return 0
    fi
    return 1
}

ABLATION_RELEVANT_COMBOS_PER_SPEC=$(( \
    ABLATION_COMBOS_PER_SPEC \
    - ${#ABLATION_METHODS[@]} \
    * ${#ABLATION_VAL_FRAMINGS[@]} \
    * ${#ABLATION_TEST_FRAMINGS[@]} \
))
