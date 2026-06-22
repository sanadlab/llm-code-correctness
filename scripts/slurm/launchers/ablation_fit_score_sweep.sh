#!/bin/bash
# Stage 2 of the ablation sweep: fit + score every combination from cached
# hidden states (scripts/experiments/fit_and_score_from_cache.py). Pure
# tensor math on already-cached activations -- no GPU/model load needed, so
# this runs as a plain CPU loop rather than a SLURM job. Run after
# ablation_cache_sweep.sh (stage 1) has populated the hidden-state cache.
#
# Grid: 4 models x 2 representations x 2 methods x 3 fit-framings x
#       2 val/test-framings x (in-domain x 2 datasets + ood x 2 sources x
#       2 datasets) = 4*2*2*3*2*6 = 576 combinations.
#
# Usage:
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "infly/OpenCoder-8B-Instruct"
    "codellama/CodeLlama-7b-Instruct-hf"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
REPRESENTATIONS=("response_avg" "sequence_last")
METHODS=("mean_diff" "pca")
FIT_FRAMINGS=("function" "neutral" "concept")
VAL_FRAMINGS=("neutral" "concept")
TEST_DATASETS=("bigcodebench" "humaneval")
FIT_SOURCES=("in-domain" "mbpp" "synthetic")

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-hidden_states}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

n=0
for MODEL in "${MODELS[@]}"; do
    for REP in "${REPRESENTATIONS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for FIT_FRAMING in "${FIT_FRAMINGS[@]}"; do
                for VAL_FRAMING in "${VAL_FRAMINGS[@]}"; do
                    for FIT_SOURCE in "${FIT_SOURCES[@]}"; do
                        for TEST_DATASET in "${TEST_DATASETS[@]}"; do
                            n=$((n + 1))
                            cmd=(python -u scripts/experiments/fit_and_score_from_cache.py
                                --model "${MODEL}"
                                --representation "${REP}"
                                --method "${METHOD}"
                                --fit-framing "${FIT_FRAMING}"
                                --val-framing "${VAL_FRAMING}"
                                --fit-source "${FIT_SOURCE}"
                                --test-dataset "${TEST_DATASET}"
                                --hidden-states-root "${HIDDEN_STATES_ROOT}"
                                --results-dir "${RESULTS_DIR}"
                                --data-root "${DATA_ROOT}")
                            echo "[${n}] ${cmd[*]}"
                            if [[ "${DRY_RUN}" -eq 0 ]]; then
                                "${cmd[@]}"
                            fi
                        done
                    done
                done
            done
        done
    done
done
echo "Total combinations: ${n}"
