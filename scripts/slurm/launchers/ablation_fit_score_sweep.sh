#!/bin/bash
# Stage 2 of the ablation sweep: fit + score every combination from cached
# hidden states (scripts/experiments/fit_and_score_from_cache.py). Pure
# tensor math on already-cached activations -- no GPU/model load needed.
#
# Grid: 4 models x 2 representations x 2 methods x 3 fit-framings x
#       2 val/test-framings x (in-domain x 2 datasets + ood x 2 sources x
#       2 datasets) = 4*2*2*3*2*6 = 576 combinations.
#
# Usage:
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh            # submit SLURM job
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --local    # run loop directly
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --dry-run  # print sbatch command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/ablation_fit_score.sbatch"

PARTITION="${PARTITION:-compute}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-04:00:00}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.3}"

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${ENV_DIR:-${SCRATCH}/envs/llm-code-correctness}"
else
    ENV_DIR="${ENV_DIR:-}"
fi

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-hidden_states}"
RESULTS_DIR="${RESULTS_DIR:-ablation_results}"
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"

MODE="submit"
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --local) MODE="local" ;;
        --dry-run) DRY_RUN=1 ;;
        *)
            echo "Unknown option: ${arg}" >&2
            echo "Usage: $0 [--local | --dry-run]" >&2
            exit 1
            ;;
    esac
done

submit_job() {
    if [[ -z "${SCRATCH:-}" ]]; then
        echo "Error: SCRATCH is not set." >&2
        exit 1
    fi
    if [[ -z "${ENV_DIR}" ]]; then
        echo "Error: ENV_DIR is not set." >&2
        exit 1
    fi

    mkdir -p "${REPO_ROOT}/logs/slurm"
    cd "${REPO_ROOT}"

    local -a sbatch_args=(
        --job-name="ablation_fit_score"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]] && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]     && sbatch_args+=(--qos="${QOS}")

    local -a env_vars=(
        "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
        "RESULTS_DIR=${RESULTS_DIR}"
        "DATA_ROOT=${DATA_ROOT}"
        "ENV_DIR=${ENV_DIR}"
        "PYTHON_MODULE=${PYTHON_MODULE}"
        "REPO_ROOT=${REPO_ROOT}"
    )
    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    echo "${full_cmd[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${full_cmd[@]}"
    fi
}

run_sweep() {
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local MODELS=(
        "Qwen/Qwen2.5-Coder-7B-Instruct"
        "infly/OpenCoder-8B-Instruct"
        "codellama/CodeLlama-7b-Instruct-hf"
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    local REPRESENTATIONS=("response_avg" "sequence_last")
    local METHODS=("mean_diff" "pca")
    local FIT_FRAMINGS=("function" "neutral" "concept")
    local VAL_FRAMINGS=("neutral" "concept")
    local TEST_DATASETS=("bigcodebench" "humaneval")
    local FIT_SOURCES=("in-domain" "mbpp" "synthetic")

    local n=0
    local MODEL REP METHOD FIT_FRAMING VAL_FRAMING FIT_SOURCE TEST_DATASET
    for MODEL in "${MODELS[@]}"; do
        for REP in "${REPRESENTATIONS[@]}"; do
            for METHOD in "${METHODS[@]}"; do
                for FIT_FRAMING in "${FIT_FRAMINGS[@]}"; do
                    for VAL_FRAMING in "${VAL_FRAMINGS[@]}"; do
                        for FIT_SOURCE in "${FIT_SOURCES[@]}"; do
                            for TEST_DATASET in "${TEST_DATASETS[@]}"; do
                                n=$((n + 1))
                                local -a cmd=(python -u scripts/experiments/fit_and_score_from_cache.py
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
}

if [[ "${MODE}" == "submit" ]]; then
    submit_job
else
    run_sweep
fi
