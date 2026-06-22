#!/bin/bash
# Submit the ablation fit+score sweep (stage 2).
#
# For each model, submits one ablation_fit_score.sbatch job that loops over
# every (representation, method, fit-framing, val-framing, fit-source,
# test-dataset) combination for that model: 144 combos per model, 4 models
# = 576 total.
#
# Splitting across collaborators: results are keyed by model first
# (ablation_results/<safe_model>/...), so two people can each run a disjoint
# subset of MODELS against the SAME shared RESULTS_DIR with zero path
# collisions -- no merge step needed afterward.
#
# Usage:
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh [--dry-run]
#   MODELS="model/a model/b" bash scripts/slurm/launchers/ablation_fit_score_sweep.sh
#   MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/ablation_fit_score.sbatch"

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS <<< "${MODELS}"
else
    MODELS=(
        "Qwen/Qwen2.5-Coder-7B-Instruct"
        "infly/OpenCoder-8B-Instruct"
        "codellama/CodeLlama-7b-Instruct-hf"
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
fi

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

submit() {
    local model="$1"
    local safe_model="${model//\//__}"

    local -a sbatch_args=(
        --job-name="ablation_fit_score_${safe_model}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]] && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]     && sbatch_args+=(--qos="${QOS}")

    local -a env_vars=(
        "MODEL=${model}"
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

run_sweep_for_model() {
    local model="$1"
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local REPRESENTATIONS=("response_avg" "sequence_last")
    local METHODS=("mean_diff" "pca")
    local FIT_FRAMINGS=("function" "neutral" "concept")
    local VAL_FRAMINGS=("neutral" "concept")
    local TEST_DATASETS=("bigcodebench" "humaneval")
    local FIT_SOURCES=("in-domain" "mbpp" "synthetic")

    local n=0
    local REP METHOD FIT_FRAMING VAL_FRAMING FIT_SOURCE TEST_DATASET
    for REP in "${REPRESENTATIONS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for FIT_FRAMING in "${FIT_FRAMINGS[@]}"; do
                for VAL_FRAMING in "${VAL_FRAMINGS[@]}"; do
                    for FIT_SOURCE in "${FIT_SOURCES[@]}"; do
                        for TEST_DATASET in "${TEST_DATASETS[@]}"; do
                            n=$((n + 1))
                            local -a cmd=(python -u scripts/experiments/fit_and_score_from_cache.py
                                --model "${model}"
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
    echo "Model ${model}: ${n} combinations"
}

run_sweep() {
    if [[ -n "${MODEL:-}" ]]; then
        run_sweep_for_model "${MODEL}"
        return
    fi

    for model in "${MODELS[@]}"; do
        run_sweep_for_model "${model}"
    done
}

if [[ "${MODE}" == "submit" ]]; then
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

    for model in "${MODELS[@]}"; do
        submit "${model}"
    done
else
    run_sweep
fi
