#!/bin/bash
# Submit the ablation fit+score sweep (stage 2).
#
# For each model, submits one ablation_fit_score.sbatch job per dataset-source
# spec needed by the ablation grid (same partition as stage 1):
#   - in-domain: bigcodebench, humaneval                  (2 specs)
#   - OOD:       (mbpp, synthetic) x (bigcodebench, humaneval)  (4 specs)
# 4 models x 6 specs = 24 jobs. Each job internally loops over the
# (representation, method, fit-framing, val-framing) combinations its spec
# needs (2 x 2 x 3 x 2 = 24 combos per job).
#
# Splitting across collaborators: results are keyed by model first
# (ablation_results/<safe_model>/...), so two people can each run a disjoint
# subset of MODELS against the SAME shared RESULTS_DIR with zero path
# collisions -- no merge step needed afterward.
#
# Usage:
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh [--dry-run]
#   MODELS="model/a model/b" bash scripts/slurm/launchers/ablation_fit_score_sweep.sh
#   MODEL="Qwen/..." FIT_SOURCE=in-domain TEST_DATASET=bigcodebench \
#     bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --local
#
# By default, skips combinations whose results.json files are already complete
# (--skip-existing). Pass --no-skip-existing to force a full re-run.

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

BENCHMARKS=("bigcodebench" "humaneval")
OOD_SOURCES=("mbpp" "synthetic")
TEST_DATASETS=("bigcodebench" "humaneval")

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
SKIP_EXISTING=1
FIT_SCORE_SCRIPT="${REPO_ROOT}/scripts/experiments/fit_and_score_from_cache.py"
for arg in "$@"; do
    case "${arg}" in
        --local) MODE="local" ;;
        --dry-run) DRY_RUN=1 ;;
        --skip-existing) SKIP_EXISTING=1 ;;
        --no-skip-existing) SKIP_EXISTING=0 ;;
        *)
            echo "Unknown option: ${arg}" >&2
            echo "Usage: $0 [--local | --dry-run | --skip-existing | --no-skip-existing]" >&2
            exit 1
            ;;
    esac
done

submit() {
    local job_name="$1"; shift
    local -a env_vars=("$@")
    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a sbatch_args=(
        --job-name="${job_name}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]] && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]     && sbatch_args+=(--qos="${QOS}")

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    echo "${full_cmd[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${full_cmd[@]}"
    fi
}

run_sweep_for_spec() {
    local model="$1"
    local fit_source="$2"
    local test_dataset="$3"

    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local REPRESENTATIONS=("response_avg" "sequence_last")
    local METHODS=("mean_diff" "pca")
    local FIT_FRAMINGS=("function" "neutral" "concept")
    local VAL_FRAMINGS=("neutral" "concept")

    local n=0
    local REP METHOD FIT_FRAMING VAL_FRAMING
    for REP in "${REPRESENTATIONS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for FIT_FRAMING in "${FIT_FRAMINGS[@]}"; do
                for VAL_FRAMING in "${VAL_FRAMINGS[@]}"; do
                    n=$((n + 1))
                    local -a cmd=(python -u "${FIT_SCORE_SCRIPT}"
                        --model "${model}"
                        --representation "${REP}"
                        --method "${METHOD}"
                        --fit-framing "${FIT_FRAMING}"
                        --val-framing "${VAL_FRAMING}"
                        --fit-source "${fit_source}"
                        --test-dataset "${test_dataset}"
                        --hidden-states-root "${HIDDEN_STATES_ROOT}"
                        --results-dir "${RESULTS_DIR}"
                        --data-root "${DATA_ROOT}")
                    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
                        cmd+=(--skip-existing)
                    fi
                    echo "[${n}] ${cmd[*]}"
                    if [[ "${DRY_RUN}" -eq 0 ]]; then
                        "${cmd[@]}"
                    fi
                done
            done
        done
    done
    echo "Model ${model} fit_source=${fit_source} test_dataset=${test_dataset}: ${n} combinations"
}

run_sweep_for_model() {
    local model="$1"
    local benchmark ood test
    for benchmark in "${BENCHMARKS[@]}"; do
        run_sweep_for_spec "${model}" "in-domain" "${benchmark}"
    done
    for ood in "${OOD_SOURCES[@]}"; do
        for test in "${TEST_DATASETS[@]}"; do
            run_sweep_for_spec "${model}" "${ood}" "${test}"
        done
    done
}

run_sweep() {
    if [[ -n "${MODEL:-}" && -n "${FIT_SOURCE:-}" && -n "${TEST_DATASET:-}" ]]; then
        run_sweep_for_spec "${MODEL}" "${FIT_SOURCE}" "${TEST_DATASET}"
        return
    fi

    if [[ -n "${MODEL:-}" ]]; then
        run_sweep_for_model "${MODEL}"
        return
    fi

    local model
    for model in "${MODELS[@]}"; do
        run_sweep_for_model "${model}"
    done
}

common_export_vars() {
    local model="$1"
    printf '%s\n' \
        "MODEL=${model}" \
        "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}" \
        "RESULTS_DIR=${RESULTS_DIR}" \
        "DATA_ROOT=${DATA_ROOT}" \
        "ENV_DIR=${ENV_DIR}" \
        "PYTHON_MODULE=${PYTHON_MODULE}" \
        "REPO_ROOT=${REPO_ROOT}" \
        "SKIP_EXISTING=${SKIP_EXISTING}"
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
        safe_model="${model//\//__}"
        mapfile -t base_vars < <(common_export_vars "${model}")
        for benchmark in "${BENCHMARKS[@]}"; do
            submit "ablation_fit_score_id_${benchmark}_${safe_model}" \
                "${base_vars[@]}" \
                "FIT_SOURCE=in-domain" "TEST_DATASET=${benchmark}"
        done
        for ood in "${OOD_SOURCES[@]}"; do
            for test in "${TEST_DATASETS[@]}"; do
                submit "ablation_fit_score_ood_${ood}_${test}_${safe_model}" \
                    "${base_vars[@]}" \
                    "FIT_SOURCE=${ood}" "TEST_DATASET=${test}"
            done
        done
    done
else
    run_sweep
fi
