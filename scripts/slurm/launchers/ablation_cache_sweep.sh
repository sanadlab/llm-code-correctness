#!/bin/bash
# Submit the full ablation hidden-state caching sweep (stage 1).
#
# For each model, submits one ablation_cache.sbatch job per dataset-source
# spec (see scripts/slurm/ablation_grid.sh). Stage 2 uses the same 6 specs.
#
# Note: OOD test bundles are keyed by split_family=ood_<source>, so the
# outer test extraction is repeated once per ood-source even though the
# underlying test data/framing is identical between mbpp- and
# synthetic-sourced cells (matches cache_hidden_states.py's existing
# on-disk layout; not deduplicated here).
#
# Splitting across collaborators: bundle_output_dir() keys every cache path
# by model first (hidden_states/<safe_model>/...), so two people can each
# run a disjoint subset of MODELS against the SAME shared OUTPUT_DIR with
# zero path collisions -- no merge step needed afterward. Each person's
# account independently caps at its own GPU limit (e.g. 3 GPUs), so two
# people = effectively double the concurrent throughput. Example:
#   # person A
#   MODELS="Qwen/Qwen2.5-Coder-7B-Instruct infly/OpenCoder-8B-Instruct" \
#     OUTPUT_DIR=/scratch/shared/ablation/hidden_states \
#     bash scripts/slurm/launchers/ablation_cache_sweep.sh
#   # person B
#   MODELS="codellama/CodeLlama-7b-Instruct-hf mistralai/Mistral-7B-Instruct-v0.3" \
#     OUTPUT_DIR=/scratch/shared/ablation/hidden_states \
#     bash scripts/slurm/launchers/ablation_cache_sweep.sh
# If you don't have a shared filesystem, leave OUTPUT_DIR at its per-clone
# default and rsync/merge the two hidden_states/ trees afterward -- the
# per-model top-level directories never overlap, so a plain merge is safe.
#
# Usage:
#   bash scripts/slurm/launchers/ablation_cache_sweep.sh [--dry-run]
#   MODELS="model/a model/b" bash scripts/slurm/launchers/ablation_cache_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/ablation_cache.sbatch"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../ablation_grid.sh
source "${SCRIPT_DIR}/../ablation_grid.sh"

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS <<< "${MODELS}"
else
    MODELS=("${ABLATION_MODELS[@]}")
fi
OUTPUT_DIR="${OUTPUT_DIR:-hidden_states}"
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"
BENCHMARKS=("${ABLATION_BENCHMARKS[@]}")
OOD_SOURCES=("${ABLATION_OOD_SOURCES[@]}")
TEST_DATASETS=("${ABLATION_TEST_DATASETS[@]}")

PARTITION="${PARTITION:-nvidia}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-04:00:00}"

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.3}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2.0}"
if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${ENV_DIR:-${SCRATCH}/envs/llm-code-correctness}"
    HF_HOME="${HF_HOME:-${SCRATCH}/hf-models}"
else
    ENV_DIR="${ENV_DIR:-}"
    HF_HOME="${HF_HOME:-}"
fi

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2; exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

submit() {
    local job_name="$1"; shift
    local -a env_vars=("$@")
    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a sbatch_args=(
        --job-name="${job_name}"
        --partition="${PARTITION}"
        --gpus="${GPUS}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]]    && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]        && sbatch_args+=(--qos="${QOS}")
    [[ -n "${CONSTRAINT}" ]] && sbatch_args+=(--constraint="${CONSTRAINT}")

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    echo "${full_cmd[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${full_cmd[@]}"
    fi
}

for MODEL in "${MODELS[@]}"; do
    safe_model="${MODEL//\//__}"
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        submit "ablation_cache_id_${BENCHMARK}_${safe_model}" \
            "MODEL=${MODEL}" "DATASET_SOURCE=icse26" "BENCHMARK=${BENCHMARK}" \
            "ENV_DIR=${ENV_DIR}" "HF_HOME=${HF_HOME}" \
            "PYTHON_MODULE=${PYTHON_MODULE}" "CUDA_MODULE=${CUDA_MODULE}" \
            "REPO_ROOT=${REPO_ROOT}" "OUTPUT_DIR=${OUTPUT_DIR}" "DATA_ROOT=${DATA_ROOT}"
    done
    for OOD in "${OOD_SOURCES[@]}"; do
        for TEST in "${TEST_DATASETS[@]}"; do
            submit "ablation_cache_ood_${OOD}_${TEST}_${safe_model}" \
                "MODEL=${MODEL}" "DATASET_SOURCE=ood" "OOD_SOURCE=${OOD}" "TEST_DATASET=${TEST}" \
                "ENV_DIR=${ENV_DIR}" "HF_HOME=${HF_HOME}" \
                "PYTHON_MODULE=${PYTHON_MODULE}" "CUDA_MODULE=${CUDA_MODULE}" \
                "REPO_ROOT=${REPO_ROOT}" "OUTPUT_DIR=${OUTPUT_DIR}" "DATA_ROOT=${DATA_ROOT}"
        done
    done
done
