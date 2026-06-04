#!/bin/bash
#SBATCH --job-name=phase1_val_mutants
#SBATCH --output=logs/slurm/phase1_val_mutants.%j.out
#SBATCH --error=logs/slurm/phase1_val_mutants.%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG=${CONFIG:-configs/phase1/bigcodebench.yaml}
STEP=${STEP:-validate_mutants}
PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}
FORCE=${FORCE:-0}
VALIDATE_PARALLEL=${VALIDATE_PARALLEL:-}
VALIDATE_NUM_SHARDS=${VALIDATE_NUM_SHARDS:-1}
VALIDATE_RESUME_MODE=${VALIDATE_RESUME_MODE:-all_missing}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    VALIDATE_SHARD_INDEX=${VALIDATE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID}}
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set."
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}"
    echo "Run 'make setup-hpc' first, or pass ENV_DIR explicitly."
    exit 1
fi

SIF_PATH="${SCRATCH}/containers/bigcodebench-evaluate.sif"
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Error: SIF image not found at ${SIF_PATH}"
    echo "Pull it first:  singularity pull ${SIF_PATH} docker://bigcodebench/bigcodebench-evaluate:latest"
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
source "${ENV_DIR}/bin/activate"

export VALIDATE_NUM_SHARDS
export VALIDATE_RESUME_MODE
if [[ -n "${VALIDATE_SHARD_INDEX:-}" ]]; then
    export VALIDATE_SHARD_INDEX
fi

echo "======================================"
echo "Phase 1 — Validate Mutants"
echo "======================================"
echo "Repo:        ${REPO_ROOT}"
echo "Config:      ${CONFIG}"
echo "Step:        ${STEP}"
echo "Environment: ${ENV_DIR}"
echo "SIF:         ${SIF_PATH}"
echo "Host:        $(hostname)"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem:         ${SLURM_MEM_PER_NODE:-?} MB"
echo "Parallel:    ${VALIDATE_PARALLEL:-config default}"
echo "Resume mode: ${VALIDATE_RESUME_MODE}"
echo "Shards:      ${VALIDATE_NUM_SHARDS}"
echo "Shard index: ${VALIDATE_SHARD_INDEX:-0}"
echo "======================================"

cmd=(python -u -m src.phase1.run --config "${CONFIG}" --step "${STEP}")
if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
fi

"${cmd[@]}"

echo "Done."
