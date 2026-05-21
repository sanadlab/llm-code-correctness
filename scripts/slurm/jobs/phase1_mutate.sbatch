#!/bin/bash
#SBATCH --job-name=phase1_mutate
#SBATCH --output=logs/slurm/phase1_mutate.%j.out
#SBATCH --error=logs/slurm/phase1_mutate.%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG=${CONFIG:-configs/phase1/bigcodebench.yaml}
STEP=${STEP:-mutate}
PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/project-abu-dhabi}
FORCE=${FORCE:-0}
MUTATE_PARALLEL=${MUTATE_PARALLEL:-${SLURM_CPUS_PER_TASK:-1}}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}"
    echo "Run 'make setup-hpc' first, or pass ENV_DIR explicitly."
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
source "${ENV_DIR}/bin/activate"

export MUTATE_PARALLEL

echo "=============================="
echo "Phase 1 — Mutate"
echo "=============================="
echo "Repo:        ${REPO_ROOT}"
echo "Config:      ${CONFIG}"
echo "Step:        ${STEP}"
echo "Environment: ${ENV_DIR}"
echo "Host:        $(hostname)"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-?}"
echo "Parallel:    ${MUTATE_PARALLEL}"
echo "=============================="

cmd=(python -u -m src.phase1.run --config "${CONFIG}" --step "${STEP}")
if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
fi

"${cmd[@]}"

echo "Done."
