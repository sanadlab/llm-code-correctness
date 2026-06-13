#!/bin/bash
#SBATCH --job-name=phase1_confound
#SBATCH --output=logs/slurm/phase1_confound.%j.out
#SBATCH --error=logs/slurm/phase1_confound.%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG=${CONFIG:-configs/phase1/bigcodebench.yaml}
STEP=${STEP:-confound}
PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.2.0}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}
HF_HOME=${HF_HOME:-${SCRATCH:-}/hf-models}
FORCE=${FORCE:-0}

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
    echo "Run 'module load python/3.11.3 && module load cuda/12.2.0 && make setup-hpc' first, or pass ENV_DIR explicitly."
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
source "${ENV_DIR}/bin/activate"

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

export HF_HOME
mkdir -p "${HF_HOME}"

echo "======================================"
echo "Phase 1 Local Confound"
echo "======================================"
echo "Repo: ${REPO_ROOT}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-unset}"
echo "Config: ${CONFIG}"
echo "Step: ${STEP}"
echo "Environment: ${ENV_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "Host: $(hostname)"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================"

cmd=(python -m src.phase1.run --config "${CONFIG}" --step "${STEP}")
if [[ "${FORCE}" == "1" ]]; then
    cmd+=(--force)
fi

"${cmd[@]}"

echo "Done."