#!/bin/bash
#SBATCH --job-name=cache_hidden_states
#SBATCH --output=logs/slurm/cache_hidden_states.%j.out
#SBATCH --error=logs/slurm/cache_hidden_states.%j.err
#SBATCH --partition=nvidia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Extract and cache hidden states (scripts/extract/cache_hidden_states.py).
#
# Overridable env vars (pass via sbatch --export):
#   MODEL, DATASET_SOURCE, SPLIT_FAMILY, PROTOCOL, DATA_SPLITS, REPRESENTATIONS
#   ENV_DIR, HF_HOME, PYTHON_MODULE, CUDA_MODULE, REPO_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL=${MODEL:-infly/OpenCoder-8B-Instruct}
DATASET_SOURCE=${DATASET_SOURCE:-mutation}
SPLIT_FAMILY=${SPLIT_FAMILY:-s8_single_arithmetic}
PROTOCOL=${PROTOCOL:-mean_diff}
DATA_SPLITS=${DATA_SPLITS:-validate}
REPRESENTATIONS=${REPRESENTATIONS:-response_avg}
VERIFY=${VERIFY:-1}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.2.0}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}
HF_HOME=${HF_HOME:-${SCRATCH:-}/hf-models}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2
    exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}" >&2
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
source "${ENV_DIR}/bin/activate"

export HF_HOME
export HF_HUB_CACHE="${HF_HOME}"

echo "======================================"
echo "Cache hidden states"
echo "======================================"
echo "Repo:            ${REPO_ROOT}"
echo "Model:           ${MODEL}"
echo "Dataset source:  ${DATASET_SOURCE}"
echo "Split family:    ${SPLIT_FAMILY}"
echo "Protocol:        ${PROTOCOL}"
echo "Data splits:     ${DATA_SPLITS}"
echo "Representations: ${REPRESENTATIONS}"
echo "Verify:          ${VERIFY}"
echo "Env:             ${ENV_DIR}"
echo "HF_HOME:         ${HF_HOME}"
echo "Host:            $(hostname)"
echo "GPUs visible:    ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================"

cmd=(
    python -u scripts/extract/cache_hidden_states.py
    --models "${MODEL}"
    --dataset-source "${DATASET_SOURCE}"
    --split-family "${SPLIT_FAMILY}"
    --protocol "${PROTOCOL}"
    --data-splits ${DATA_SPLITS}
    --representations ${REPRESENTATIONS}
)
if [[ "${VERIFY}" == "1" ]]; then
    cmd+=(--verify)
fi

"${cmd[@]}"

echo "Done."
