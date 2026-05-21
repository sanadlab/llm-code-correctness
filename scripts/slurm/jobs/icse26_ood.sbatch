#!/bin/bash
#SBATCH --job-name=icse26_ood
#SBATCH --output=logs/slurm/icse26_ood.%j.out
#SBATCH --error=logs/slurm/icse26_ood.%j.err
#SBATCH --partition=nvidia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

# ICSE26 monitoring — OOD nested cross-validation.
# Runs scripts/experiments/icse26_ood.py.
#
# Overridable env vars (passed via --export by the launcher):
#   TEST_DATASET                bigcodebench | humaneval
#   OOD_SOURCE                  synthetic | mbpp
#   MODEL                       HF model id
#   PROTOCOL                    pca | function
#   OUTER_FOLDS                 "0 1 ... 9"  (default: "" → all from manifest)
#   INNER_FOLDS                 "0 1 2 3"    (default: "" → all from manifest)
#   BATCH_SIZE / MAX_LENGTH / DTYPE
#   VECTORS_DIR / RESULTS_DIR / DATA_ROOT
#   TEST_SPLIT_FAMILY / OOD_SPLIT_FAMILY
#   ENV_DIR / HF_HOME / PYTHON_MODULE / CUDA_MODULE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

TEST_DATASET=${TEST_DATASET:-bigcodebench}
OOD_SOURCE=${OOD_SOURCE:-synthetic}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
PROTOCOL=${PROTOCOL:-pca}
OUTER_FOLDS=${OUTER_FOLDS:-}
INNER_FOLDS=${INNER_FOLDS:-}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-4096}
DTYPE=${DTYPE:-float16}
VECTORS_DIR=${VECTORS_DIR:-vectors}
RESULTS_DIR=${RESULTS_DIR:-results}
DATA_ROOT=${DATA_ROOT:-dataset/icse26}
TEST_SPLIT_FAMILY=${TEST_SPLIT_FAMILY:-icse26}
OOD_SPLIT_FAMILY=${OOD_SPLIT_FAMILY:-icse26}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.2.0}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/project-abu-dhabi}
HF_HOME=${HF_HOME:-${SCRATCH:-}/hf-models}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2; exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}" >&2; exit 1
fi
if [[ ! -d "${HF_HOME}" ]]; then
    echo "Error: HF_HOME does not exist: ${HF_HOME}" >&2; exit 1
fi

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
source "${ENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "======================================"
echo "ICSE26 OOD nested CV"
echo "======================================"
echo "Repo:           ${REPO_ROOT}"
echo "Test dataset:   ${TEST_DATASET}  (${TEST_SPLIT_FAMILY})"
echo "OOD source:     ${OOD_SOURCE}  (${OOD_SPLIT_FAMILY})"
echo "Model:          ${MODEL}"
echo "Protocol:       ${PROTOCOL}"
echo "Outer folds:    ${OUTER_FOLDS:-<all>}"
echo "Inner folds:    ${INNER_FOLDS:-<all>}"
echo "Batch / maxlen: ${BATCH_SIZE} / ${MAX_LENGTH}"
echo "Dtype:          ${DTYPE}"
echo "Env:            ${ENV_DIR}"
echo "HF_HOME:        ${HF_HOME}"
echo "Host:           $(hostname)"
echo "GPUs visible:   ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================"

EXTRA_ARGS=()
if [[ -n "${OUTER_FOLDS}" ]]; then
    EXTRA_ARGS+=(--outer-folds ${OUTER_FOLDS})
fi
if [[ -n "${INNER_FOLDS}" ]]; then
    EXTRA_ARGS+=(--inner-folds ${INNER_FOLDS})
fi

python -u scripts/experiments/icse26_ood.py \
    --test-dataset "${TEST_DATASET}" \
    --ood-source "${OOD_SOURCE}" \
    --data-root "${DATA_ROOT}" \
    --test-split-family "${TEST_SPLIT_FAMILY}" \
    --ood-split-family "${OOD_SPLIT_FAMILY}" \
    --model "${MODEL}" \
    --protocol "${PROTOCOL}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --vectors-dir "${VECTORS_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    "${EXTRA_ARGS[@]}"
