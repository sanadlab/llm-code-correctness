#!/bin/bash
#SBATCH --job-name=cache_icse26_val_1_4
#SBATCH --output=logs/slurm/cache_icse26_val_1_4.%j.out
#SBATCH --error=logs/slurm/cache_icse26_val_1_4.%j.err
#SBATCH --partition=nvidia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

# Cache hidden states for mutation-sourced splits (e.g. icse26_val_1_4).
# Runs scripts/extract/cache_hidden_states.py via the launcher --local.
#
# Overridable env vars (passed via --export by the launcher):
#   MODEL / SPLIT_FAMILY / BENCHMARK / DATA_ROOT / OUTPUT_DIR
#   DATA_SPLITS / PROTOCOL / REPRESENTATIONS / FOLDS
#   BATCH_SIZE / FIT_MAX_LENGTH / EVAL_MAX_LENGTH / DTYPE / FORCE
#   ENV_DIR / HF_HOME / PYTHON_MODULE / CUDA_MODULE / REPO_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL=${MODEL:-}
SPLIT_FAMILY=${SPLIT_FAMILY:-icse26_val_1_4}
BENCHMARK=${BENCHMARK:-bigcodebench}
DATA_ROOT=${DATA_ROOT:-/scratch/re2230}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/re2230/hidden_states_re2230}
DATA_SPLITS=${DATA_SPLITS:-fit}
PROTOCOL=${PROTOCOL:-mean_diff}
REPRESENTATIONS=${REPRESENTATIONS:-response_avg}
FOLDS=${FOLDS:-}
BATCH_SIZE=${BATCH_SIZE:-8}
FIT_MAX_LENGTH=${FIT_MAX_LENGTH:-2048}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-4096}
DTYPE=${DTYPE:-float16}
FORCE=${FORCE:-0}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.2.0}
ENV_DIR=${ENV_DIR:-/scratch/re2230/envs/llm-code-correctness}
HF_HOME=${HF_HOME:-/scratch/re2230/hf-models}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -z "${MODEL}" ]]; then
    echo "Error: MODEL is not set." >&2
    exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}" >&2
    exit 1
fi
if [[ ! -d "${HF_HOME}" ]]; then
    echo "Error: HF_HOME does not exist: ${HF_HOME}" >&2
    exit 1
fi
if [[ ! -f "${DATA_ROOT}/${SPLIT_FAMILY}/manifest.json" ]]; then
    echo "Error: split family not found at ${DATA_ROOT}/${SPLIT_FAMILY}/manifest.json" >&2
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
source "${ENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export MODEL SPLIT_FAMILY BENCHMARK DATA_ROOT OUTPUT_DIR
export DATA_SPLITS PROTOCOL REPRESENTATIONS FOLDS
export BATCH_SIZE FIT_MAX_LENGTH EVAL_MAX_LENGTH DTYPE FORCE

echo "======================================"
echo "Cache hidden states (icse26_val_1_4)"
echo "======================================"
echo "Repo:             ${REPO_ROOT}"
echo "Model:            ${MODEL}"
echo "Split family:     ${SPLIT_FAMILY}"
echo "Benchmark:        ${BENCHMARK}"
echo "Data root:        ${DATA_ROOT}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Data splits:      ${DATA_SPLITS}"
echo "Protocol:         ${PROTOCOL}"
echo "Representations:  ${REPRESENTATIONS}"
echo "Folds:            ${FOLDS:-<all>}"
echo "Batch / lengths:  ${BATCH_SIZE} / fit=${FIT_MAX_LENGTH} eval=${EVAL_MAX_LENGTH}"
echo "Dtype:            ${DTYPE}"
echo "Force:            ${FORCE}"
echo "Env:              ${ENV_DIR}"
echo "HF_HOME:          ${HF_HOME}"
echo "Host:             $(hostname)"
echo "GPUs visible:     ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================"

bash "${REPO_ROOT}/scripts/slurm/launchers/cache_icse26_val_1_4.sh" --local

echo "Done."
