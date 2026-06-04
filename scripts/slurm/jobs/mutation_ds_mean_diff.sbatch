#!/bin/bash
#SBATCH --job-name=mutation_ds_mean_diff
#SBATCH --output=logs/slurm/mutation_ds_mean_diff.%j.out
#SBATCH --error=logs/slurm/mutation_ds_mean_diff.%j.err
#SBATCH --partition=nvidia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Mutation_DS — mean-of-diffs, function-framing fit, neutral val/test.
# Runs scripts/experiments/mutation_ds_mean_diff.py.
#
# Overridable env vars (passed via --export by the launcher):
#   MODEL                      HF model id
#   SPLIT_FAMILY               s1_baseline | s2_test_confound | ...   (default: s1_baseline)
#   REPRESENTATION                                                    (default: response_avg)
#   METHOD                                                            (default: mean_diff)
#   FOLDS                      "0 1 2 ..."                            (default: "0 1 ... 9")
#   BATCH_SIZE / MAX_LENGTH / DTYPE
#   VECTORS_DIR / RESULTS_DIR / DATA_ROOT
#   ENV_DIR / HF_HOME / PYTHON_MODULE / CUDA_MODULE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
SPLIT_FAMILY=${SPLIT_FAMILY:-s1_baseline}
REPRESENTATION=${REPRESENTATION:-response_avg}
METHOD=${METHOD:-mean_diff}
FOLDS=${FOLDS:-0 1 2 3 4 5 6 7 8 9}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
VECTORS_DIR=${VECTORS_DIR:-vectors}
RESULTS_DIR=${RESULTS_DIR:-results}
DATA_ROOT=${DATA_ROOT:-dataset/bigcodebench/phase2}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.2.0}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}
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
if [[ ! -d "${DATA_ROOT}/${SPLIT_FAMILY}" ]]; then
    echo "Error: Mutation_DS split not found at ${DATA_ROOT}/${SPLIT_FAMILY}." >&2
    echo "Hint: run 'make decompress-datasets' to extract dataset/bigcodebench/phase2/*.tar.zst." >&2
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
source "${ENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "======================================"
echo "Mutation_DS — mean-of-diffs (function fit, neutral eval)"
echo "======================================"
echo "Repo:           ${REPO_ROOT}"
echo "Split family:   ${SPLIT_FAMILY}"
echo "Model:          ${MODEL}"
echo "Representation: ${REPRESENTATIONS:-${REPRESENTATION}}"
echo "Method:         ${METHOD}"
echo "Folds:          ${FOLDS}"
echo "Batch / maxlen: ${BATCH_SIZE} / ${MAX_LENGTH}"
echo "Dtype:          ${DTYPE}"
echo "Env:            ${ENV_DIR}"
echo "HF_HOME:        ${HF_HOME}"
echo "Host:           $(hostname)"
echo "GPUs visible:   ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================"

python -u scripts/experiments/mutation_ds_mean_diff.py \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --split-family "${SPLIT_FAMILY}" \
    --representation "${REPRESENTATION}" \
    --method "${METHOD}" \
    --folds ${FOLDS} \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --vectors-dir "${VECTORS_DIR}" \
    --results-dir "${RESULTS_DIR}"
