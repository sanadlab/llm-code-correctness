#!/bin/bash
#SBATCH --job-name=icse26_replication
#SBATCH --output=logs/slurm/icse26_replication.%j.out
#SBATCH --error=logs/slurm/icse26_replication.%j.err
#SBATCH --partition=cs
#SBATCH --account=condo_cs_nadi
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# ICSE26 paper replication.
# Runs scripts/experiments/icse26_replication.py end-to-end on one model.
#
# Overridable env vars (passed via --export by the launcher):
#   DATASET            bigcodebench | humaneval   (default: bigcodebench)
#   MODEL              HF model id                 (default: Qwen/Qwen2.5-Coder-7B-Instruct)
#   REPRESENTATION                                 (default: sequence_last)
#   METHOD                                         (default: pca)
#   FRAMING_FIT/VAL/TEST                           (default: raw/concept/concept)
#   FOLDS              "0 1 2 ..."                 (default: "0")
#   BATCH_SIZE                                     (default: 8)
#   MAX_LENGTH                                     (default: 2048)
#   DTYPE                                          (default: float16)
#   ENV_DIR / HF_HOME / PYTHON_MODULE / CUDA_MODULE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

DATASET=${DATASET:-bigcodebench}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
REPRESENTATION=${REPRESENTATION:-sequence_last}
METHOD=${METHOD:-pca}
FRAMING_FIT=${FRAMING_FIT:-raw}
FRAMING_VAL=${FRAMING_VAL:-concept}
FRAMING_TEST=${FRAMING_TEST:-concept}
FOLDS=${FOLDS:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
VECTORS_DIR=${VECTORS_DIR:-vectors}
RESULTS_DIR=${RESULTS_DIR:-results}
DATA_ROOT=${DATA_ROOT:-dataset/icse26}
SPLIT_FAMILY=${SPLIT_FAMILY:-icse26}
SPLITS_DIR=${SPLITS_DIR:-}

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

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
source "${ENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "======================================"
echo "ICSE26 replication"
echo "======================================"
echo "Repo:           ${REPO_ROOT}"
echo "Dataset:        ${DATASET}"
echo "Model:          ${MODEL}"
echo "Representation: ${REPRESENTATION}"
echo "Method:         ${METHOD}"
echo "Framings:       fit=${FRAMING_FIT}  val=${FRAMING_VAL}  test=${FRAMING_TEST}"
echo "Folds:          ${FOLDS}"
echo "Batch / maxlen: ${BATCH_SIZE} / ${MAX_LENGTH}"
echo "Dtype:          ${DTYPE}"
echo "Env:            ${ENV_DIR}"
echo "HF_HOME:        ${HF_HOME}"
echo "Host:           $(hostname)"
echo "GPUs visible:   ${CUDA_VISIBLE_DEVICES:-all}"
echo "======================================"

EXTRA_ARGS=()
if [[ -n "${SPLITS_DIR}" ]]; then
    EXTRA_ARGS+=(--splits-dir "${SPLITS_DIR}")
fi

python -u scripts/experiments/icse26_replication.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --representation "${REPRESENTATION}" \
    --method "${METHOD}" \
    --framing-fit "${FRAMING_FIT}" \
    --framing-val "${FRAMING_VAL}" \
    --framing-test "${FRAMING_TEST}" \
    --folds ${FOLDS} \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --vectors-dir "${VECTORS_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --data-root "${DATA_ROOT}" \
    --split-family "${SPLIT_FAMILY}" \
    "${EXTRA_ARGS[@]}"
