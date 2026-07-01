#!/bin/bash
#SBATCH --job-name=ablation_cache
#SBATCH --output=logs/slurm/ablation_cache.%j.out
#SBATCH --error=logs/slurm/ablation_cache.%j.err
#SBATCH --partition=nvidia
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

# Ablation sweep — stage 1 (hidden-state caching).
# Runs scripts/extract/cache_hidden_states.py for one model x dataset-source
# spec. Framing/representation axes come from scripts/slurm/ablation_grid.sh.
#
# Overridable env vars (passed via --export by the launcher):
#   MODEL            HF model id
#   DATASET_SOURCE   icse26 | ood | mutants_only | mutants_confounds
#   BENCHMARK        bigcodebench | humaneval        (DATASET_SOURCE=icse26)
#   OOD_SOURCE       mbpp | synthetic                (DATASET_SOURCE=ood)
#   TEST_DATASET     bigcodebench | humaneval        (legacy default; ignored for ood/mutants cache)
#   SPLIT_FAMILY     paired or weighted                (DATASET_SOURCE=mutants_only|mutants_confounds)
#   MUTANTS_BENCHMARK  bcb_based | he_based            (DATASET_SOURCE=mutants_only|mutants_confounds)
#   BATCH_SIZE / FIT_MAX_LENGTH / EVAL_MAX_LENGTH / DTYPE
#   OUTPUT_DIR / DATA_ROOT
#   ENV_DIR / HF_HOME / PYTHON_MODULE / CUDA_MODULE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
DATASET_SOURCE=${DATASET_SOURCE:-icse26}
BENCHMARK=${BENCHMARK:-bigcodebench}
OOD_SOURCE=${OOD_SOURCE:-mbpp}
TEST_DATASET=${TEST_DATASET:-bigcodebench}
MUTANTS_BENCHMARK=${MUTANTS_BENCHMARK:-bcb_based}
BATCH_SIZE=${BATCH_SIZE:-8}
FIT_MAX_LENGTH=${FIT_MAX_LENGTH:-2048}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-4096}
DTYPE=${DTYPE:-float16}
OUTPUT_DIR=${OUTPUT_DIR:-hidden_states}
if [[ -z "${DATA_ROOT:-}" ]]; then
    case "${DATASET_SOURCE}" in
        mutants_only)      DATA_ROOT="dataset/mutants_only_${MUTANTS_BENCHMARK}" ;;
        mutants_confounds) DATA_ROOT="dataset/mutants_confounds_${MUTANTS_BENCHMARK}" ;;
        *)                 DATA_ROOT="dataset/icse26" ;;
    esac
fi

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.2.0}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}
HF_HOME=${HF_HOME:-${SCRATCH:-}/.cache/huggingface}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# shellcheck source=../ablation_grid.sh
source "${REPO_ROOT}/scripts/slurm/ablation_grid.sh"

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

# shellcheck source=../stage_hf_model.sh
source "${REPO_ROOT}/scripts/slurm/stage_hf_model.sh"
stage_hf_model_to_local "${MODEL}"

echo "======================================"
echo "Ablation cache sweep (stage 1)"
echo "======================================"
echo "Repo:           ${REPO_ROOT}"
echo "Model:          ${MODEL}"
echo "Dataset source: ${DATASET_SOURCE}"
[[ "${DATASET_SOURCE}" == "icse26" ]] && echo "Benchmark:      ${BENCHMARK}"
[[ "${DATASET_SOURCE}" == "ood" ]] && echo "OOD source:     ${OOD_SOURCE}"
[[ "${DATASET_SOURCE}" == "mutants_only" ]] && echo "Split family:   ${SPLIT_FAMILY}" && echo "Mutants benchmark: ${MUTANTS_BENCHMARK}"
[[ "${DATASET_SOURCE}" == "mutants_confounds" ]] && echo "Split family:   ${SPLIT_FAMILY}" && echo "Mutants benchmark: ${MUTANTS_BENCHMARK}"
echo "Host:           $(hostname)"
echo "======================================"

COMMON_ARGS=(
    --models "${MODEL}"
    --data-root "${DATA_ROOT}"
    --representations "${ABLATION_REPRESENTATIONS[@]}"
    --batch-size "${BATCH_SIZE}"
    --fit-max-length "${FIT_MAX_LENGTH}"
    --eval-max-length "${EVAL_MAX_LENGTH}"
    --dtype "${DTYPE}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${DATASET_SOURCE}" == "icse26" ]]; then
    DATASET_ARGS=(--dataset-source icse26 --benchmark "${BENCHMARK}")
    for framing in "${ABLATION_FIT_FRAMINGS[@]}"; do
        echo "--- fit framing=${framing} ---"
        python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
            --data-splits fit --framing "${framing}"
    done
    for framing in "${ABLATION_VAL_FRAMINGS[@]}"; do
        echo "--- validate+test framing=${framing} ---"
        python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
            --data-splits validate test --framing "${framing}"
    done
elif [[ "${DATASET_SOURCE}" == "mutants_only" ]]; then
    SPLIT_FAMILY=${SPLIT_FAMILY:?SPLIT_FAMILY required for mutants_only}
    DATASET_ARGS=(--dataset-source mutants_only --split-family "${SPLIT_FAMILY}" --mutants-benchmark "${MUTANTS_BENCHMARK}")
    for framing in "${ABLATION_FIT_FRAMINGS[@]}"; do
        echo "--- fit framing=${framing} ---"
        python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
            --data-splits fit --framing "${framing}"
    done
    for framing in "${ABLATION_VAL_FRAMINGS[@]}"; do
        if [[ "${SPLIT_FAMILY}" == "weighted" ]]; then
            echo "--- validate+test framing=${framing} ---"
            python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
                --data-splits validate test --framing "${framing}"
        else
            echo "--- validate framing=${framing} ---"
            python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
                --data-splits validate --framing "${framing}"
        fi
    done
elif [[ "${DATASET_SOURCE}" == "mutants_confounds" ]]; then
    SPLIT_FAMILY=${SPLIT_FAMILY:?SPLIT_FAMILY required for mutants_confounds}
    DATASET_ARGS=(--dataset-source mutants_confounds --split-family "${SPLIT_FAMILY}" --mutants-benchmark "${MUTANTS_BENCHMARK}")
    for framing in "${ABLATION_FIT_FRAMINGS[@]}"; do
        echo "--- fit framing=${framing} ---"
        python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
            --data-splits fit --framing "${framing}"
    done
    for framing in "${ABLATION_VAL_FRAMINGS[@]}"; do
        if [[ "${SPLIT_FAMILY}" == "weighted" ]]; then
            echo "--- validate+test framing=${framing} ---"
            python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
                --data-splits validate test --framing "${framing}"
        else
            echo "--- validate framing=${framing} ---"
            python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
                --data-splits validate --framing "${framing}"
        fi
    done
else
    DATASET_ARGS=(--dataset-source ood --ood-source "${OOD_SOURCE}")
    for framing in "${ABLATION_FIT_FRAMINGS[@]}"; do
        echo "--- inner fit framing=${framing} ---"
        python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
            --data-splits fit --framing "${framing}"
    done
    for framing in "${ABLATION_VAL_FRAMINGS[@]}"; do
        echo "--- inner validate framing=${framing} ---"
        python -u scripts/extract/cache_hidden_states.py "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" \
            --data-splits validate --framing "${framing}"
    done
fi

echo "Done."
