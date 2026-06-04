#!/bin/bash
# Launcher for the ICSE26 OOD nested CV experiment.
#
# Submits scripts/slurm/jobs/icse26_ood.sbatch.
#
# Usage:
#   bash scripts/slurm/launchers/icse26_ood.sh [options]
#
# Example:
#   bash scripts/slurm/launchers/icse26_ood.sh \
#       --test-dataset bigcodebench --ood-source synthetic \
#       --model Qwen/Qwen2.5-Coder-7B-Instruct --protocol pca

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/icse26_ood.sbatch"

# ---- defaults ---------------------------------------------------------------
TEST_DATASET="bigcodebench"
OOD_SOURCE="synthetic"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
PROTOCOL="pca"
OUTER_FOLDS=""
INNER_FOLDS=""
BATCH_SIZE=8
MAX_LENGTH=4096
DTYPE="float16"
VECTORS_DIR="${VECTORS_DIR:-vectors}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"
TEST_SPLIT_FAMILY="${TEST_SPLIT_FAMILY:-icse26}"
OOD_SPLIT_FAMILY="${OOD_SPLIT_FAMILY:-icse26}"

PARTITION="nvidia"
ACCOUNT=""
QOS=""
CONSTRAINT=""
GPUS=1
CPUS=8
MEM="96G"
TIME="04:00:00"

PYTHON_MODULE="python/3.11.3"
CUDA_MODULE="cuda/12.2.0"

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${SCRATCH}/envs/llm-code-correctness"
    HF_HOME="${SCRATCH}/hf-models"
else
    ENV_DIR=""
    HF_HOME=""
fi

DEPENDENCY=""
DRY_RUN=0
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/slurm/launchers/icse26_ood.sh [options]

Experiment options:
  --test-dataset NAME       bigcodebench | humaneval        (default: bigcodebench)
  --ood-source NAME         synthetic | mbpp                (default: synthetic)
  --model NAME              HuggingFace model id            (default: Qwen/Qwen2.5-Coder-7B-Instruct)
  --protocol NAME           pca | function                  (default: pca)
  --outer-folds "N ..."     Subset of test folds            (default: all from manifest)
  --inner-folds "N ..."     Subset of fit/val folds         (default: all from manifest)
  --batch-size N            Extraction batch size           (default: 8)
  --max-length N            Max token length                (default: 4096)
  --dtype NAME              bfloat16 | float16 | float32    (default: float16)
  --data-root PATH          Root for datasets               (default: dataset/icse26)
  --test-split-family STR   Subdir under <data-root>/<test-dataset>/ (default: icse26)
  --ood-split-family STR    Subdir under <data-root>/<ood-source>/   (default: x_fold)

SLURM options:
  --partition NAME --account NAME --qos NAME --constraint EXPR
  --gpus N --cpus N --mem SIZE --time HH:MM:SS
  --dependency STR

Environment options:
  --env-dir PATH --hf-home PATH --python-module NAME --cuda-module NAME

  --dry-run               Print sbatch command without submitting
  --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-dataset)      TEST_DATASET="$2";      shift 2 ;;
        --ood-source)        OOD_SOURCE="$2";        shift 2 ;;
        --model)             MODEL="$2";             shift 2 ;;
        --protocol)          PROTOCOL="$2";          shift 2 ;;
        --outer-folds)       OUTER_FOLDS="$2";       shift 2 ;;
        --inner-folds)       INNER_FOLDS="$2";       shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";        shift 2 ;;
        --max-length)        MAX_LENGTH="$2";        shift 2 ;;
        --dtype)             DTYPE="$2";             shift 2 ;;
        --data-root)         DATA_ROOT="$2";         shift 2 ;;
        --test-split-family) TEST_SPLIT_FAMILY="$2"; shift 2 ;;
        --ood-split-family)  OOD_SPLIT_FAMILY="$2";  shift 2 ;;
        --partition)         PARTITION="$2";         shift 2 ;;
        --account)           ACCOUNT="$2";           shift 2 ;;
        --qos)               QOS="$2";               shift 2 ;;
        --constraint)        CONSTRAINT="$2";        shift 2 ;;
        --gpus)              GPUS="$2";              shift 2 ;;
        --cpus)              CPUS="$2";              shift 2 ;;
        --mem)               MEM="$2";               shift 2 ;;
        --time)              TIME="$2";              shift 2 ;;
        --env-dir)           ENV_DIR="$2";           shift 2 ;;
        --hf-home)           HF_HOME="$2";           shift 2 ;;
        --python-module)     PYTHON_MODULE="$2";     shift 2 ;;
        --cuda-module)       CUDA_MODULE="$2";       shift 2 ;;
        --dependency)        DEPENDENCY="$2";        shift 2 ;;
        --dry-run)           DRY_RUN=1;              shift ;;
        --help|-h)           usage; exit 0 ;;
        *)  echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2; exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: virtualenv not found: ${ENV_DIR}" >&2; exit 1
fi
if [[ ! -d "${HF_HOME}" ]]; then
    echo "Error: HF_HOME not found: ${HF_HOME}" >&2; exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

safe_model="${MODEL//\//__}"
sbatch_args=(
    --job-name="icse26_ood_${TEST_DATASET}_${OOD_SOURCE}_${PROTOCOL}_${safe_model}"
    --partition="${PARTITION}"
    --gpus="${GPUS}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
)
[[ -n "${ACCOUNT}" ]]    && sbatch_args+=(--account="${ACCOUNT}")
[[ -n "${QOS}" ]]        && sbatch_args+=(--qos="${QOS}")
[[ -n "${CONSTRAINT}" ]] && sbatch_args+=(--constraint="${CONSTRAINT}")
[[ -n "${DEPENDENCY}" ]] && sbatch_args+=(--dependency="${DEPENDENCY}")

env_vars=(
    "TEST_DATASET=${TEST_DATASET}"
    "OOD_SOURCE=${OOD_SOURCE}"
    "MODEL=${MODEL}"
    "PROTOCOL=${PROTOCOL}"
    "OUTER_FOLDS=${OUTER_FOLDS}"
    "INNER_FOLDS=${INNER_FOLDS}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "MAX_LENGTH=${MAX_LENGTH}"
    "DTYPE=${DTYPE}"
    "ENV_DIR=${ENV_DIR}"
    "HF_HOME=${HF_HOME}"
    "PYTHON_MODULE=${PYTHON_MODULE}"
    "CUDA_MODULE=${CUDA_MODULE}"
    "REPO_ROOT=${REPO_ROOT}"
    "VECTORS_DIR=${VECTORS_DIR}"
    "RESULTS_DIR=${RESULTS_DIR}"
    "DATA_ROOT=${DATA_ROOT}"
    "TEST_SPLIT_FAMILY=${TEST_SPLIT_FAMILY}"
    "OOD_SPLIT_FAMILY=${OOD_SPLIT_FAMILY}"
)
export_str=$(IFS=,; echo "${env_vars[*]}")

full_cmd=(
    sbatch
    "${sbatch_args[@]}"
    --export="${export_str}"
    "${JOB_SCRIPT}"
)

echo "================================================================"
echo "ICSE26 OOD nested CV"
echo "================================================================"
echo "  test:           ${TEST_DATASET}"
echo "  ood:            ${OOD_SOURCE}"
echo "  model:          ${MODEL}"
echo "  protocol:       ${PROTOCOL}"
echo "  outer folds:    ${OUTER_FOLDS:-<all>}"
echo "  inner folds:    ${INNER_FOLDS:-<all>}"
echo "  results:        ${RESULTS_DIR}"
echo "  vectors:        ${VECTORS_DIR}"
echo "  partition:      ${PARTITION}, gpus=${GPUS}, time=${TIME}"
echo "================================================================"
echo "${full_cmd[@]}"
echo "================================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run)"
    exit 0
fi

job_id=$("${full_cmd[@]}" | tee /dev/stderr | awk '/^Submitted batch job/ {print $4}')
echo "Submitted: job ${job_id}"
echo "Logs: logs/slurm/icse26_ood.${job_id}.out"
