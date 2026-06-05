#!/bin/bash
# Launcher for the Mutation_DS mean-of-diffs experiment.
#
# Submits scripts/slurm/jobs/mutation_ds_mean_diff.sbatch with sensible defaults.
# The framing protocol (fit=function±, val/test=neutral, chat-templated) is
# baked into the experiment script — only model, split-family, fold subset,
# and resource knobs are configurable.
#
# Usage:
#   bash scripts/slurm/launchers/mutation_ds_mean_diff.sh [options]
#
# Examples:
#   bash scripts/slurm/launchers/mutation_ds_mean_diff.sh
#   bash scripts/slurm/launchers/mutation_ds_mean_diff.sh \
#       --split-family s1_baseline --model codellama/CodeLlama-7b-Instruct-hf
#   bash scripts/slurm/launchers/mutation_ds_mean_diff.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/mutation_ds_mean_diff.sbatch"

# ---- defaults ---------------------------------------------------------------
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
SPLIT_FAMILY="${SPLIT_FAMILY:-s1_baseline}"
REPRESENTATION="response_avg"
METHOD="mean_diff"
FOLDS="0 1 2 3 4 5 6 7 8 9"
BATCH_SIZE=8
MAX_LENGTH=2048
DTYPE="float16"
VECTORS_DIR="${VECTORS_DIR:-vectors}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_ROOT="${DATA_ROOT:-dataset/bigcodebench/phase2}"

PARTITION="cs"
ACCOUNT="condo_cs_nadi"
QOS=""
CONSTRAINT=""
GPUS=1
CPUS=8
MEM="64G"
TIME="01:00:00"

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
Usage: scripts/slurm/launchers/mutation_ds_mean_diff.sh [options]

Submit the Mutation_DS mean-of-diffs experiment to SLURM.

Experiment options:
  --split-family NAME     s1_baseline | s2_test_confound | s3_holdout_*  (default: s1_baseline)
  --data-root PATH        Parent dir holding <split-family>/           (default: dataset/bigcodebench/phase2)
  --model NAME            HuggingFace model id                          (default: Qwen/Qwen2.5-Coder-7B-Instruct)
  --representation NAME   prompt_avg|prompt_last|response_avg|sequence_last  (default: response_avg)
  --method NAME           mean_diff | pca                               (default: mean_diff)
  --folds "N ..."         Space-separated fold indices                  (default: "0..9")
  --batch-size N          Extraction batch size                         (default: 8)
  --max-length N          Max token length                              (default: 2048)
  --dtype NAME            bfloat16 | float16 | float32                  (default: float16)

SLURM options:
  --partition NAME --account NAME --qos NAME --constraint EXPR
  --gpus N --cpus N --mem SIZE --time HH:MM:SS --dependency STR

Environment options:
  --env-dir PATH          Virtualenv to activate           (default: $SCRATCH/envs/llm-code-correctness)
  --hf-home PATH          HuggingFace cache dir            (default: $SCRATCH/hf-models)
  --python-module NAME --cuda-module NAME

  --dry-run               Print sbatch command without submitting
  --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)       DATA_ROOT="$2";     shift 2 ;;
        --split-family)    SPLIT_FAMILY="$2";  shift 2 ;;
        --model)           MODEL="$2";         shift 2 ;;
        --representation)  REPRESENTATION="$2";shift 2 ;;
        --method)          METHOD="$2";        shift 2 ;;
        --folds)           FOLDS="$2";         shift 2 ;;
        --batch-size)      BATCH_SIZE="$2";    shift 2 ;;
        --max-length)      MAX_LENGTH="$2";    shift 2 ;;
        --dtype)           DTYPE="$2";         shift 2 ;;
        --partition)       PARTITION="$2";     shift 2 ;;
        --account)         ACCOUNT="$2";       shift 2 ;;
        --qos)             QOS="$2";           shift 2 ;;
        --constraint)      CONSTRAINT="$2";    shift 2 ;;
        --gpus)            GPUS="$2";          shift 2 ;;
        --cpus)            CPUS="$2";          shift 2 ;;
        --mem)             MEM="$2";           shift 2 ;;
        --time)            TIME="$2";          shift 2 ;;
        --env-dir)         ENV_DIR="$2";       shift 2 ;;
        --hf-home)         HF_HOME="$2";       shift 2 ;;
        --python-module)   PYTHON_MODULE="$2"; shift 2 ;;
        --cuda-module)     CUDA_MODULE="$2";   shift 2 ;;
        --dependency)      DEPENDENCY="$2";    shift 2 ;;
        --dry-run)         DRY_RUN=1;          shift ;;
        --help|-h)         usage; exit 0 ;;
        *)  echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2; exit 1
fi
if [[ -z "${ENV_DIR}" ]]; then
    echo "Error: --env-dir required (SCRATCH unavailable)." >&2; exit 1
fi
if [[ -z "${HF_HOME}" ]]; then
    echo "Error: --hf-home required (SCRATCH unavailable)." >&2; exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: virtualenv not found: ${ENV_DIR}" >&2; exit 1
fi
if [[ ! -d "${HF_HOME}" ]]; then
    echo "Error: HF_HOME not found: ${HF_HOME}" >&2; exit 1
fi
if [[ ! -d "${REPO_ROOT}/${DATA_ROOT}/${SPLIT_FAMILY}" ]]; then
    echo "Error: split not found: ${DATA_ROOT}/${SPLIT_FAMILY}" >&2
    echo "Hint:  run 'make decompress-datasets' to extract ${DATA_ROOT}/*.tar.zst" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

sbatch_args=(
    --job-name="mutation_ds_${SPLIT_FAMILY}_${METHOD}_${REPRESENTATION}"
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
    "MODEL=${MODEL}"
    "SPLIT_FAMILY=${SPLIT_FAMILY}"
    "REPRESENTATION=${REPRESENTATION}"
    "METHOD=${METHOD}"
    "FOLDS=${FOLDS}"
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
)
export_str=$(IFS=,; echo "${env_vars[*]}")

full_cmd=(
    sbatch
    "${sbatch_args[@]}"
    --export="${export_str}"
    "${JOB_SCRIPT}"
)

echo "================================================================"
echo "Mutation_DS — mean-of-diffs (function fit, neutral eval)"
echo "================================================================"
echo "  data-root:      ${DATA_ROOT}"
echo "  split-family:   ${SPLIT_FAMILY}"
echo "  model:          ${MODEL}"
echo "  representation: ${REPRESENTATION}"
echo "  method:         ${METHOD}"
echo "  framings:       fit=function±  val=neutral  test=neutral  (chat-templated)"
echo "  folds:          ${FOLDS}"
echo "  batch/maxlen:   ${BATCH_SIZE} / ${MAX_LENGTH}"
echo "  dtype:          ${DTYPE}"
echo "  resources:      ${PARTITION}, ${GPUS}×GPU, ${CPUS}×CPU, ${MEM}, ${TIME}"
echo "  env:            ${ENV_DIR}"
echo "  HF_HOME:        ${HF_HOME}"
if [[ -n "${DEPENDENCY}" ]]; then
    echo "  dependency:     ${DEPENDENCY}"
fi
echo ""
echo "  sbatch: ${full_cmd[*]}"
echo "================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "(dry-run — not submitting)"
    exit 0
fi

output=$("${full_cmd[@]}" 2>&1)
echo "${output}"
job_id=$(echo "${output}" | grep -oP 'Submitted batch job \K[0-9]+' || true)
if [[ -n "${job_id}" ]]; then
    echo "Submitted: job ${job_id}"
    echo "Logs: logs/slurm/mutation_ds_mean_diff.${job_id}.out"
fi
