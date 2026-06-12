#!/bin/bash
# Launcher for the ICSE26 paper-replication experiment.
#
# Submits scripts/slurm/jobs/icse26_replication.sbatch with sensible defaults.
# Every option can be overridden on the command line.
#
# Usage:
#   bash scripts/slurm/launchers/icse26_replication.sh [options]
#
# Examples:
#   # Defaults: bigcodebench, Qwen, all folds, raw fit + concept eval, PCA
#   bash scripts/slurm/launchers/icse26_replication.sh
#
#   # Subset of folds with mean_diff
#   bash scripts/slurm/launchers/icse26_replication.sh \
#       --folds "0 1 2" --method mean_diff --representation response_avg
#
#   # humaneval, different model
#   bash scripts/slurm/launchers/icse26_replication.sh \
#       --dataset humaneval --model codellama/CodeLlama-7b-Instruct-hf
#
#   # Dry-run (print sbatch command, don't submit)
#   bash scripts/slurm/launchers/icse26_replication.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/icse26_replication.sbatch"

# ---- defaults ---------------------------------------------------------------
DATASET="bigcodebench"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
REPRESENTATION="sequence_last"
METHOD="pca"
FRAMING_FIT="raw"
FRAMING_VAL="concept"
FRAMING_TEST="concept"
FOLDS="0 1 2 3 4 5 6 7 8 9"
BATCH_SIZE=8
MAX_LENGTH=4096
DTYPE="float16"
VECTORS_DIR="${VECTORS_DIR:-vectors}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"
SPLIT_FAMILY="${SPLIT_FAMILY:-icse26}"
SPLITS_DIR="${SPLITS_DIR:-}"

PARTITION="cs"
ACCOUNT="condo_cs_nadi"
QOS=""
CONSTRAINT=""
GPUS=1
CPUS=8
MEM="64G"
TIME="00:10:00"

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
Usage: scripts/slurm/launchers/icse26_replication.sh [options]

Submit the ICSE26 paper-replication experiment to SLURM.

Experiment options:
  --dataset NAME          bigcodebench | humaneval         (default: bigcodebench)
  --model NAME            HuggingFace model id             (default: Qwen/Qwen2.5-Coder-7B-Instruct)
  --representation NAME   prompt_avg|prompt_last|response_avg|sequence_last  (default: response_avg)
  --method NAME           mean_diff | pca                  (default: mean_diff)
  --framing-fit NAME      neutral | concept | function     (default: neutral)
  --framing-val NAME      neutral | concept                (default: concept)
  --framing-test NAME     neutral | concept                (default: concept)
  --folds "N ..."         Space-separated fold indices     (default: "0")
  --batch-size N          Extraction batch size            (default: 4)
  --max-length N          Max token length                 (default: 2048)
  --dtype NAME            bfloat16 | float16 | float32     (default: float16)

SLURM options:
  --partition NAME        SLURM partition                  (default: nvidia)
  --account NAME          SLURM account                    (default: none)
  --qos NAME              SLURM QoS                        (default: none)
  --constraint EXPR       SLURM --constraint expression    (default: none)
  --gpus N                Number of GPUs                   (default: 1)
  --cpus N                CPUs per task                    (default: 8)
  --mem SIZE              Memory request                   (default: 64G)
  --time HH:MM:SS         Wall-clock limit                 (default: 00:10:00)
  --dependency STR        SLURM dependency (e.g. afterok:12345)

Environment options:
  --env-dir PATH          Virtualenv to activate           (default: $SCRATCH/envs/llm-code-correctness)
  --hf-home PATH          HuggingFace cache dir            (default: $SCRATCH/hf-models)
  --python-module NAME    Module to load                   (default: python/3.11.3)
  --cuda-module NAME      Module to load                   (default: cuda/12.2.0)

  --dry-run               Print sbatch command without submitting
  --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET="$2";       shift 2 ;;
        --data-root)       DATA_ROOT="$2";     shift 2 ;;
        --split-family)    SPLIT_FAMILY="$2";  shift 2 ;;
        --splits-dir)      SPLITS_DIR="$2";    shift 2 ;;
        --model)           MODEL="$2";         shift 2 ;;
        --representation)  REPRESENTATION="$2";shift 2 ;;
        --method)          METHOD="$2";        shift 2 ;;
        --framing-fit)     FRAMING_FIT="$2";   shift 2 ;;
        --framing-val)     FRAMING_VAL="$2";   shift 2 ;;
        --framing-test)    FRAMING_TEST="$2";  shift 2 ;;
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

# ---- validation -------------------------------------------------------------
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

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

# ---- build sbatch command ---------------------------------------------------
sbatch_args=(
    --job-name="icse26_replication_${DATASET}_${METHOD}_${REPRESENTATION}"
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
    "DATASET=${DATASET}"
    "MODEL=${MODEL}"
    "REPRESENTATION=${REPRESENTATION}"
    "METHOD=${METHOD}"
    "FRAMING_FIT=${FRAMING_FIT}"
    "FRAMING_VAL=${FRAMING_VAL}"
    "FRAMING_TEST=${FRAMING_TEST}"
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
    "SPLIT_FAMILY=${SPLIT_FAMILY}"
    "SPLITS_DIR=${SPLITS_DIR}"
)
export_str=$(IFS=,; echo "${env_vars[*]}")

full_cmd=(
    sbatch
    "${sbatch_args[@]}"
    --export="${export_str}"
    "${JOB_SCRIPT}"
)

# ---- print summary ----------------------------------------------------------
echo "================================================================"
echo "ICSE26 replication"
echo "================================================================"
echo "  dataset:        ${DATASET}"
echo "  data-root:      ${DATA_ROOT}"
echo "  split-family:   ${SPLIT_FAMILY}"
echo "  model:          ${MODEL}"
echo "  representation: ${REPRESENTATION}"
echo "  method:         ${METHOD}"
echo "  framings:       fit=${FRAMING_FIT}  val=${FRAMING_VAL}  test=${FRAMING_TEST}"
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
    echo "Logs: logs/slurm/icse26_replication.${job_id}.out"
fi
