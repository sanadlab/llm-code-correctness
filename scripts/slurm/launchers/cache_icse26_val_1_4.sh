#!/bin/bash
# Launcher for caching hidden states on mutation-sourced splits (icse26_val_1_4).
#
# Submits scripts/slurm/jobs/cache_icse26_val_1_4.sbatch with sensible defaults,
# or runs locally with --local. GPU job; loads the model once per invocation.
#
# Usage:
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh [options]
#
# Examples:
#   # Submit default model (fit+validate+test bundles)
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh
#
#   # Cache fit + validate + test for mean_diff protocol
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh \
#       --data-splits "fit validate test"
#
#   # Run locally
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh --local
#
#   # Dry-run (print sbatch command, don't submit)
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh --dry-run
#
#   # Sweep multiple models
#   MODELS="Qwen/Qwen2.5-Coder-7B-Instruct infly/OpenCoder-8B-Instruct" \
#     bash scripts/slurm/launchers/cache_icse26_val_1_4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/cache_icse26_val_1_4.sbatch"

# ---- defaults ---------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
SPLIT_FAMILY="${SPLIT_FAMILY:-icse26_val_1_4}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
DATA_ROOT="${DATA_ROOT:-/scratch/re2230}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/re2230/hidden_states_re2230}"
DATA_SPLITS="${DATA_SPLITS:-fit validate test}"
PROTOCOL="${PROTOCOL:-mean_diff}"
REPRESENTATIONS="${REPRESENTATIONS:-response_avg}"
FOLDS="${FOLDS:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FIT_MAX_LENGTH="${FIT_MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-4096}"
DTYPE="${DTYPE:-float16}"
FORCE=0

PARTITION="${PARTITION:-nvidia}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-04:00:00}"

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.3}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2.0}"
if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${ENV_DIR:-${SCRATCH}/envs/llm-code-correctness}"
    HF_HOME="${HF_HOME:-${SCRATCH}/hf-models}"
else
    ENV_DIR="${ENV_DIR:-/scratch/re2230/envs/llm-code-correctness}"
    HF_HOME="${HF_HOME:-/scratch/re2230/hf-models}"
fi

MODE="submit"
DRY_RUN=0
DEPENDENCY=""
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/slurm/launchers/cache_icse26_val_1_4.sh [options]

Cache hidden states for mutation-sourced splits (default: icse26_val_1_4).

Experiment options:
  --model NAME              HuggingFace model id                  (default: Qwen/Qwen2.5-Coder-7B-Instruct)
  --split-family NAME       Split family under data-root          (default: icse26_val_1_4)
  --benchmark NAME          bigcodebench | humaneval              (default: bigcodebench; mutation source uses bigcodebench)
  --data-root PATH          Parent dir holding <split-family>/    (default: /scratch/re2230)
  --output-dir PATH         Hidden-state cache root               (default: /scratch/re2230/hidden_states_re2230)
  --data-splits "S ..."     fit | validate | test                 (default: fit validate test)
  --protocol NAME           mean_diff | pca                       (default: mean_diff)
  --representations "R ..." response_avg | sequence_last | ...    (default: response_avg)
  --folds "N ..."           Space-separated fold indices        (default: all folds)
  --batch-size N            Extraction batch size                 (default: 8)
  --fit-max-length N        Max tokens for fit split              (default: 2048)
  --eval-max-length N       Max tokens for validate/test          (default: 4096)
  --dtype NAME              bfloat16 | float16 | float32          (default: float16)
  --force                   Re-extract even if bundle exists

SLURM options:
  --partition NAME --account NAME --qos NAME --constraint EXPR
  --gpus N --cpus N --mem SIZE --time HH:MM:SS --dependency STR

Environment options:
  --env-dir PATH            Virtualenv to activate               (default: $SCRATCH/envs/llm-code-correctness)
  --hf-home PATH            HuggingFace cache dir                (default: $SCRATCH/hf-models)
  --python-module NAME --cuda-module NAME

  --local                   Run locally instead of submitting
  --dry-run                 Print sbatch command without submitting
  --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)              MODEL="$2";              shift 2 ;;
        --split-family)       SPLIT_FAMILY="$2";       shift 2 ;;
        --benchmark)          BENCHMARK="$2";          shift 2 ;;
        --data-root)          DATA_ROOT="$2";          shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2 ;;
        --data-splits)        DATA_SPLITS="$2";        shift 2 ;;
        --protocol)           PROTOCOL="$2";           shift 2 ;;
        --representations)    REPRESENTATIONS="$2";    shift 2 ;;
        --folds)              FOLDS="$2";              shift 2 ;;
        --batch-size)         BATCH_SIZE="$2";         shift 2 ;;
        --fit-max-length)     FIT_MAX_LENGTH="$2";     shift 2 ;;
        --eval-max-length)    EVAL_MAX_LENGTH="$2";    shift 2 ;;
        --dtype)              DTYPE="$2";              shift 2 ;;
        --force)              FORCE=1;                   shift ;;
        --partition)          PARTITION="$2";          shift 2 ;;
        --account)            ACCOUNT="$2";            shift 2 ;;
        --qos)                QOS="$2";                shift 2 ;;
        --constraint)         CONSTRAINT="$2";         shift 2 ;;
        --gpus)               GPUS="$2";               shift 2 ;;
        --cpus)               CPUS="$2";               shift 2 ;;
        --mem)                MEM="$2";                shift 2 ;;
        --time)               TIME="$2";               shift 2 ;;
        --env-dir)            ENV_DIR="$2";            shift 2 ;;
        --hf-home)            HF_HOME="$2";            shift 2 ;;
        --python-module)      PYTHON_MODULE="$2";      shift 2 ;;
        --cuda-module)        CUDA_MODULE="$2";        shift 2 ;;
        --dependency)         DEPENDENCY="$2";         shift 2 ;;
        --local)              MODE="local";            shift ;;
        --dry-run)            DRY_RUN=1;               shift ;;
        --help|-h)            usage; exit 0 ;;
        *)  echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS_ARRAY <<< "${MODELS}"
else
    MODELS_ARRAY=("${MODEL}")
fi

run_for_model() {
    local model="$1"
    cd "${REPO_ROOT}"
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    if [[ ! -f "${DATA_ROOT}/${SPLIT_FAMILY}/manifest.json" ]]; then
        echo "Error: split family not found at ${DATA_ROOT}/${SPLIT_FAMILY}/manifest.json" >&2
        exit 1
    fi

    local -a cmd=(
        python -u scripts/extract/cache_hidden_states.py
        --dataset-source mutation
        --split-family "${SPLIT_FAMILY}"
        --benchmark "${BENCHMARK}"
        --data-root "${DATA_ROOT}"
        --data-splits ${DATA_SPLITS}
        --protocol "${PROTOCOL}"
        --representations ${REPRESENTATIONS}
        --models "${model}"
        --output-dir "${OUTPUT_DIR}"
        --batch-size "${BATCH_SIZE}"
        --fit-max-length "${FIT_MAX_LENGTH}"
        --eval-max-length "${EVAL_MAX_LENGTH}"
        --dtype "${DTYPE}"
    )
    if [[ -n "${FOLDS}" ]]; then
        cmd+=(--folds ${FOLDS})
    fi
    if [[ "${FORCE}" == "1" ]]; then
        cmd+=(--force)
    fi

    echo "${cmd[*]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${cmd[@]}"
    fi
}

submit_for_model() {
    local model="$1"
    local safe_model="${model//\//__}"

    local -a sbatch_args=(
        --job-name="cache_hs_${SPLIT_FAMILY}_${safe_model}"
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

    local -a env_vars=(
        "MODEL=${model}"
        "SPLIT_FAMILY=${SPLIT_FAMILY}"
        "BENCHMARK=${BENCHMARK}"
        "DATA_ROOT=${DATA_ROOT}"
        "OUTPUT_DIR=${OUTPUT_DIR}"
        "DATA_SPLITS=${DATA_SPLITS}"
        "PROTOCOL=${PROTOCOL}"
        "REPRESENTATIONS=${REPRESENTATIONS}"
        "BATCH_SIZE=${BATCH_SIZE}"
        "FIT_MAX_LENGTH=${FIT_MAX_LENGTH}"
        "EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH}"
        "DTYPE=${DTYPE}"
        "FORCE=${FORCE}"
        "ENV_DIR=${ENV_DIR}"
        "HF_HOME=${HF_HOME}"
        "PYTHON_MODULE=${PYTHON_MODULE}"
        "CUDA_MODULE=${CUDA_MODULE}"
        "REPO_ROOT=${REPO_ROOT}"
    )
    if [[ -n "${FOLDS}" ]]; then
        env_vars+=("FOLDS=${FOLDS}")
    fi

    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    echo "${full_cmd[*]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${full_cmd[@]}"
    fi
}

if [[ "${MODE}" == "local" ]]; then
    if [[ ! -d "${ENV_DIR}" ]]; then
        echo "Error: environment not found at ${ENV_DIR}" >&2
        exit 1
    fi
    if [[ ! -d "${HF_HOME}" ]]; then
        echo "Error: HF_HOME does not exist: ${HF_HOME}" >&2
        exit 1
    fi
    for model in "${MODELS_ARRAY[@]}"; do
        run_for_model "${model}"
    done
    exit 0
fi

# ---- submit mode ------------------------------------------------------------
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}" >&2
    exit 1
fi
if [[ ! -d "${HF_HOME}" ]]; then
    echo "Error: HF_HOME does not exist: ${HF_HOME}" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "================================================================"
echo "Cache hidden states (icse26_val_1_4)"
echo "================================================================"
echo "  models:             ${MODELS_ARRAY[*]}"
echo "  split-family:       ${SPLIT_FAMILY}"
echo "  benchmark:          ${BENCHMARK}"
echo "  data-root:          ${DATA_ROOT}"
echo "  output-dir:         ${OUTPUT_DIR}"
echo "  data-splits:        ${DATA_SPLITS}"
echo "  protocol:           ${PROTOCOL}"
echo "  representations:    ${REPRESENTATIONS}"
echo "  folds:              ${FOLDS:-<all>}"
echo "  batch / lengths:    ${BATCH_SIZE} / fit=${FIT_MAX_LENGTH} eval=${EVAL_MAX_LENGTH}"
echo "  dtype:              ${DTYPE}"
echo "  force:              ${FORCE}"
echo "  resources:          ${PARTITION}, ${GPUS}×GPU, ${CPUS}×CPU, ${MEM}, ${TIME}"
echo "  env / hf-home:      ${ENV_DIR} / ${HF_HOME}"
if [[ -n "${DEPENDENCY}" ]]; then
    echo "  dependency:         ${DEPENDENCY}"
fi
echo "================================================================"

for model in "${MODELS_ARRAY[@]}"; do
    submit_for_model "${model}"
done
