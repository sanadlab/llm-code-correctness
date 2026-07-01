#!/bin/bash
# Launcher for fit-vectors + score from cached hidden states.
#
# Submits scripts/slurm/jobs/val_one_of_4_from_cache.sbatch with sensible
# defaults, or runs locally with --local. CPU-only; no model load.
#
# Usage:
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh [options]
#
# Examples:
#   # Submit default model to SLURM
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh
#
#   # Run locally (also used by the sbatch job)
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh --local
#
#   # Different model, subset of folds
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh \
#       --model codellama/CodeLlama-7b-Instruct-hf --folds "0 1 2"
#
#   # Dry-run (print sbatch command, don't submit)
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh --dry-run
#
#   # Sweep multiple models locally
#   MODELS="model/a model/b" bash scripts/slurm/launchers/val_one_of_4_from_cache.sh --local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/val_one_of_4_from_cache.sbatch"

# ---- defaults ---------------------------------------------------------------
MODEL="${MODEL:-codellama/CodeLlama-7b-Instruct-hf}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
SPLIT_FAMILY="${SPLIT_FAMILY:-icse26_val_1_4}"
DISTRIBUTION="${DISTRIBUTION:-id}"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-}"
VECTORS_ROOT="${VECTORS_ROOT:-vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-results_icse26_val_1_4}"
REPRESENTATION="${REPRESENTATION:-response_avg}"
METHOD="${METHOD:-mean_diff}"
FIT_FRAMING="${FIT_FRAMING:-function}"
VAL_FRAMING="${VAL_FRAMING:-neutral}"
TEST_FRAMING="${TEST_FRAMING:-neutral}"
LAYER_SELECTION_METRIC="${LAYER_SELECTION_METRIC:-one_of_4}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
SKIP_TEST=0

PARTITION="${PARTITION:-compute}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-02:00:00}"
ENV_DIR="${ENV_DIR:-/scratch/re2230/envs/llm-code-correctness}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"

MODE="submit"
DRY_RUN=0
DEPENDENCY=""
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/slurm/launchers/val_one_of_4_from_cache.sh [options]

Score validate/test from cached hidden states after fitting vectors on cached fit bundles.

Experiment options:
  --model NAME              HuggingFace model id              (default: codellama/CodeLlama-7b-Instruct-hf)
  --benchmark NAME          bigcodebench | humaneval          (default: bigcodebench)
  --split-family NAME       Split family for bundles/vectors/results (default: icse26_val_1_4)
  --distribution NAME       Distribution id                   (default: id)
  --hidden-states-root PATH Hidden-state cache root           (default: python script default)
  --vectors-root PATH       Direction vectors output root     (default: vectors)
  --results-root PATH       Results output root               (default: results_icse26_val_1_4)
  --representation NAME     response_avg | sequence_last | ... (default: response_avg)
  --method NAME             mean_diff | pca                   (default: mean_diff)
  --fit-framing NAME        function | neutral | concept      (default: function)
  --val-framing NAME        neutral | concept                 (default: neutral)
  --test-framing NAME       neutral | concept                 (default: neutral)
  --layer-selection-metric  one_of_4 | pairwise_acc           (default: one_of_4)
  --folds "N ..."           Space-separated fold indices      (default: "0..9")
  --skip-test               Only score validation bundles

SLURM options:
  --partition NAME --account NAME --qos NAME
  --cpus N --mem SIZE --time HH:MM:SS --dependency STR

Environment options:
  --env-dir PATH            Virtualenv root                   (default: /scratch/re2230/envs/llm-code-correctness)
  --python-bin PATH         Python interpreter                (default: <env-dir>/bin/python)

  --local                   Run locally instead of submitting
  --dry-run                 Print sbatch command without submitting
  --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                 MODEL="$2";                 shift 2 ;;
        --benchmark)             BENCHMARK="$2";             shift 2 ;;
        --split-family)          SPLIT_FAMILY="$2";          shift 2 ;;
        --distribution)          DISTRIBUTION="$2";          shift 2 ;;
        --hidden-states-root)    HIDDEN_STATES_ROOT="$2";    shift 2 ;;
        --vectors-root)          VECTORS_ROOT="$2";          shift 2 ;;
        --results-root)          RESULTS_ROOT="$2";          shift 2 ;;
        --representation)        REPRESENTATION="$2";        shift 2 ;;
        --method)                METHOD="$2";                shift 2 ;;
        --fit-framing)           FIT_FRAMING="$2";           shift 2 ;;
        --val-framing)           VAL_FRAMING="$2";           shift 2 ;;
        --test-framing)          TEST_FRAMING="$2";          shift 2 ;;
        --layer-selection-metric) LAYER_SELECTION_METRIC="$2"; shift 2 ;;
        --folds)                 FOLDS="$2";                 shift 2 ;;
        --skip-test)             SKIP_TEST=1;                  shift ;;
        --partition)             PARTITION="$2";             shift 2 ;;
        --account)               ACCOUNT="$2";               shift 2 ;;
        --qos)                   QOS="$2";                   shift 2 ;;
        --cpus)                  CPUS="$2";                  shift 2 ;;
        --mem)                   MEM="$2";                   shift 2 ;;
        --time)                  TIME="$2";                  shift 2 ;;
        --env-dir)               ENV_DIR="$2"; PYTHON_BIN="${ENV_DIR}/bin/python"; shift 2 ;;
        --python-bin)            PYTHON_BIN="$2";            shift 2 ;;
        --dependency)            DEPENDENCY="$2";            shift 2 ;;
        --local)                 MODE="local";               shift ;;
        --dry-run)               DRY_RUN=1;                  shift ;;
        --help|-h)               usage; exit 0 ;;
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

    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Error: python not found at ${PYTHON_BIN}" >&2
        exit 1
    fi

    local -a cmd=(
        "${PYTHON_BIN}" -u scripts/experiments/fit_vectors_from_cache.py
        --model "${model}"
        --benchmark "${BENCHMARK}"
        --split-family "${SPLIT_FAMILY}"
        --distribution "${DISTRIBUTION}"
        --vectors-root "${VECTORS_ROOT}"
        --results-root "${RESULTS_ROOT}"
        --representation "${REPRESENTATION}"
        --method "${METHOD}"
        --fit-framing "${FIT_FRAMING}"
        --val-framing "${VAL_FRAMING}"
        --test-framing "${TEST_FRAMING}"
        --layer-selection-metric "${LAYER_SELECTION_METRIC}"
        --folds ${FOLDS}
    )
    if [[ -n "${HIDDEN_STATES_ROOT}" ]]; then
        cmd+=(--hidden-states-root "${HIDDEN_STATES_ROOT}")
    fi
    if [[ "${SKIP_TEST}" == "1" ]]; then
        cmd+=(--skip-test)
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
        --job-name="val_one_of_4_${safe_model}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]]    && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]        && sbatch_args+=(--qos="${QOS}")
    [[ -n "${DEPENDENCY}" ]] && sbatch_args+=(--dependency="${DEPENDENCY}")

    local -a env_vars=(
        "MODEL=${model}"
        "BENCHMARK=${BENCHMARK}"
        "SPLIT_FAMILY=${SPLIT_FAMILY}"
        "DISTRIBUTION=${DISTRIBUTION}"
        "VECTORS_ROOT=${VECTORS_ROOT}"
        "RESULTS_ROOT=${RESULTS_ROOT}"
        "REPRESENTATION=${REPRESENTATION}"
        "METHOD=${METHOD}"
        "FIT_FRAMING=${FIT_FRAMING}"
        "VAL_FRAMING=${VAL_FRAMING}"
        "TEST_FRAMING=${TEST_FRAMING}"
        "LAYER_SELECTION_METRIC=${LAYER_SELECTION_METRIC}"
        "FOLDS=${FOLDS}"
        "SKIP_TEST=${SKIP_TEST}"
        "PYTHON_BIN=${PYTHON_BIN}"
        "REPO_ROOT=${REPO_ROOT}"
    )
    if [[ -n "${HIDDEN_STATES_ROOT}" ]]; then
        env_vars+=("HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}")
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
    for model in "${MODELS_ARRAY[@]}"; do
        run_for_model "${model}"
    done
    exit 0
fi

# ---- submit mode ------------------------------------------------------------
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "================================================================"
echo "Validation 1-of-4 from cache (fit vectors + score)"
echo "================================================================"
echo "  models:               ${MODELS_ARRAY[*]}"
echo "  benchmark:            ${BENCHMARK}"
echo "  split-family:         ${SPLIT_FAMILY}"
echo "  hidden-states-root:   ${HIDDEN_STATES_ROOT:-<python default>}"
echo "  vectors-root:         ${VECTORS_ROOT}"
echo "  results-root:         ${RESULTS_ROOT}"
echo "  representation:       ${REPRESENTATION}"
echo "  method:               ${METHOD}"
echo "  framings:             fit=${FIT_FRAMING}  val=${VAL_FRAMING}  test=${TEST_FRAMING}"
echo "  layer metric:         ${LAYER_SELECTION_METRIC}"
echo "  folds:                ${FOLDS}"
echo "  skip-test:            ${SKIP_TEST}"
echo "  resources:            ${PARTITION}, ${CPUS}×CPU, ${MEM}, ${TIME}"
echo "  python:               ${PYTHON_BIN}"
if [[ -n "${DEPENDENCY}" ]]; then
    echo "  dependency:           ${DEPENDENCY}"
fi
echo "================================================================"

for model in "${MODELS_ARRAY[@]}"; do
    submit_for_model "${model}"
done
