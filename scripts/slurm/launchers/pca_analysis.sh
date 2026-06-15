#!/bin/bash
# Launcher for cache-only PCA visualization (scripts/analysis/pca.py).
#
# Usage:
#   bash scripts/slurm/launchers/pca_analysis.sh [options]
#
# Examples:
#   bash scripts/slurm/launchers/pca_analysis.sh
#   bash scripts/slurm/launchers/pca_analysis.sh \
#       --split-family s8_single_arithmetic \
#       --model codellama/CodeLlama-7b-Instruct-hf \
#       --folds 0 --tasks all
#   bash scripts/slurm/launchers/pca_analysis.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/pca_analysis.sbatch"

# ---- defaults (match README Mutation_DS PCA example) -------------------------
BENCHMARK="bigcodebench"
SPLIT_FAMILY="s8_single_arithmetic"
MODEL="codellama/CodeLlama-7b-Instruct-hf"
DATA_ROOT="dataset/bigcodebench/phase2"
FOLDS="0"
LAYERS="all"
TASKS="all"
APPROACH="both"
NORMALIZE_VECTOR=1
IMPLEMENTATION_MARKERS=1
METHOD="mean_diff"
FIT_FRAMING="function"
REPRESENTATION="response_avg"
VAL_FRAMING="neutral"
TEST_FRAMING="neutral"
HIDDEN_STATES_ROOT="hidden_states"
RESULTS_ROOT="results"
VECTORS_ROOT="vectors"
OUTPUT_DIR="visualizations/pca"

PARTITION=""
ACCOUNT=""
QOS=""
CPUS=8
MEM="32G"
TIME="08:00:00"
PYTHON_MODULE="python/3.11.3"

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${SCRATCH}/envs/llm-code-correctness"
else
    ENV_DIR=""
fi

DRY_RUN=0
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/slurm/launchers/pca_analysis.sh [options]

Submit cache-only PCA visualization to SLURM (CPU, no GPU).

Experiment options:
  --benchmark NAME          bigcodebench | humaneval                 (default: bigcodebench)
  --split-family NAME       e.g. icse26 | s8_single_arithmetic     (default: s8_single_arithmetic)
  --data-root PATH          Split manifest parent dir                (default: dataset/bigcodebench/phase2)
  --model NAME              HuggingFace model id                     (default: codellama/CodeLlama-7b-Instruct-hf)
  --folds "N ..."           Space-separated fold indices or "all"    (default: "0")
  --layers "..."            all | selected | layer ids               (default: "all")
  --tasks "..."             all | task ids | omit for fold-level     (default: "all")
  --approach NAME           A | B | both                             (default: both)
  --no-normalize-vector     Disable --normalize-vector
  --no-implementation-markers  Disable --implementation-markers
  --method NAME             mean_diff | pca                          (default: mean_diff)
  --representation NAME     prompt_avg|response_avg|...              (default: response_avg)
  --output-dir PATH         Output root                              (default: visualizations/pca)

SLURM options:
  --partition NAME --account NAME --qos NAME
  --cpus N --mem SIZE --time HH:MM:SS
  --env-dir PATH --python-module NAME
  --dry-run                 Print sbatch command without submitting
  --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)              BENCHMARK="$2";              shift 2 ;;
        --split-family)           SPLIT_FAMILY="$2";           shift 2 ;;
        --data-root)              DATA_ROOT="$2";              shift 2 ;;
        --model)                  MODEL="$2";                  shift 2 ;;
        --folds)                  FOLDS="$2";                  shift 2 ;;
        --layers)                 LAYERS="$2";                 shift 2 ;;
        --tasks)                  TASKS="$2";                  shift 2 ;;
        --approach)               APPROACH="$2";               shift 2 ;;
        --no-normalize-vector)    NORMALIZE_VECTOR=0;          shift ;;
        --no-implementation-markers) IMPLEMENTATION_MARKERS=0; shift ;;
        --method)                 METHOD="$2";                 shift 2 ;;
        --representation)         REPRESENTATION="$2";         shift 2 ;;
        --output-dir)             OUTPUT_DIR="$2";             shift 2 ;;
        --partition)              PARTITION="$2";              shift 2 ;;
        --account)                ACCOUNT="$2";                shift 2 ;;
        --qos)                    QOS="$2";                    shift 2 ;;
        --cpus)                   CPUS="$2";                   shift 2 ;;
        --mem)                    MEM="$2";                    shift 2 ;;
        --time)                   TIME="$2";                   shift 2 ;;
        --env-dir)                ENV_DIR="$2";                shift 2 ;;
        --python-module)          PYTHON_MODULE="$2";          shift 2 ;;
        --dry-run)                DRY_RUN=1;                   shift ;;
        --help|-h)                usage; exit 0 ;;
        *)                        echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${ENV_DIR}" ]]; then
    echo "Error: --env-dir required (SCRATCH unavailable)." >&2
    exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: virtualenv not found: ${ENV_DIR}" >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

sbatch_args=(
    --job-name="pca_${SPLIT_FAMILY}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
)
[[ -n "${PARTITION}" ]] && sbatch_args+=(--partition="${PARTITION}")
[[ -n "${ACCOUNT}" ]]   && sbatch_args+=(--account="${ACCOUNT}")
[[ -n "${QOS}" ]]       && sbatch_args+=(--qos="${QOS}")

env_vars=(
    "BENCHMARK=${BENCHMARK}"
    "SPLIT_FAMILY=${SPLIT_FAMILY}"
    "MODEL=${MODEL}"
    "DATA_ROOT=${DATA_ROOT}"
    "FOLDS=${FOLDS}"
    "LAYERS=${LAYERS}"
    "TASKS=${TASKS}"
    "APPROACH=${APPROACH}"
    "NORMALIZE_VECTOR=${NORMALIZE_VECTOR}"
    "IMPLEMENTATION_MARKERS=${IMPLEMENTATION_MARKERS}"
    "METHOD=${METHOD}"
    "FIT_FRAMING=${FIT_FRAMING}"
    "REPRESENTATION=${REPRESENTATION}"
    "VAL_FRAMING=${VAL_FRAMING}"
    "TEST_FRAMING=${TEST_FRAMING}"
    "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
    "RESULTS_ROOT=${RESULTS_ROOT}"
    "VECTORS_ROOT=${VECTORS_ROOT}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "ENV_DIR=${ENV_DIR}"
    "PYTHON_MODULE=${PYTHON_MODULE}"
    "REPO_ROOT=${REPO_ROOT}"
)
export_str=$(IFS=,; echo "${env_vars[*]}")

full_cmd=(
    sbatch
    "${sbatch_args[@]}"
    --export="${export_str}"
    "${JOB_SCRIPT}"
)

echo "================================================================"
echo "PCA analysis (cache-only, CPU)"
echo "================================================================"
echo "  benchmark:      ${BENCHMARK}"
echo "  split-family:   ${SPLIT_FAMILY}"
echo "  model:          ${MODEL}"
echo "  data-root:      ${DATA_ROOT}"
echo "  folds/layers:   ${FOLDS} / ${LAYERS}"
echo "  tasks:          ${TASKS}"
echo "  approach:       ${APPROACH}"
echo "  normalize:      ${NORMALIZE_VECTOR}"
echo "  impl markers:   ${IMPLEMENTATION_MARKERS}"
echo "  resources:      ${PARTITION:-auto}, ${CPUS}×CPU, ${MEM}, ${TIME}"
echo "  env:            ${ENV_DIR}"
echo ""
echo "  sbatch: ${full_cmd[*]}"
echo "================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "(dry-run — not submitted)"
    exit 0
fi

"${full_cmd[@]}"
