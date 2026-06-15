#!/bin/bash
#SBATCH --job-name=pca_analysis
#SBATCH --output=logs/slurm/pca_analysis.%j.out
#SBATCH --error=logs/slurm/pca_analysis.%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Cache-only PCA visualization (scripts/analysis/pca.py).
# No GPU / model inference — reads precomputed results, vectors, and hidden states.
#
# Overridable env vars (passed via --export by the launcher):
#   BENCHMARK                 bigcodebench | humaneval           (default: bigcodebench)
#   SPLIT_FAMILY              e.g. icse26 | s8_single_arithmetic (default: s8_single_arithmetic)
#   MODEL                     HF model id                        (default: codellama/CodeLlama-7b-Instruct-hf)
#   DATA_ROOT                 dataset root for split manifests     (default: dataset/bigcodebench/phase2)
#   FOLDS                     "0" | "0 1 ..." | "all"             (default: "0")
#   LAYERS                    "all" | "selected" | layer ids     (default: "all")
#   TASKS                     unset | "all" | task id list        (default: "all")
#   APPROACH                  A | B | both                       (default: both)
#   NORMALIZE_VECTOR          0 | 1                              (default: 1)
#   IMPLEMENTATION_MARKERS    0 | 1                              (default: 1)
#   METHOD / FIT_FRAMING / REPRESENTATION / VAL_FRAMING / TEST_FRAMING
#   HIDDEN_STATES_ROOT / RESULTS_ROOT / VECTORS_ROOT / OUTPUT_DIR
#   ENV_DIR / PYTHON_MODULE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

BENCHMARK=${BENCHMARK:-bigcodebench}
SPLIT_FAMILY=${SPLIT_FAMILY:-s8_single_arithmetic}
MODEL=${MODEL:-codellama/CodeLlama-7b-Instruct-hf}
DATA_ROOT=${DATA_ROOT:-dataset/bigcodebench/phase2}
FOLDS=${FOLDS:-0}
LAYERS=${LAYERS:-all}
TASKS=${TASKS:-all}
APPROACH=${APPROACH:-both}
NORMALIZE_VECTOR=${NORMALIZE_VECTOR:-1}
IMPLEMENTATION_MARKERS=${IMPLEMENTATION_MARKERS:-1}
METHOD=${METHOD:-mean_diff}
FIT_FRAMING=${FIT_FRAMING:-function}
REPRESENTATION=${REPRESENTATION:-response_avg}
VAL_FRAMING=${VAL_FRAMING:-neutral}
TEST_FRAMING=${TEST_FRAMING:-neutral}
HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT:-hidden_states}
RESULTS_ROOT=${RESULTS_ROOT:-results}
VECTORS_ROOT=${VECTORS_ROOT:-vectors}
OUTPUT_DIR=${OUTPUT_DIR:-visualizations/pca}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2
    exit 1
fi
if [[ ! -d "${ENV_DIR}" ]]; then
    echo "Error: environment not found at ${ENV_DIR}" >&2
    exit 1
fi

module purge
module load "${PYTHON_MODULE}"
source "${ENV_DIR}/bin/activate"

export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "======================================"
echo "PCA analysis (cache-only)"
echo "======================================"
echo "Repo:           ${REPO_ROOT}"
echo "Benchmark:      ${BENCHMARK}"
echo "Split family:   ${SPLIT_FAMILY}"
echo "Model:          ${MODEL}"
echo "Data root:      ${DATA_ROOT}"
echo "Method:         ${METHOD}"
echo "Representation: ${REPRESENTATION}"
echo "Framings:       fit=${FIT_FRAMING}  val=${VAL_FRAMING}  test=${TEST_FRAMING}"
echo "Folds:          ${FOLDS}"
echo "Layers:         ${LAYERS}"
echo "Tasks:          ${TASKS:-<fold-level>}"
echo "Approach:       ${APPROACH}"
echo "Normalize vec:  ${NORMALIZE_VECTOR}"
echo "Impl markers:   ${IMPLEMENTATION_MARKERS}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Env:            ${ENV_DIR}"
echo "Host:           $(hostname)"
echo "CPUs:           ${SLURM_CPUS_PER_TASK:-?}"
echo "======================================"

cmd=(
    python -u scripts/analysis/pca.py
    --benchmark "${BENCHMARK}"
    --split-family "${SPLIT_FAMILY}"
    --model "${MODEL}"
    --data-root "${DATA_ROOT}"
    --folds ${FOLDS}
    --layers ${LAYERS}
    --approach "${APPROACH}"
    --method "${METHOD}"
    --fit-framing "${FIT_FRAMING}"
    --representation "${REPRESENTATION}"
    --val-framing "${VAL_FRAMING}"
    --test-framing "${TEST_FRAMING}"
    --hidden-states-root "${HIDDEN_STATES_ROOT}"
    --results-root "${RESULTS_ROOT}"
    --vectors-root "${VECTORS_ROOT}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${TASKS:-}" ]]; then
    cmd+=(--tasks ${TASKS})
fi
if [[ "${NORMALIZE_VECTOR}" == "1" ]]; then
    cmd+=(--normalize-vector)
fi
if [[ "${IMPLEMENTATION_MARKERS}" == "1" ]]; then
    cmd+=(--implementation-markers)
fi

"${cmd[@]}"

echo "Done."
