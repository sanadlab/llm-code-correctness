#!/bin/bash
#SBATCH --job-name=val_one_of_4
#SBATCH --output=logs/slurm/val_one_of_4_from_cache.%j.out
#SBATCH --error=logs/slurm/val_one_of_4_from_cache.%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Fit vectors from cached fit bundles, then score validate/test (icse26_val_1_4).
# CPU-only tensor math on already-cached activations; no GPU/model load.
# Runs scripts/experiments/fit_vectors_from_cache.py via the launcher --local.
#
# Overridable env vars (passed via --export by the launcher):
#   MODEL / BENCHMARK / SPLIT_FAMILY / DISTRIBUTION
#   HIDDEN_STATES_ROOT / VECTORS_ROOT / RESULTS_ROOT
#   REPRESENTATION / METHOD / FIT_FRAMING / VAL_FRAMING / TEST_FRAMING
#   LAYER_SELECTION_METRIC / FOLDS / SKIP_TEST
#   PYTHON_BIN / REPO_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL=${MODEL:-}
BENCHMARK=${BENCHMARK:-bigcodebench}
SPLIT_FAMILY=${SPLIT_FAMILY:-icse26_val_1_4}
DISTRIBUTION=${DISTRIBUTION:-id}
HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT:-}
VECTORS_ROOT=${VECTORS_ROOT:-vectors}
RESULTS_ROOT=${RESULTS_ROOT:-results_icse26_val_1_4}
REPRESENTATION=${REPRESENTATION:-response_avg}
METHOD=${METHOD:-mean_diff}
FIT_FRAMING=${FIT_FRAMING:-function}
VAL_FRAMING=${VAL_FRAMING:-neutral}
TEST_FRAMING=${TEST_FRAMING:-neutral}
LAYER_SELECTION_METRIC=${LAYER_SELECTION_METRIC:-one_of_4}
FOLDS=${FOLDS:-0 1 2 3 4 5 6 7 8 9}
SKIP_TEST=${SKIP_TEST:-0}

PYTHON_BIN=${PYTHON_BIN:-/scratch/re2230/envs/llm-code-correctness/bin/python}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -z "${MODEL}" ]]; then
    echo "Error: MODEL is not set." >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN
export MODEL BENCHMARK SPLIT_FAMILY DISTRIBUTION
export HIDDEN_STATES_ROOT VECTORS_ROOT RESULTS_ROOT
export REPRESENTATION METHOD FIT_FRAMING VAL_FRAMING TEST_FRAMING
export LAYER_SELECTION_METRIC FOLDS SKIP_TEST

# Skip `module load` — the venv on SCRATCH is self-contained and starts faster.
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "======================================"
echo "Fit vectors + score from cache"
echo "======================================"
echo "Repo:                 ${REPO_ROOT}"
echo "Model:                ${MODEL}"
echo "Benchmark:            ${BENCHMARK}"
echo "Split family:         ${SPLIT_FAMILY}"
echo "Hidden states root:   ${HIDDEN_STATES_ROOT:-<python default>}"
echo "Vectors root:         ${VECTORS_ROOT}"
echo "Results root:         ${RESULTS_ROOT}"
echo "Representation:       ${REPRESENTATION}"
echo "Method:               ${METHOD}"
echo "Framings:             fit=${FIT_FRAMING}  val=${VAL_FRAMING}  test=${TEST_FRAMING}"
echo "Layer metric:         ${LAYER_SELECTION_METRIC}"
echo "Folds:                ${FOLDS}"
echo "Skip test:            ${SKIP_TEST}"
echo "Host:                 $(hostname)"
echo "CPUs:                 ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem:                  ${SLURM_MEM_PER_NODE:-?}"
echo "======================================"

local_args=(--local)
if [[ "${SKIP_TEST}" == "1" ]]; then
    local_args+=(--skip-test)
fi

bash "${REPO_ROOT}/scripts/slurm/launchers/val_one_of_4_from_cache.sh" "${local_args[@]}"

echo "Done."
