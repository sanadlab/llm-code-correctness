#!/bin/bash
#SBATCH --job-name=ablation_fit_score_rel
#SBATCH --output=logs/slurm/ablation_fit_score_relevant.%j.out
#SBATCH --error=logs/slurm/ablation_fit_score_relevant.%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Ablation sweep — stage 2 (relevant configurations only).
# Same as ablation_fit_score.sbatch but skips response_avg whenever concept
# framing appears on fit, validate, or test (see ablation_grid.sh).
# CPU-only; reuses stage-1 hidden-state caches.
#
# Overridable env vars (passed via --export by the launcher):
#   MODEL              HF model id (required)
#   FIT_SOURCE         in-domain | mbpp | synthetic (required)
#   TEST_DATASET       bigcodebench | humaneval (required)
#   SKIP_EXISTING      1 to skip complete combos (default: 1)
#   HIDDEN_STATES_ROOT / RESULTS_DIR / DATA_ROOT
#   ENV_DIR / PYTHON_MODULE / REPO_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL=${MODEL:-}
FIT_SOURCE=${FIT_SOURCE:-}
TEST_DATASET=${TEST_DATASET:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT:-hidden_states}
RESULTS_DIR=${RESULTS_DIR:-relevant_ablation_results}
DATA_ROOT=${DATA_ROOT:-dataset/icse26}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -z "${MODEL}" ]]; then
    echo "Error: MODEL is not set." >&2
    exit 1
fi
if [[ -z "${FIT_SOURCE}" ]]; then
    echo "Error: FIT_SOURCE is not set." >&2
    exit 1
fi
if [[ -z "${TEST_DATASET}" ]]; then
    echo "Error: TEST_DATASET is not set." >&2
    exit 1
fi
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

export MODEL FIT_SOURCE TEST_DATASET HIDDEN_STATES_ROOT RESULTS_DIR DATA_ROOT SKIP_EXISTING

echo "======================================"
echo "Ablation fit+score sweep (relevant)"
echo "======================================"
echo "Repo:               ${REPO_ROOT}"
echo "Model:              ${MODEL}"
echo "Fit source:         ${FIT_SOURCE}"
echo "Test dataset:       ${TEST_DATASET}"
echo "Hidden states root: ${HIDDEN_STATES_ROOT}"
echo "Results dir:        ${RESULTS_DIR}"
echo "Data root:          ${DATA_ROOT}"
echo "Skip existing:      ${SKIP_EXISTING}"
echo "Host:               $(hostname)"
echo "CPUs:               ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem:                ${SLURM_MEM_PER_NODE:-?}"
echo "======================================"

local_args=(--local)
if [[ "${SKIP_EXISTING}" == "0" ]]; then
    local_args+=(--no-skip-existing)
fi

bash "${REPO_ROOT}/scripts/slurm/launchers/ablation_fit_score_relevant_sweep.sh" "${local_args[@]}"

echo "Done."
