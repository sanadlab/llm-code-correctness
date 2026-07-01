#!/bin/bash
#SBATCH --job-name=rq1_matrix
#SBATCH --output=logs/slurm/rq1_matrix.%j.out
#SBATCH --error=logs/slurm/rq1_matrix.%j.err
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# RQ1 matrix fit+score sweep (batched slice of run_rq1_matrix_sweep.py).
# CPU-only; reads cached hidden states, no GPU.
#
# Overridable env vars (passed via --export by the launcher):
#   OFFSET / LIMIT       row slice (set by BATCH_INDEX * BATCH_SIZE)
#   BATCH_INDEX          zero-based batch id (for logging only)
#   BATCH_SIZE           rows per batch (for logging only)
#   HIDDEN_STATES_ROOT / RESULTS_DIR / DATA_ROOT / OUTPUT
#   JOBS                 parallel workers inside the job (default: SLURM_CPUS)
#   ENV_DIR / PYTHON_MODULE / REPO_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-50}
BATCH_INDEX=${BATCH_INDEX:-0}
BATCH_SIZE=${BATCH_SIZE:-50}

HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT:-/scratch/re2230/hidden_states_sohaila}
RESULTS_DIR=${RESULTS_DIR:-rq1_results}
DATA_ROOT=${DATA_ROOT:-dataset/icse26}
OUTPUT=${OUTPUT:-analysis/ablation/rq1_results.csv}
JOBS=${JOBS:-2}
SKIP_COMPLETED=${SKIP_COMPLETED:-0}

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}
PYTHON_BIN=${PYTHON_BIN:-${ENV_DIR}/bin/python}

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set." >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Skip `module load` — the venv on SCRATCH is self-contained and starts faster.
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "======================================"
echo "RQ1 matrix sweep (batched)"
echo "======================================"
echo "Repo:               ${REPO_ROOT}"
echo "Batch:              index=${BATCH_INDEX} offset=${OFFSET} limit=${LIMIT}"
echo "Hidden states root: ${HIDDEN_STATES_ROOT}"
echo "Results dir:        ${RESULTS_DIR}"
echo "Output CSV:         ${OUTPUT}"
echo "Data root:          ${DATA_ROOT}"
echo "Jobs (parallel):    ${JOBS}"
echo "Skip completed:     ${SKIP_COMPLETED}"
echo "Host:               $(hostname)"
echo "CPUs:               ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem:                ${SLURM_MEM_PER_NODE:-?}"
echo "======================================"

"${PYTHON_BIN}" -u scripts/experiments/run_rq1_matrix_sweep.py \
    --hidden-states-root "${HIDDEN_STATES_ROOT}" \
    --results-dir "${RESULTS_DIR}" \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT}" \
    --offset "${OFFSET}" \
    --limit "${LIMIT}" \
    --jobs "${JOBS}" \
    $( [[ "${SKIP_COMPLETED}" == "1" ]] && echo --skip-completed )

echo "Done."
