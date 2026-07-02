#!/bin/bash
#SBATCH --job-name=rej_confound_audit
#SBATCH --output=logs/slurm/rej_confound_audit.%j.out
#SBATCH --error=logs/slurm/rej_confound_audit.%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
cd "${REPO_ROOT}"

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.3}
ENV_DIR=${ENV_DIR:-${SCRATCH:-}/envs/llm-code-correctness}

module purge
module load "${PYTHON_MODULE}"
source "${ENV_DIR}/bin/activate"

export BIGCODEBENCH_TIMEOUT_PER_TASK="${BIGCODEBENCH_TIMEOUT_PER_TASK:-360}"
export BIGCODEBENCH_GT_TIMEOUT_MULTIPLIER="${BIGCODEBENCH_GT_TIMEOUT_MULTIPLIER:-15}"
export VALIDATE_PARALLEL="${VALIDATE_PARALLEL:-31}"

python -u scripts/analyze_rejected_confounds.py --batch-size 50
