#!/bin/bash
#SBATCH --job-name=build_frozen_2iso
#SBATCH --output=logs/slurm/build_frozen_2iso.%j.out
#SBATCH --error=logs/slurm/build_frozen_2iso.%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
cd "${REPO_ROOT}"

module purge
module load python/3.11.3
source "${SCRATCH}/envs/llm-code-correctness/bin/activate"

CONFIG="${CONFIG:-configs/phase1/2_isolated_run.yaml}"
REFERENCE_ROOT="${REFERENCE_ROOT:-dataset/isolated_bigcodebench}"
SAMPLE_N="${SAMPLE_N:-270}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

export CONFIG REFERENCE_ROOT SAMPLE_N SAMPLE_SEED
bash scripts/build_frozen_run.sh
