#!/bin/bash
# Thin wrapper around the SLURM launcher (local mode).
#
# Fit vectors from cached fit bundles, then score validate/test for
# icse26_val_1_4. Writes vectors/ and results_icse26_val_1_4/ by default.
#
# Usage:
#   bash scripts/analysis/Bash_Scripts/run_val_one_of_4_from_cache.sh
#
# Environment variables (optional overrides):
#   MODEL / BENCHMARK / SPLIT_FAMILY / DISTRIBUTION
#   HIDDEN_STATES_ROOT / VECTORS_ROOT / RESULTS_ROOT / FOLDS
#   LAYER_SELECTION_METRIC   one_of_4 (default) or pairwise_acc
#   MODELS                  space-separated list for multi-model local runs
#
# For SLURM submission, use the launcher directly:
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh
#   bash scripts/slurm/launchers/val_one_of_4_from_cache.sh --dry-run
#   MODEL="codellama/CodeLlama-7b-Instruct-hf" \
#     bash scripts/slurm/launchers/val_one_of_4_from_cache.sh --folds "0 1 2"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

exec bash "${REPO_ROOT}/scripts/slurm/launchers/val_one_of_4_from_cache.sh" --local "$@"
