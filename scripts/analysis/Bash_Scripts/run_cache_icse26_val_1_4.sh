#!/bin/bash
# Thin wrapper around the SLURM launcher (local mode).
#
# Cache hidden states for mutation-sourced icse26_val_1_4 splits.
#
# Usage:
#   bash scripts/analysis/Bash_Scripts/run_cache_icse26_val_1_4.sh
#
# Environment variables (optional overrides):
#   MODEL / SPLIT_FAMILY / DATA_ROOT / OUTPUT_DIR / DATA_SPLITS
#   PROTOCOL / REPRESENTATIONS / FOLDS / FORCE
#   MODELS                  space-separated list for multi-model local runs
#
# For SLURM submission, use the launcher directly:
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh --dry-run
#   bash scripts/slurm/launchers/cache_icse26_val_1_4.sh \
#       --data-splits "fit validate test"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

exec bash "${REPO_ROOT}/scripts/slurm/launchers/cache_icse26_val_1_4.sh" --local "$@"
