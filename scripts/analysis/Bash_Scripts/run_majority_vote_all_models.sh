#!/usr/bin/env bash
# =============================================================================
# run_majority_vote_all_models.sh
# =============================================================================
# Run majority_vote_layer_selection.py for each model in MODELS.
#
# Usage
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_all_models.sh
#
# icse26_val_1_4 (default):
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_all_models.sh
#
# Original icse26:
#   DATASET_SOURCE=icse26 VECTORS_DATASET_SOURCE=icse26 RESULTS_ROOT=results \
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_all_models.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

MODELS=(
  "codellama/CodeLlama-7b-Instruct-hf"
#  "Qwen/Qwen2.5-Coder-7B-Instruct"
#  "mistralai/Mistral-7B-Instruct-v0.3"
#  "infly/OpenCoder-8B-Instruct"
)

for model in "${MODELS[@]}"; do
  echo "=========================================="
  echo "Majority vote for model: ${model}"
  echo "=========================================="
  MODEL="${model}" \
    bash scripts/analysis/Bash_Scripts/run_majority_vote_layer_selection.sh
  echo ""
done

echo "All models processed."
