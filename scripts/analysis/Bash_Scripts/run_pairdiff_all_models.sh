#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_all_models.sh
# =============================================================================
# Batch pair-diff cosine analysis for multiple models on the experiment-selected
# layer (one layer per fold from results.json).
#
# What it runs
#   plot_pairdiff_reading_cosine.py per model × fold with --layers selected
#   --overlay, then one cross-fold overlay (--overlay-folds) per model.
#
# Outputs
#   outputs/phase2_task2_cosine/selected/<model>/.../overlay_layers_cosine.png
#   outputs/phase2_task2_cosine/selected/<model>/.../overlay_folds/
#
# Prerequisites
#   Hidden-state bundles, direction vectors, and results.json for each fold.
#   For icse26_val_1_4: run val_one_of_4_from_cache.py first to populate
#   results_icse26_val_1_4/.
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_all_models.sh
#
# icse26_val_1_4 (default):
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_all_models.sh
#
# Original icse26 test split:
#   DATASET_SOURCE=icse26 VECTORS_DATASET_SOURCE=icse26 \
#   RESULTS_ROOT=results SPLIT=test \
#   HIDDEN_STATES_ROOT=../hidden_states_sohaila \
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_all_models.sh
#
# Environment variables
#   DATASET_SOURCE          split family for bundles + results (default: icse26_val_1_4)
#   VECTORS_DATASET_SOURCE  split family for direction vectors (default: icse26)
#   HIDDEN_STATES_ROOT      path to hidden states (default: ../hidden_states_re2230)
#   VECTORS_ROOT            path to vectors (default: vectors)
#   RESULTS_ROOT            path to results (default: results_icse26_val_1_4)
#   SPLIT                   validate or test (default: validate for icse26_val_1_4)
#   OUTPUT_DIR              plot output root (default: outputs/phase2_task2_cosine/selected)
#   FOLDS                   space-separated fold list (default: 0-9)
#   MAX_TASKS               optional task limit passed to Python (default: 200)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-icse26_val_1_4}"
VECTORS_DATASET_SOURCE="${VECTORS_DATASET_SOURCE:-icse26}"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_icse26_val_1_4}"
SPLIT="${SPLIT:-validate}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2_task2_cosine/selected}"
PYTHON="${PYTHON:-python3}"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${FOLDS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  FOLDS=(${FOLDS_OVERRIDE})
fi
MAX_TASKS="${MAX_TASKS:-200}"

MODELS=(
  "codellama/CodeLlama-7b-Instruct-hf"
#  "Qwen/Qwen2.5-Coder-7B-Instruct"
#  "mistralai/Mistral-7B-Instruct-v0.3"
#  "infly/OpenCoder-8B-Instruct"
)

BENCHMARK="bigcodebench"

echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "VECTORS_ROOT=${VECTORS_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FOLDS=${FOLDS[*]}"

for model in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    echo "=========================================="
    echo "Processing model: ${model}"
    echo "Fold: ${fold}"
    echo "=========================================="
    "${PYTHON}" scripts/analysis/plot_pairdiff_reading_cosine.py \
      --dataset-source "${DATASET_SOURCE}" \
      --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
      --hidden-states-root "${HIDDEN_STATES_ROOT}" \
      --vectors-root "${VECTORS_ROOT}" \
      --results-root "${RESULTS_ROOT}" \
      --model "${model}" \
      --benchmark "${BENCHMARK}" \
      --split "${SPLIT}" \
      --fold "${fold}" \
      --layers selected \
      --overlay \
      --max-tasks "${MAX_TASKS}" \
      --output-dir "${OUTPUT_DIR}"
    echo ""
  done

  echo "=========================================="
  echo "Generating fold-overlay for model: ${model}"
  echo "=========================================="
  "${PYTHON}" scripts/analysis/plot_pairdiff_reading_cosine.py \
    --dataset-source "${DATASET_SOURCE}" \
    --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
    --hidden-states-root "${HIDDEN_STATES_ROOT}" \
    --vectors-root "${VECTORS_ROOT}" \
    --results-root "${RESULTS_ROOT}" \
    --model "${model}" \
    --benchmark "${BENCHMARK}" \
    --split "${SPLIT}" \
    --fold 0 \
    --layers selected \
    --overlay-folds \
    --folds "${FOLDS[@]}" \
    --max-tasks "${MAX_TASKS}" \
    --output-dir "${OUTPUT_DIR}"
  echo ""
done

echo "=========================================="
echo "All models processed. Outputs in:"
echo "  ${OUTPUT_DIR}"
echo "  (overlay figures under each model path in overlay_folds/)"
echo "=========================================="
