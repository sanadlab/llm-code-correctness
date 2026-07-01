#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_icse26_val_1_4_val_test.sh
# =============================================================================
# Run pairdiff score-diff density overlays AND task-max (CSV + density + scatter)
# for icse26_val_1_4 on both validate and test splits.
#
# Uses matching vectors under vectors/<model>/icse26_val_1_4/...
#
# What it runs (per model, per split, per fold):
#   1. plot_pairdiff_score_diff.py          → CSV + density overlay
#   2. plot_pairdiff_task_max_score_diff.py → CSV + density overlay
#   3. plot_pairdiff_task_max_scatter.py    → cross-fold scatter grids (once per split)
#
# Usage:
#   source /scratch/re2230/envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_icse26_val_1_4_val_test.sh
#
# Environment variables (optional):
#   MODELS                  space-separated model list
#   FOLDS_OVERRIDE          e.g. "0 1" for a quick smoke test
#   HIDDEN_STATES_ROOT      default: ../hidden_states_re2230
#   VECTORS_ROOT            default: vectors
#   RESULTS_ROOT            default: results_icse26_val_1_4
#   PERCENTILE              default: 90
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

DATASET_SOURCE="icse26_val_1_4"
VECTORS_DATASET_SOURCE="icse26_val_1_4"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_icse26_val_1_4}"
PYTHON="${PYTHON:-/scratch/re2230/envs/llm-code-correctness/bin/python}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
LAYERS="${LAYERS:-top-percentile}"
CSV_LAYERS="${CSV_LAYERS:-all}"
PERCENTILE="${PERCENTILE:-90}"
GRID_COLS="${GRID_COLS:-3}"
SCORE_DIFF_OUTPUT_DIR="${SCORE_DIFF_OUTPUT_DIR:-outputs/phase2_task2_score_diff/icse26_val_1_4}"
TASK_MAX_OUTPUT_DIR="${TASK_MAX_OUTPUT_DIR:-outputs/phase2_task2_task_max/icse26_val_1_4}"
SCATTER_OUTPUT_DIR="${SCATTER_OUTPUT_DIR:-outputs/phase2_task2_task_max/scatter_cross_fold/icse26_val_1_4}"

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_ARRAY=(${MODELS})
else
  MODEL_ARRAY=(
    "codellama/CodeLlama-7b-Instruct-hf"
  )
fi

FOLDS=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${FOLDS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  FOLDS=(${FOLDS_OVERRIDE})
fi

model_to_safe_dir() {
  echo "$1" | sed 's|/|__|g'
}

echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "FOLDS=${FOLDS[*]}"
echo "MODELS=${MODEL_ARRAY[*]}"

for model in "${MODEL_ARRAY[@]}"; do
  MODEL_SAFE="$(model_to_safe_dir "$model")"

  for split in validate test; do
    echo ""
    echo "######################################################################"
    echo "# ${model} | split=${split}"
    echo "######################################################################"

    for fold in "${FOLDS[@]}"; do
      echo "=========================================="
      echo "Score-diff density overlay | fold ${fold}"
      echo "=========================================="
      "${PYTHON}" -u scripts/analysis/plot_pairdiff_score_diff.py \
        --dataset-source "${DATASET_SOURCE}" \
        --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
        --hidden-states-root "${HIDDEN_STATES_ROOT}" \
        --vectors-root "${VECTORS_ROOT}" \
        --results-root "${RESULTS_ROOT}" \
        --model "${model}" \
        --benchmark "${BENCHMARK}" \
        --split "${split}" \
        --fold "${fold}" \
        --layers "${LAYERS}" \
        --csv-layers "${CSV_LAYERS}" \
        --percentile "${PERCENTILE}" \
        --output-dir "${SCORE_DIFF_OUTPUT_DIR}"
    done

    echo ""
    echo "=========================================="
    echo "Task-max CSV + density overlays | folds ${FOLDS[*]}"
    echo "=========================================="
    "${PYTHON}" -u scripts/analysis/plot_pairdiff_task_max_score_diff.py \
      --dataset-source "${DATASET_SOURCE}" \
      --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
      --hidden-states-root "${HIDDEN_STATES_ROOT}" \
      --vectors-root "${VECTORS_ROOT}" \
      --results-root "${RESULTS_ROOT}" \
      --model "${model}" \
      --benchmark "${BENCHMARK}" \
      --split "${split}" \
      --folds "${FOLDS[@]}" \
      --layers "${LAYERS}" \
      --csv-layers "${CSV_LAYERS}" \
      --percentile "${PERCENTILE}" \
      --output-dir "${TASK_MAX_OUTPUT_DIR}"

    TASK_MAX_ROOT="${TASK_MAX_OUTPUT_DIR}/${MODEL_SAFE}/${DATASET_SOURCE}/${BENCHMARK}/id/${split}"
    if [[ ! -d "$TASK_MAX_ROOT" ]]; then
      echo "ERROR: Task-max root not found: ${TASK_MAX_ROOT}" >&2
      exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Task-max cross-fold scatter grids"
    echo "=========================================="
    "${PYTHON}" -u scripts/analysis/plot_pairdiff_task_max_scatter.py \
      --mode cross-fold \
      --task-max-root "${TASK_MAX_ROOT}" \
      --folds "${FOLDS[@]}" \
      --layers "${LAYERS}" \
      --percentile "${PERCENTILE}" \
      --grid-cols "${GRID_COLS}" \
      --results-root "${RESULTS_ROOT}" \
      --output-dir "${SCATTER_OUTPUT_DIR}/${MODEL_SAFE}/${DATASET_SOURCE}/${BENCHMARK}/id/${split}"
  done
done

echo ""
echo "=========================================="
echo "Score-diff overlays: ${SCORE_DIFF_OUTPUT_DIR}"
echo "Task-max overlays:     ${TASK_MAX_OUTPUT_DIR}"
echo "Task-max scatters:   ${SCATTER_OUTPUT_DIR}"
echo "=========================================="
