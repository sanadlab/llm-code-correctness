#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_task_max_scatter.sh
# =============================================================================
# Full pipeline: task-level (1-of-N) score ranking CSVs + cross-fold scatter grids.
#
# What it runs
#   1. plot_pairdiff_task_max_score_diff.py per fold → pairdiff_task_max_cosine.csv
#   2. plot_pairdiff_task_max_scatter.py --mode cross-fold
#
# Marker semantics (all task implementations from detailed.json):
#   o = correct has highest projection score in the task
#   x = any incorrect beats the correct sample
#
# Outputs
#   outputs/phase2_task2_task_max/top_percentile/     CSVs per fold
#   outputs/phase2_task2_task_max/scatter_cross_fold/ scatter PNGs per layer
#
# Prerequisites
#   Hidden-state bundles, directions, detailed.json, results.json per fold.
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_task_max_scatter.sh
#
# Environment variables
#   LAYERS       scatter layer filter (default: top-percentile; use all for every layer)
#   CSV_LAYERS   layers written to CSV (default: all)
#   PERCENTILE   top-percentile threshold (default: 90)
# =============================================================================
set -euo pipefail

# Generate task-max pairdiff CSVs and cross-fold scatter grids.
# Marker semantics (from detailed.json scores across all task implementations):
#   o = correct implementation has the highest score in the task
#   x = any incorrect implementation scores higher than the correct one
#
# Usage:
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_task_max_scatter.sh
#
# Optional environment variables:
#   HIDDEN_STATES_ROOT: path to hidden states directory (default: ./../hidden_states_sohaila)
#   PYTHON: python interpreter (default: python3)
#   MODELS: space-separated model names
#   BENCHMARK: benchmark name (default: bigcodebench)
#   SPLIT: data split (default: test)
#   FOLDS: space-separated fold indices (default: 0 1 2 3 4 5 6 7 8 9)
#   LAYERS: layer filter for scatter plots (default: top-percentile)
#   CSV_LAYERS: layers written to CSV (default: all)
#   PERCENTILE: percentile when LAYERS=top-percentile (default: 90)
#   GRID_COLS: subplot columns in cross-fold grids (default: 3)
#   TASK_MAX_OUTPUT_DIR: CSV output root (default: outputs/phase2_task2_task_max/top_percentile)
#   SCATTER_OUTPUT_DIR: scatter PNG output root (default: outputs/phase2_task2_task_max/scatter_cross_fold)

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-./../hidden_states_sohaila}"
PYTHON="${PYTHON:-python3}"
MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
#  "codellama/CodeLlama-7b-Instruct-hf"
#  "mistralai/Mistral-7B-Instruct-v0.3"
#  "infly/OpenCoder-8B-Instruct"
)
BENCHMARK="${BENCHMARK:-bigcodebench}"
SPLIT="${SPLIT:-test}"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
LAYERS="${LAYERS:-top-percentile}"
CSV_LAYERS="${CSV_LAYERS:-all}"
PERCENTILE="${PERCENTILE:-90}"
GRID_COLS="${GRID_COLS:-3}"
TASK_MAX_OUTPUT_DIR="${TASK_MAX_OUTPUT_DIR:-outputs/phase2_task2_task_max/top_percentile}"
SCATTER_OUTPUT_DIR="${SCATTER_OUTPUT_DIR:-outputs/phase2_task2_task_max/scatter_cross_fold}"

model_to_safe_dir() {
  echo "$1" | sed 's|/|__|g'
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter '$PYTHON' not found." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  MODEL_SAFE="$(model_to_safe_dir "$model")"

  echo "=========================================="
  echo "Generating task-max CSVs for model: $model"
  echo "Split: $SPLIT | CSV layers: $CSV_LAYERS"
  echo "=========================================="

  for fold in "${FOLDS[@]}"; do
    echo "Fold: $fold"
    "$PYTHON" scripts/analysis/plot_pairdiff_task_max_score_diff.py \
      --hidden-states-root "$HIDDEN_STATES_ROOT" \
      --vectors-root vectors \
      --results-root results \
      --model "$model" \
      --benchmark "$BENCHMARK" \
      --split "$SPLIT" \
      --fold "$fold" \
      --layers "$LAYERS" \
      --csv-layers "$CSV_LAYERS" \
      --percentile "$PERCENTILE" \
      --output-dir "$TASK_MAX_OUTPUT_DIR"
  done

  TASK_MAX_ROOT="${TASK_MAX_OUTPUT_DIR}/${MODEL_SAFE}/icse26/${BENCHMARK}/id/${SPLIT}"

  echo ""
  echo "=========================================="
  echo "Cross-fold task-max scatter grids for model: $model"
  echo "Task-max root: $TASK_MAX_ROOT"
  echo "Scatter layers: $LAYERS (percentile=$PERCENTILE)"
  echo "=========================================="

  "$PYTHON" scripts/analysis/plot_pairdiff_task_max_scatter.py \
    --mode cross-fold \
    --task-max-root "$TASK_MAX_ROOT" \
    --folds "${FOLDS[@]}" \
    --layers "$LAYERS" \
    --percentile "$PERCENTILE" \
    --grid-cols "$GRID_COLS" \
    --results-root results \
    --output-dir "${SCATTER_OUTPUT_DIR}/${MODEL_SAFE}/icse26/${BENCHMARK}/id/${SPLIT}"

  echo ""
done

echo "=========================================="
echo "Task-max CSVs: $TASK_MAX_OUTPUT_DIR"
echo "Task-max scatter grids: $SCATTER_OUTPUT_DIR"
echo "=========================================="
