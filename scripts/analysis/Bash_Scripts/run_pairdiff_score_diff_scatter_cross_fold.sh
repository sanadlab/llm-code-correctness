#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_score_diff_scatter_cross_fold.sh
# =============================================================================
# Cross-fold scatter grids for pairwise score_diff vs. cosine similarity.
#
# What it runs
#   plot_pairdiff_score_diff_scatter.py --mode cross-fold
#   One PNG per layer; each PNG has a subplot per fold.
#   Markers: o = correct beats paired incorrect, x = paired incorrect wins.
#
# Outputs
#   outputs/phase2_task2_score_diff/scatter_cross_fold/
#     layer_{L}_cosine_score_diff_scatter_cross_fold.png
#
# Prerequisites
#   Run run_pairdiff_score_diff_graphs.sh first (CSV_LAYERS=all by default).
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_score_diff_scatter_cross_fold.sh
#
# Environment variables
#   LAYERS                  layer filter (default: top-percentile union across folds)
#   PERCENTILE              percentile threshold (default: 90)
#   SCORE_DIFF_OUTPUT_DIR   CSV root (default: outputs/phase2_task2_score_diff/top_percentile)
#   SCATTER_OUTPUT_DIR      scatter PNG root
#   GRID_COLS               subplot columns (default: 3)
# =============================================================================
set -euo pipefail

# Cross-fold scatter grids: one PNG per layer, subplot per fold.
# Uses the union of each fold's layer set (top-percentile includes selected layer).
#
# Prerequisites:
#   Run run_pairdiff_score_diff_graphs.sh first (CSV_LAYERS=all by default).
#
# Usage:
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_score_diff_scatter_cross_fold.sh
#
# Optional environment variables:
#   PYTHON: python interpreter (default: python3)
#   MODELS: space-separated model names
#   BENCHMARK: benchmark name (default: bigcodebench)
#   SPLIT: data split (default: test)
#   FOLDS: space-separated fold indices (default: 0 1 2 3 4 5 6 7 8 9)
#   LAYERS: all | selected | top-percentile | comma-separated indices (default: top-percentile)
#   PERCENTILE: percentile when LAYERS=top-percentile (default: 90)
#   GRID_COLS: subplot columns in each grid image (default: 3)
#   SCORE_DIFF_OUTPUT_DIR: root of score-diff CSV outputs (default: outputs/phase2_task2_score_diff/top_percentile)
#   SCATTER_OUTPUT_DIR: where to write scatter grids (default: outputs/phase2_task2_score_diff/scatter_cross_fold)

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

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
PERCENTILE="${PERCENTILE:-90}"
GRID_COLS="${GRID_COLS:-3}"
SCORE_DIFF_OUTPUT_DIR="${SCORE_DIFF_OUTPUT_DIR:-outputs/phase2_task2_score_diff/top_percentile}"
SCATTER_OUTPUT_DIR="${SCATTER_OUTPUT_DIR:-outputs/phase2_task2_score_diff/scatter_cross_fold}"

model_to_safe_dir() {
  echo "$1" | sed 's|/|__|g'
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter '$PYTHON' not found." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  MODEL_SAFE="$(model_to_safe_dir "$model")"
  SCORE_DIFF_ROOT="${SCORE_DIFF_OUTPUT_DIR}/${MODEL_SAFE}/icse26/${BENCHMARK}/id/${SPLIT}"

  if [[ ! -d "$SCORE_DIFF_ROOT" ]]; then
    echo "ERROR: Score-diff root not found: $SCORE_DIFF_ROOT" >&2
    echo "Run run_pairdiff_score_diff_graphs.sh first." >&2
    exit 1
  fi

  echo "=========================================="
  echo "Cross-fold scatter grids for model: $model"
  echo "Score-diff root: $SCORE_DIFF_ROOT"
  echo "Layers: $LAYERS (percentile=$PERCENTILE)"
  echo "Folds: ${FOLDS[*]}"
  echo "=========================================="

  "$PYTHON" scripts/analysis/plot_pairdiff_score_diff_scatter.py \
    --mode cross-fold \
    --score-diff-root "$SCORE_DIFF_ROOT" \
    --folds "${FOLDS[@]}" \
    --layers "$LAYERS" \
    --percentile "$PERCENTILE" \
    --grid-cols "$GRID_COLS" \
    --results-root results \
    --output-dir "${SCATTER_OUTPUT_DIR}/${MODEL_SAFE}/icse26/${BENCHMARK}/id/${SPLIT}"

  echo ""
done

echo "=========================================="
echo "Cross-fold scatter grids written under: $SCATTER_OUTPUT_DIR"
echo "=========================================="
