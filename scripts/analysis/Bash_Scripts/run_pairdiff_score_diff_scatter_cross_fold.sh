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
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_score_diff_scatter_cross_fold.sh
#
# icse26_val_1_4 (default):
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_score_diff_scatter_cross_fold.sh
#
# Environment variables
#   DATASET_SOURCE          split family used in CSV paths (default: icse26_val_1_4)
#   RESULTS_ROOT            results root for layer filtering (default: results_icse26_val_1_4)
#   LAYERS                  layer filter (default: top-percentile union across folds)
#   PERCENTILE              percentile threshold (default: 90)
#   SCORE_DIFF_OUTPUT_DIR   CSV root (default: outputs/phase2_task2_score_diff/top_percentile)
#   SCATTER_OUTPUT_DIR      scatter PNG root
#   GRID_COLS               subplot columns (default: 3)
#   FOLDS_OVERRIDE          space-separated fold list (default: 0-9)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-icse26}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
PYTHON="${PYTHON:-python3}"
MODELS=(
  "codellama/CodeLlama-7b-Instruct-hf"
#  "Qwen/Qwen2.5-Coder-7B-Instruct"
#  "mistralai/Mistral-7B-Instruct-v0.3"
#  "infly/OpenCoder-8B-Instruct"
)
BENCHMARK="${BENCHMARK:-bigcodebench}"
SPLIT="${SPLIT:-validate}"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${FOLDS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  FOLDS=(${FOLDS_OVERRIDE})
fi
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

echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SPLIT=${SPLIT}"
echo "SCORE_DIFF_OUTPUT_DIR=${SCORE_DIFF_OUTPUT_DIR}"
echo "SCATTER_OUTPUT_DIR=${SCATTER_OUTPUT_DIR}"

for model in "${MODELS[@]}"; do
  MODEL_SAFE="$(model_to_safe_dir "$model")"
  SCORE_DIFF_ROOT="${SCORE_DIFF_OUTPUT_DIR}/${MODEL_SAFE}/${DATASET_SOURCE}/${BENCHMARK}/id/${SPLIT}"

  if [[ ! -d "$SCORE_DIFF_ROOT" ]]; then
    echo "ERROR: Score-diff root not found: $SCORE_DIFF_ROOT" >&2
    echo "Run run_pairdiff_score_diff_graphs.sh first." >&2
    exit 1
  fi

  echo "=========================================="
  echo "Cross-fold scatter grids for model: ${model}"
  echo "Score-diff root: ${SCORE_DIFF_ROOT}"
  echo "Layers: ${LAYERS} (percentile=${PERCENTILE})"
  echo "Folds: ${FOLDS[*]}"
  echo "=========================================="

  "$PYTHON" scripts/analysis/plot_pairdiff_score_diff_scatter.py \
    --mode cross-fold \
    --score-diff-root "$SCORE_DIFF_ROOT" \
    --folds "${FOLDS[@]}" \
    --layers "$LAYERS" \
    --percentile "$PERCENTILE" \
    --grid-cols "$GRID_COLS" \
    --results-root "$RESULTS_ROOT" \
    --output-dir "${SCATTER_OUTPUT_DIR}/${MODEL_SAFE}/${DATASET_SOURCE}/${BENCHMARK}/id/${SPLIT}"

  echo ""
done

echo "=========================================="
echo "Cross-fold scatter grids written under: ${SCATTER_OUTPUT_DIR}"
echo "=========================================="
