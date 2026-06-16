#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_score_diff_graphs.sh
# =============================================================================
# Batch pairwise score-diff analysis: CSV + density overlays per fold.
#
# What it runs
#   plot_pairdiff_score_diff.py per model × fold.
#   Compares correct vs. incorrect projection scores (detailed.json) per pair.
#   Overlay markers: o = score_diff >= 0, x = score_diff < 0.
#
# Outputs
#   outputs/phase2_task2_score_diff/top_percentile/
#     pairdiff_score_diff_cosine.csv per fold
#     top_90th_percentile_layers_cosine_score_diff_fold_{N}.png
#
# Prerequisites
#   Hidden-state bundles, directions, detailed.json, results.json per fold.
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_score_diff_graphs.sh
#
# Environment variables
#   HIDDEN_STATES_ROOT  hidden states root
#   LAYERS              overlay layer filter (default: top-percentile)
#   CSV_LAYERS          layers in CSV (default: all; needed for scatter grids)
#   PERCENTILE          top-percentile threshold (default: 90)
# =============================================================================
set -euo pipefail

# Run pairdiff score-difference analysis with cosine overlays per fold.
# Generates: CSV with score_diff + cosine_similarity, and overlay plots with x/o markers.
#
# Usage:
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_score_diff_graphs.sh
#
# Optional environment variables:
#   HIDDEN_STATES_ROOT: path to hidden states directory (default: ./../hidden_states_sohaila)
#   PYTHON: python interpreter to use (default: python3)
#   MODELS: space-separated list of model names to process
#   BENCHMARK: benchmark name (default: bigcodebench)
#   SPLIT: split name (default: test)
#   FOLDS: space-separated list of folds (default: 0 1 2 3 4 5 6 7 8 9)
#   LAYERS: layer selection for overlay plots (default: top-percentile)
#   CSV_LAYERS: layers written to CSV (default: all; needed for cross-fold scatter grids)
#   PERCENTILE: percentile threshold when LAYERS=top-percentile (default: 90)

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
BENCHMARK="bigcodebench"
SPLIT="validate"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
LAYERS="${LAYERS:-top-percentile}"
CSV_LAYERS="${CSV_LAYERS:-all}"
PERCENTILE="${PERCENTILE:-90}"
OUTPUT_DIR="outputs/phase2_task2_score_diff/top_percentile"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter '$PYTHON' not found." >&2
  exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(sys.version_info[:2])')
PY_MAJOR=$(echo "$PY_VERSION" | sed -E "s/\(|\)|,//g" | awk '{print $1}')
PY_MINOR=$(echo "$PY_VERSION" | sed -E "s/\(|\)|,//g" | awk '{print $2}')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 7 ]; }; then
  echo "ERROR: Python 3.7+ is required to run the analysis script. Found $PY_MAJOR.$PY_MINOR." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    echo "=========================================="
    echo "Processing model: $model"
    echo "Fold: $fold"
    echo "Layers: $LAYERS | CSV layers: $CSV_LAYERS"
    echo "Percentile: $PERCENTILE"
    echo "=========================================="
    "$PYTHON" scripts/analysis/plot_pairdiff_score_diff.py \
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
      --output-dir "$OUTPUT_DIR"
    echo ""
  done
done

echo "=========================================="
echo "Score-diff cosine overlay plots generated in: $OUTPUT_DIR"
echo "=========================================="
