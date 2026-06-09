#!/usr/bin/env bash
set -euo pipefail

# Run pairdiff cosine analysis for top-percentile layers per fold.
# Generates: one cosine-similarity overlay plot per fold for the top percentile layers.
#
# Usage:
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/run_pairdiff_top_percentile_graphs.sh
#
# Optional environment variables:
#   HIDDEN_STATES_ROOT: path to hidden states directory (default: ./../hidden_states_sohaila)
#   PYTHON: python interpreter to use (default: python3)
#   MODELS: space-separated list of model names to process
#   BENCHMARK: benchmark name (default: humaneval)
#   SPLIT: split name (default: validate)
#   FOLDS: space-separated list of folds (default: 0 1 2 3 4 5 6 7 8 9)
#   PERCENTILE: percentile threshold for top layer selection (default: 90)

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-./../hidden_states_sohaila}"
PYTHON="${PYTHON:-python3}"
MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
#  "codellama/CodeLlama-7b-Instruct-hf"
#  "mistralai/Mistral-7B-Instruct-v0.3"
#  "infly/OpenCoder-8B-Instruct"
)
BENCHMARK="bigcodebench"
SPLIT="test"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
PERCENTILE="${PERCENTILE:-90}"
OUTPUT_DIR="outputs/phase2_task2_cosine/top_percentile"

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
    echo "Percentile: $PERCENTILE"
    echo "=========================================="
    "$PYTHON" scripts/analysis/plot_pairdiff_reading_cosine.py \
      --hidden-states-root "$HIDDEN_STATES_ROOT" \
      --vectors-root vectors \
      --results-root results \
      --model "$model" \
      --benchmark "$BENCHMARK" \
      --split "$SPLIT" \
      --fold "$fold" \
      --layers all \
      --top-percentile \
      --percentile "$PERCENTILE" \
      --output-dir "$OUTPUT_DIR"
    echo ""
  done

done

echo "=========================================="
echo "Top-percentile overlay plots generated in: $OUTPUT_DIR"
echo "=========================================="
