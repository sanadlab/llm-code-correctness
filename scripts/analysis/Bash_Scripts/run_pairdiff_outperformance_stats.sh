#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_outperformance_stats.sh
# =============================================================================
# Batch cross-fold statistics: layers beating the selected layer on cosine sim.
#
# What it runs
#   pairdiff_cosine_outperformance_stats.py per model.
#   Compares mean cosine similarity of candidate layers vs. the experiment
#   selected_layer from results.json within the top-percentile layer set.
#
# Outputs
#   outputs/phase2_task2_cosine/outperformance_stats/
#     JSON and CSV aggregate summaries per model
#
# Prerequisites
#   Per-fold summary.json under outputs/phase2_task2_cosine/top_percentile/
#   (from run_pairdiff_top_percentile_graphs.sh or plot_pairdiff_reading_cosine.py)
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_outperformance_stats.sh
#

# =============================================================================
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
)
BENCHMARK="bigcodebench"
SPLIT="test"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
LAYERS="${LAYERS:-top-percentile}"
PERCENTILE="${PERCENTILE:-90}"
COSINE_OUTPUT_DIR="${COSINE_OUTPUT_DIR:-outputs/phase2_task2_cosine/top_percentile}"
OUTPUT_DIR="outputs/phase2_task2_cosine/outperformance_stats"

for model in "${MODELS[@]}"; do
  echo "=========================================="
  echo "Outperformance stats for model: $model"
  echo "Layers: $LAYERS (percentile=$PERCENTILE)"
  echo "Folds: ${FOLDS[*]}"
  echo "=========================================="
  "$PYTHON" scripts/analysis/pairdiff_cosine_outperformance_stats.py \
    --model "$model" \
    --benchmark "$BENCHMARK" \
    --split "$SPLIT" \
    --folds "${FOLDS[@]}" \
    --layers "$LAYERS" \
    --percentile "$PERCENTILE" \
    --cosine-output-dir "$COSINE_OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
  echo ""
done

echo "Statistics written under: $OUTPUT_DIR"
