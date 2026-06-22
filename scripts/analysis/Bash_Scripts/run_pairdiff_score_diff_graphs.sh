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
#   For icse26_val_1_4: run val_one_of_4_from_cache.sh first.
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_score_diff_graphs.sh
#
# icse26_val_1_4 (default):
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_score_diff_graphs.sh
#
# Original icse26 test split:
#   DATASET_SOURCE=icse26 VECTORS_DATASET_SOURCE=icse26 \
#   RESULTS_ROOT=results SPLIT=test \
#   HIDDEN_STATES_ROOT=../hidden_states_sohaila \
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_score_diff_graphs.sh
#
# Environment variables
#   DATASET_SOURCE          split family for bundles + results (default: icse26_val_1_4)
#   VECTORS_DATASET_SOURCE  split family for direction vectors (default: icse26)
#   HIDDEN_STATES_ROOT      hidden states root (default: ../hidden_states_re2230)
#   VECTORS_ROOT            vectors root (default: vectors)
#   RESULTS_ROOT            results root (default: results_icse26_val_1_4)
#   SPLIT                   validate or test (default: validate)
#   LAYERS                  overlay layer filter (default: top-percentile)
#   CSV_LAYERS              layers in CSV (default: all; needed for scatter grids)
#   PERCENTILE              top-percentile threshold (default: 90)
#   OUTPUT_DIR              plot/CSV output root
#   FOLDS_OVERRIDE          space-separated fold list (default: 0-9)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-icse26}"
VECTORS_DATASET_SOURCE="${VECTORS_DATASET_SOURCE:-icse26}"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_sohaila}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
SPLIT="${SPLIT:-validate}"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
FOLDS=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${FOLDS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  FOLDS=(${FOLDS_OVERRIDE})
fi

MODELS=(
  "codellama/CodeLlama-7b-Instruct-hf"
#  "Qwen/Qwen2.5-Coder-7B-Instruct"
#  "mistralai/Mistral-7B-Instruct-v0.3"
#  "infly/OpenCoder-8B-Instruct"
)
BENCHMARK="bigcodebench"
LAYERS="${LAYERS:-top-percentile}"
CSV_LAYERS="${CSV_LAYERS:-all}"
PERCENTILE="${PERCENTILE:-90}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2_task2_score_diff/top_percentile}"

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

echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SPLIT=${SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FOLDS=${FOLDS[*]}"

for model in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    echo "=========================================="
    echo "Processing model: ${model}"
    echo "Fold: ${fold}"
    echo "Layers: ${LAYERS} | CSV layers: ${CSV_LAYERS}"
    echo "Percentile: ${PERCENTILE}"
    echo "=========================================="
    echo "Launching Python at $(date '+%Y-%m-%d %H:%M:%S')"
    "$PYTHON" -u scripts/analysis/plot_pairdiff_score_diff.py \
      --dataset-source "${DATASET_SOURCE}" \
      --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
      --hidden-states-root "${HIDDEN_STATES_ROOT}" \
      --vectors-root "${VECTORS_ROOT}" \
      --results-root "${RESULTS_ROOT}" \
      --model "${model}" \
      --benchmark "${BENCHMARK}" \
      --split "${SPLIT}" \
      --fold "${fold}" \
      --layers "${LAYERS}" \
      --csv-layers "${CSV_LAYERS}" \
      --percentile "${PERCENTILE}" \
      --output-dir "${OUTPUT_DIR}"
    echo ""
  done
done

echo "=========================================="
echo "Score-diff cosine overlay plots generated in: ${OUTPUT_DIR}"
echo "=========================================="
