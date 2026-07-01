#!/usr/bin/env bash
# =============================================================================
# run_pairdiff_task_max_scatter.sh
# =============================================================================
# Full pipeline: task-level (1-of-N) score ranking CSVs, density overlays,
# and cross-fold scatter grids.
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
#   outputs/phase2_task2_task_max/top_percentile/     CSVs + density overlays per fold
#   outputs/phase2_task2_task_max/scatter_cross_fold/ scatter PNGs per layer
#
# Prerequisites
#   Hidden-state bundles, directions, detailed.json, results.json per fold.
#   For icse26_val_1_4: run fit_vectors_from_cache (via run_val_one_of_4_from_cache.sh) first.
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_task_max_scatter.sh
#
# icse26_val_1_4 (default):
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_task_max_scatter.sh
#
# Original icse26 test split:
#   DATASET_SOURCE=icse26 VECTORS_DATASET_SOURCE=icse26 \
#   RESULTS_ROOT=results SPLIT=test \
#   HIDDEN_STATES_ROOT=../hidden_states_sohaila \
#   ./scripts/analysis/Bash_Scripts/run_pairdiff_task_max_scatter.sh
#
# Environment variables
#   DATASET_SOURCE          split family for bundles + results (default: icse26_val_1_4)
#   VECTORS_DATASET_SOURCE  split family for direction vectors (default: icse26)
#   HIDDEN_STATES_ROOT      hidden states root (default: ../hidden_states_re2230)
#   VECTORS_ROOT            vectors root (default: vectors)
#   RESULTS_ROOT            results root (default: results_icse26_val_1_4)
#   SPLIT                   validate or test (default: validate)
#   LAYERS                  scatter layer filter (default: top-percentile)
#   CSV_LAYERS              layers written to CSV (default: all)
#   PERCENTILE              top-percentile threshold (default: 90)
#   GRID_COLS               subplot columns (default: 3)
#   FOLDS_OVERRIDE          space-separated fold list (default: 0-9)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/scripts/analysis:$ROOT_DIR"

DATASET_SOURCE="${DATASET_SOURCE:-icse26_val_1_4}"
VECTORS_DATASET_SOURCE="${VECTORS_DATASET_SOURCE:-icse26_val_1_4}"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_icse26_val_1_4}"
PYTHON="${PYTHON:-/scratch/re2230/envs/llm-code-correctness/bin/python}"
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

echo "ROOT_DIR=${ROOT_DIR}"
echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SPLIT=${SPLIT}"
echo "FOLDS=${FOLDS[*]}"

for model in "${MODELS[@]}"; do
  MODEL_SAFE="$(model_to_safe_dir "$model")"

  echo "=========================================="
  echo "Generating task-max CSVs for model: ${model}"
  echo "Split: ${SPLIT} | CSV layers: ${CSV_LAYERS} | Folds: ${FOLDS[*]}"
  echo "=========================================="

  "$PYTHON" scripts/analysis/plot_pairdiff_task_max_score_diff.py \
    --dataset-source "${DATASET_SOURCE}" \
    --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
    --hidden-states-root "${HIDDEN_STATES_ROOT}" \
    --vectors-root "${VECTORS_ROOT}" \
    --results-root "${RESULTS_ROOT}" \
    --model "${model}" \
    --benchmark "${BENCHMARK}" \
    --split "${SPLIT}" \
    --folds "${FOLDS[@]}" \
    --layers "${LAYERS}" \
    --csv-layers "${CSV_LAYERS}" \
    --percentile "${PERCENTILE}" \
    --output-dir "${TASK_MAX_OUTPUT_DIR}"

  TASK_MAX_ROOT="${TASK_MAX_OUTPUT_DIR}/${MODEL_SAFE}/${DATASET_SOURCE}/${BENCHMARK}/id/${SPLIT}"

  if [[ ! -d "$TASK_MAX_ROOT" ]]; then
    echo "ERROR: Task-max root not found: ${TASK_MAX_ROOT}" >&2
    exit 1
  fi

  echo ""
  echo "=========================================="
  echo "Cross-fold task-max scatter grids for model: ${model}"
  echo "Task-max root: ${TASK_MAX_ROOT}"
  echo "Scatter layers: ${LAYERS} (percentile=${PERCENTILE})"
  echo "=========================================="

  "$PYTHON" scripts/analysis/plot_pairdiff_task_max_scatter.py \
    --mode cross-fold \
    --task-max-root "${TASK_MAX_ROOT}" \
    --folds "${FOLDS[@]}" \
    --layers "${LAYERS}" \
    --percentile "${PERCENTILE}" \
    --grid-cols "${GRID_COLS}" \
    --results-root "${RESULTS_ROOT}" \
    --output-dir "${SCATTER_OUTPUT_DIR}/${MODEL_SAFE}/${DATASET_SOURCE}/${BENCHMARK}/id/${SPLIT}"

  echo ""
done

echo "=========================================="
echo "Task-max CSVs: ${TASK_MAX_OUTPUT_DIR}"
echo "Task-max scatter grids: ${SCATTER_OUTPUT_DIR}"
echo "=========================================="
