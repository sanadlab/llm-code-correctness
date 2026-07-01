#!/usr/bin/env bash
# Compare tiered layer selection (val one-of-4 -> cosine -> deepest) vs
# validation pairwise-accuracy selection per fold.
#
# Supports mutation split families, icse26_val_1_4, and icse26 via DATASET_SOURCE.
#
# Examples:
#   # icse26_val_1_4 (default)
#   ./scripts/analysis/Bash_Scripts/run_compare_val_one_of_4_vs_val_acc_layer_selection.sh
#
#   # icse26
#   DATASET_SOURCE=icse26 RESULTS_ROOT=results ./scripts/analysis/Bash_Scripts/run_compare_val_one_of_4_vs_val_acc_layer_selection.sh
#
#   # mutation split family (e.g. s1_baseline)
#   DATASET_SOURCE=s1_baseline RESULTS_ROOT=results ./scripts/analysis/Bash_Scripts/run_compare_val_one_of_4_vs_val_acc_layer_selection.sh
#
# Environment overrides:
#   MODEL, BENCHMARK, DATASET_SOURCE, VECTORS_DATASET_SOURCE
#   HIDDEN_STATES_ROOT, VECTORS_ROOT, RESULTS_ROOT, OUTPUT_DIR, FOLDS, MAX_TASKS

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts/analysis:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-codellama/CodeLlama-7b-Instruct-hf}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
DATASET_SOURCE="${DATASET_SOURCE:-icse26_val_1_4}"
VECTORS_DATASET_SOURCE="${VECTORS_DATASET_SOURCE:-${DATASET_SOURCE}}"
DISTRIBUTION="${DISTRIBUTION:-id}"

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_icse26_val_1_4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2_task2_tiered_vs_val_acc_layer_selection}"
PYTHON="${PYTHON:-/scratch/re2230/envs/llm-code-correctness/bin/python}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
MAX_TASKS="${MAX_TASKS:-}"

MAX_TASKS_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  MAX_TASKS_ARGS=(--max-tasks "${MAX_TASKS}")
fi

echo "Running tiered vs val-acc layer selection comparison"
echo "  MODEL=${MODEL}"
echo "  DATASET_SOURCE=${DATASET_SOURCE}"
echo "  VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
echo "  RESULTS_ROOT=${RESULTS_ROOT}"
echo "  HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"

"${PYTHON}" -u scripts/analysis/compare_val_one_of_4_vs_val_acc_layer_selection.py \
  --hidden-states-root "${HIDDEN_STATES_ROOT}" \
  --vectors-root "${VECTORS_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --model "${MODEL}" \
  --dataset-source "${DATASET_SOURCE}" \
  --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
  --benchmark "${BENCHMARK}" \
  --distribution "${DISTRIBUTION}" \
  --framing neutral \
  --representation response_avg \
  --method mean_diff \
  --fit-framing function \
  --val-framing neutral \
  --test-framing neutral \
  --folds ${FOLDS} \
  --output-dir "${OUTPUT_DIR}" \
  "${MAX_TASKS_ARGS[@]}"

echo "Done."
