#!/usr/bin/env bash
set -euo pipefail

echo "Starting bash script..."
echo "Script location: $0"

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
echo "ROOT_DIR=${ROOT_DIR}"

cd "$ROOT_DIR"
echo "Current directory after cd:"
pwd

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts/analysis"
echo "PYTHONPATH=${PYTHONPATH}"

MODEL="codellama/CodeLlama-7b-Instruct-hf"
BENCHMARK="bigcodebench"
DATASET_SOURCE="icse26_val_1_4"
VECTORS_DATASET_SOURCE="${VECTORS_DATASET_SOURCE:-icse26_val_1_4}"
DISTRIBUTION="id"

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_icse26_val_1_4}"
COSINE_SELECTION_SPLIT="${COSINE_SELECTION_SPLIT:-validate}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2_task2_cosine_selected_layer_table}"
MAX_TASKS="${MAX_TASKS:-}"
PYTHON="${PYTHON:-/scratch/re2230/envs/llm-code-correctness/bin/python}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

export PYTHONUNBUFFERED=1

echo "Checking Python script exists..."
ls -lh scripts/analysis/cosine_selection_layer_sim.py

echo "Checking important directories..."
echo "VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
ls -ld "${HIDDEN_STATES_ROOT}" || true
ls -ld "${VECTORS_ROOT}" || true
ls -ld "${RESULTS_ROOT}" || true

echo "Running Python script..."
"$PYTHON" --version
echo "Command: $PYTHON -u scripts/analysis/cosine_selection_layer_sim.py --cosine-selection-splits ${COSINE_SELECTION_SPLIT} --folds ${FOLDS} --output-dir ${OUTPUT_DIR}${MAX_TASKS:+ --max-tasks ${MAX_TASKS}}"

MAX_TASKS_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  MAX_TASKS_ARGS=(--max-tasks "${MAX_TASKS}")
fi

"$PYTHON" -u scripts/analysis/cosine_selection_layer_sim.py \
  --hidden-states-root "${HIDDEN_STATES_ROOT}" \
  --vectors-root "${VECTORS_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --model "${MODEL}" \
  --benchmark "${BENCHMARK}" \
  --dataset-source "${DATASET_SOURCE}" \
  --vectors-dataset-source "${VECTORS_DATASET_SOURCE}" \
  --distribution "${DISTRIBUTION}" \
  --cosine-selection-splits "${COSINE_SELECTION_SPLIT}" \
  --framing neutral \
  --representation response_avg \
  --method mean_diff \
  --fit-framing function \
  --val-framing neutral \
  --test-framing neutral \
  --folds ${FOLDS} \
  --output-dir "${OUTPUT_DIR}" 

echo "Finished Python script."
