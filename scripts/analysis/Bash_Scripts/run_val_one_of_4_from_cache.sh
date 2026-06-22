#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts/analysis"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
SPLIT_FAMILY="${SPLIT_FAMILY:-icse26_val_1_4}"
VECTORS_SPLIT_FAMILY="${VECTORS_SPLIT_FAMILY:-icse26}"
DISTRIBUTION="${DISTRIBUTION:-id}"

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_icse26_val_1_4}"
PYTHON="${PYTHON:-python3}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "VECTORS_ROOT=${VECTORS_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"

"${PYTHON}" -u scripts/analysis/val_one_of_4_from_cache.py \
  --model "${MODEL}" \
  --benchmark "${BENCHMARK}" \
  --split-family "${SPLIT_FAMILY}" \
  --vectors-split-family "${VECTORS_SPLIT_FAMILY}" \
  --distribution "${DISTRIBUTION}" \
  --hidden-states-root "${HIDDEN_STATES_ROOT}" \
  --vectors-root "${VECTORS_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --representation response_avg \
  --method mean_diff \
  --fit-framing function \
  --val-framing neutral \
  --test-framing neutral \
  --layer-selection-metric one_of_4 \
  --folds ${FOLDS}

echo "Finished. Results under ${RESULTS_ROOT}"
