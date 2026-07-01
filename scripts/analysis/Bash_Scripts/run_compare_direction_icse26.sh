#!/usr/bin/env bash
# Compare refitted direction vectors (from cached hidden states) against
# vectors/ for the original icse26 split family.
#
# Uses hidden_states_sohaila (symlink to Sohaila's icse26 caches).
#
# Usage:
#   source /scratch/re2230/envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_compare_direction_icse26.sh
#
# Environment variables:
#   MODEL              HuggingFace model id (default: codellama/CodeLlama-7b-Instruct-hf)
#   BENCHMARK          bigcodebench or humaneval (default: bigcodebench)
#   METHOD             mean_diff or pca (default: mean_diff)
#   REPRESENTATION     response_avg or sequence_last (default: response_avg)
#   FIT_FRAMING        default: function
#   HIDDEN_STATES_ROOT default: ../hidden_states_sohaila
#   VECTORS_ROOT       default: vectors
#   OUTPUT_VECTORS_ROOT default: vectors_refit_from_cache_icse26
#   SUMMARY_CSV        default: analysis/direction_compare/<model_slug>_icse26_summary.csv
#   FOLDS              space-separated fold list (default: all folds in vectors/)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

MODEL="${MODEL:-codellama/CodeLlama-7b-Instruct-hf}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
METHOD="${METHOD:-mean_diff}"
REPRESENTATION="${REPRESENTATION:-response_avg}"
FIT_FRAMING="${FIT_FRAMING:-function}"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_sohaila}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
OUTPUT_VECTORS_ROOT="${OUTPUT_VECTORS_ROOT:-${ROOT_DIR}/vectors_refit_from_cache_icse26}"
PYTHON="${PYTHON:-/scratch/re2230/envs/llm-code-correctness/bin/python}"

MODEL_SLUG="${MODEL//\//__}"
SUMMARY_CSV="${SUMMARY_CSV:-${ROOT_DIR}/analysis/direction_compare/${MODEL_SLUG}_icse26_summary.csv}"

ARGS=(
  --scan-vectors-root
  --model "$MODEL"
  --filter-split-family --split-family icse26
  --filter-benchmark --benchmark "$BENCHMARK"
  --filter-method --method "$METHOD"
  --filter-representation --representation "$REPRESENTATION"
  --filter-fit-framing --fit-framing "$FIT_FRAMING"
  --hidden-states-root "$HIDDEN_STATES_ROOT"
  --vectors-root "$VECTORS_ROOT"
  --output-vectors-root "$OUTPUT_VECTORS_ROOT"
  --summary-csv "$SUMMARY_CSV"
)

if [[ -n "${FOLDS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--folds ${FOLDS})
fi

echo "MODEL=${MODEL}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "SUMMARY_CSV=${SUMMARY_CSV}"

"$PYTHON" scripts/analysis/compare_direction_vectors.py "${ARGS[@]}"
