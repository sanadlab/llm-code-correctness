#!/usr/bin/env bash
# =============================================================================
# run_majority_vote_layer_selection.sh
# =============================================================================
# Cross-fold majority-vote layer selection: each fold contributes two votes
# (cosine similarity + validation 1-of-4 accuracy); the layer with the most
# votes across folds wins.
#
# Prerequisites
#   - Cached hidden-state bundles and direction vectors.
#   - results.json per fold with val_per_layer (run val_one_of_4_from_cache.py
#     for icse26_val_1_4 before this script).
#
# Usage
#   source ../envs/llm-code-correctness/bin/activate
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_layer_selection.sh
#
# icse26_val_1_4 (default):
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_layer_selection.sh
#
# Original icse26:
#   DATASET_SOURCE=icse26 VECTORS_DATASET_SOURCE=icse26 RESULTS_ROOT=results \
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_layer_selection.sh
#
# Mutation dataset (when caches and results exist):
#   DATASET_SOURCE=mutation VECTORS_DATASET_SOURCE=mutation \
#   RESULTS_ROOT=results_mutation \
#   ./scripts/analysis/Bash_Scripts/run_majority_vote_layer_selection.sh
#
# Environment variables
#   MODEL                   HuggingFace model id
#   BENCHMARK               bigcodebench or humaneval
#   DATASET_SOURCE          split family for bundles + results (default: icse26_val_1_4)
#   VECTORS_DATASET_SOURCE  split family for direction vectors (default: icse26)
#   HIDDEN_STATES_ROOT      path to hidden states
#   VECTORS_ROOT            path to vectors
#   RESULTS_ROOT            path to results (default: results_icse26_val_1_4 or results)
#   SELECTION_SPLIT         validate or test (default: validate)
#   OUTPUT_DIR              output root (default: outputs/phase2_task2_majority_vote_layer)
#   FOLDS                   space-separated fold list (default: 0-9)
#   MAX_TASKS               optional task limit passed to Python
#   FOLD_TIEBREAK           one_of_4, pairwise_acc, or cosine when signals disagree
#                           (default: pairwise_acc for icse26, one_of_4 for icse26_val_1_4)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts/analysis"

MODEL="${MODEL:-codellama/CodeLlama-7b-Instruct-hf}"
BENCHMARK="${BENCHMARK:-bigcodebench}"
DATASET_SOURCE="${DATASET_SOURCE:-icse26_val_1_4}"
VECTORS_DATASET_SOURCE="${VECTORS_DATASET_SOURCE:-icse26}"
DISTRIBUTION="${DISTRIBUTION:-id}"
HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-${ROOT_DIR}/../hidden_states_re2230}"
VECTORS_ROOT="${VECTORS_ROOT:-${ROOT_DIR}/vectors}"
if [[ -z "${RESULTS_ROOT:-}" ]]; then
  if [[ "${DATASET_SOURCE}" == "icse26_val_1_4" ]]; then
    RESULTS_ROOT="${ROOT_DIR}/results_icse26_val_1_4"
  else
    RESULTS_ROOT="${ROOT_DIR}/results"
  fi
fi
SELECTION_SPLIT="${SELECTION_SPLIT:-validate}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2_task2_majority_vote_layer}"
PYTHON="${PYTHON:-python3}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
MAX_TASKS="${MAX_TASKS:-}"
FOLD_TIEBREAK="${FOLD_TIEBREAK:-}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL=${MODEL}"
echo "BENCHMARK=${BENCHMARK}"
echo "DATASET_SOURCE=${DATASET_SOURCE}"
echo "VECTORS_DATASET_SOURCE=${VECTORS_DATASET_SOURCE}"
echo "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
echo "VECTORS_ROOT=${VECTORS_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "SELECTION_SPLIT=${SELECTION_SPLIT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FOLDS=${FOLDS}"
echo "FOLD_TIEBREAK=${FOLD_TIEBREAK}"

CMD=(
  "${PYTHON}" -u scripts/analysis/majority_vote_layer_selection.py
  --hidden-states-root "${HIDDEN_STATES_ROOT}"
  --vectors-root "${VECTORS_ROOT}"
  --results-root "${RESULTS_ROOT}"
  --model "${MODEL}"
  --dataset-source "${DATASET_SOURCE}"
  --vectors-dataset-source "${VECTORS_DATASET_SOURCE}"
  --benchmark "${BENCHMARK}"
  --distribution "${DISTRIBUTION}"
  --selection-split "${SELECTION_SPLIT}"
  --framing neutral
  --representation response_avg
  --method mean_diff
  --fit-framing function
  --val-framing neutral
  --test-framing neutral
  --folds ${FOLDS}
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${FOLD_TIEBREAK}" ]]; then
  CMD+=(--fold-tiebreak "${FOLD_TIEBREAK}")
fi

if [[ -n "${MAX_TASKS}" ]]; then
  CMD+=(--max-tasks "${MAX_TASKS}")
fi

"${CMD[@]}"

echo "Finished. Outputs under ${OUTPUT_DIR}"
