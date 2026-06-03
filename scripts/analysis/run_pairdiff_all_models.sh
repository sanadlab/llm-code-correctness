#!/usr/bin/env bash
set -euo pipefail

# Run pairdiff cosine analysis for ALL models, plotting all layers per model.
# Generates: one histogram per layer per model

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
#   "codellama/CodeLlama-7b-Instruct-hf"
#   "mistralai/Mistral-7B-Instruct-v0.3"
#   "infly/OpenCoder-8B-Instruct"
)

BENCHMARK="bigcodebench"
SPLIT="test"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
MAX_TASKS=200

for model in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
    echo "=========================================="
    echo "Processing model: $model"
    echo "Fold: $fold"
    echo "=========================================="
    /scratch/re2230/envs/llm-code-correctness/bin/python3 scripts/analysis/plot_pairdiff_reading_cosine.py \
        --hidden-states-root /scratch/re2230/hidden_states_sohaila \
        --vectors-root vectors \
        --results-root results \
        --model "$model" \
        --benchmark "$BENCHMARK" \
        --split "$SPLIT" \
        --fold "$fold" \
        --layers selected \
        --max-tasks "$MAX_TASKS" \
        --overlay \
        --output-dir "outputs/phase2_task2_cosine/selected"
    echo ""
    done

    echo "=========================================="
    echo "Generating fold-overlay for model: $model"
    echo "=========================================="
    /scratch/re2230/envs/llm-code-correctness/bin/python3 scripts/analysis/plot_pairdiff_reading_cosine.py \
        --hidden-states-root /scratch/re2230/hidden_states_sohaila \
        --vectors-root vectors \
        --results-root results \
        --model "$model" \
        --benchmark "$BENCHMARK" \
        --split "$SPLIT" \
        --fold 0 \
        --layers selected \
        --max-tasks "$MAX_TASKS" \
        --overlay-folds \
        --folds "${FOLDS[@]}" \
        --output-dir "outputs/phase2_task2_cosine/selected"
    echo ""
done


echo "=========================================="
echo "All models processed. Outputs in:"
echo "  outputs/phase2_task2_cosine/selected"
echo "  (overlay figures under each model path in overlay_folds/)"
echo "=========================================="
