#!/usr/bin/env bash
set -euo pipefail

# Small initial sample wrapper for plot_pairdiff_reading_cosine.py
# This runs on Qwen / BigCodeBench / test / fold 0 using only the first 5 tasks.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"
/scratch/re2230/envs/llm-code-correctness/bin/python3 scripts/analysis/plot_pairdiff_reading_cosine.py \
  --hidden-states-root /scratch/re2230/hidden_states_sohaila \
  --vectors-root vectors \
  --results-root results \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --benchmark bigcodebench \
  --split test \
  --fold 0 \
  --layers 18,19,20 \
  --max-tasks 200 \
  --overlay \
  --output-dir outputs/phase2_task2_cosine/sample_run_layers18-20_tasks200
