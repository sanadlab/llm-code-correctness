#!/bin/bash
# Submit the full OOD nested CV sweep:
#   4 instruct models × 2 OOD sources × 2 test datasets × 2 protocols = 32 jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "infly/OpenCoder-8B-Instruct"
    "codellama/CodeLlama-7b-Instruct-hf"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
OOD_SOURCES=("synthetic" "mbpp")
TEST_DATASETS=("bigcodebench" "humaneval")
PROTOCOLS=("pca" "function")

# Per-test-dataset time budget (BCB is ~4x larger than HE).
declare -A TIMES=(
    [bigcodebench]="01:30:00"
    [humaneval]="00:45:00"
)

DRY_RUN=${DRY_RUN:-0}

for MODEL in "${MODELS[@]}"; do
    for OOD in "${OOD_SOURCES[@]}"; do
        for TEST in "${TEST_DATASETS[@]}"; do
            for PROTO in "${PROTOCOLS[@]}"; do
                EXTRA=()
                [[ "${DRY_RUN}" -eq 1 ]] && EXTRA+=(--dry-run)
                bash "${SCRIPT_DIR}/icse26_ood.sh" \
                    --model "${MODEL}" \
                    --ood-source "${OOD}" \
                    --test-dataset "${TEST}" \
                    --protocol "${PROTO}" \
                    --time "${TIMES[${TEST}]}" \
                    "${EXTRA[@]}" \
                    2>&1 | tail -3
            done
        done
    done
done
