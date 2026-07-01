#!/bin/bash
# Submit the ablation fit+score sweep (stage 2).
#
<<<<<<< HEAD
# For each model, submits one ablation_fit_score.sbatch job that loops over
# every (representation, method, fit-framing, val-framing, fit-source,
# test-dataset) combination for that model: 144 combos per model, 4 models
# = 576 total.
=======
# For each model, submits one ablation_fit_score.sbatch job per dataset-source
# spec (see scripts/slurm/ablation_grid.sh; same specs as stage 1, one per
# mutants strategy x MUTANTS_BENCHMARKS entry).
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
#
# Splitting across collaborators: results are keyed by model first
# (ablation_results/<safe_model>/...), so two people can each run a disjoint
# subset of MODELS against the SAME shared RESULTS_DIR with zero path
# collisions -- no merge step needed afterward.
#
# Usage:
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh [--dry-run]
<<<<<<< HEAD
#   MODELS="model/a model/b" bash scripts/slurm/launchers/ablation_fit_score_sweep.sh
#   MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --local
=======
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --paper-configs [--local]
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --mutants-only [--local]
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --mutants-confounds [--local]
#   MODELS="model/a model/b" bash scripts/slurm/launchers/ablation_fit_score_sweep.sh
#   MODEL="Qwen/..." FIT_SOURCE=in-domain TEST_DATASET=bigcodebench \
#     bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --local
#   # Mutants only: paired + weighted x all mutant test targets (default)
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --mutants-only --local
#   # Mutants only, narrowed:
#   MUTANT_STRATEGIES="paired" MUTANT_TEST_DATASETS="mutants_only" \
#     MODEL="codellama/CodeLlama-7b-Instruct-hf" \
#     bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --mutants-only --local
#
#   # Mutants confounds only:
#   bash scripts/slurm/launchers/ablation_fit_score_sweep.sh --mutants-confounds --local
#
# Env vars for --mutants-only / --mutants-confounds (optional):
#   MUTANT_STRATEGIES     default: paired weighted
#   MUTANT_TEST_DATASETS  bigcodebench | humaneval | mutants_only
#   TEST_DATASET          shorthand to run one test target (same as MUTANT_TEST_DATASETS)
#   MUTANTS_BENCHMARKS    bcb_based (default) | he_based | "bcb_based he_based"
#                         e.g. MUTANTS_BENCHMARKS=he_based bash ... --mutants-only --local
#
# By default, skips combinations whose results.json files are already complete
# (--skip-existing). Pass --no-skip-existing to force a full re-run.
#
# --paper-configs restricts stage 2 to the paper subset: fit framings
# function/neutral/concept (no raw), and val framing must equal test framing
# (24 combos per dataset-source spec instead of 56).
#
# --val-on-test-dataset (mutants fit sources only): layer selection uses icse26
# validate for --test-dataset bigcodebench/humaneval instead of mutants validate.
# Results are written under val_ds=<benchmark>/ in the eval segment path.
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/ablation_fit_score.sbatch"
<<<<<<< HEAD
=======
# shellcheck source=../ablation_grid.sh
source "${SCRIPT_DIR}/../ablation_grid.sh"
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS <<< "${MODELS}"
else
<<<<<<< HEAD
    MODELS=(
        "Qwen/Qwen2.5-Coder-7B-Instruct"
        "infly/OpenCoder-8B-Instruct"
        "codellama/CodeLlama-7b-Instruct-hf"
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
fi

=======
    MODELS=("${ABLATION_MODELS[@]}")
fi

BENCHMARKS=("${ABLATION_BENCHMARKS[@]}")
OOD_SOURCES=("${ABLATION_OOD_SOURCES[@]}")
TEST_DATASETS=("${ABLATION_TEST_DATASETS[@]}")

>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
PARTITION="${PARTITION:-compute}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-04:00:00}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.3}"
<<<<<<< HEAD
=======
# Stage 2 is CPU-only (reuses cached hidden states; no model load), so no GPU
# is requested by default. Set GPUS and/or GPU_TYPE to request one, e.g.:
#   GPU_TYPE=v100 bash ... --mutants-only          # --gres=gpu:v100:1
#   GPU_TYPE=v100 GPUS=2 bash ... --mutants-only   # --gres=gpu:v100:2
#   GPUS=1 bash ... --mutants-only                 # --gpus=1 (any GPU on the partition)
GPUS="${GPUS:-}"
GPU_TYPE="${GPU_TYPE:-}"
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${ENV_DIR:-${SCRATCH}/envs/llm-code-correctness}"
else
    ENV_DIR="${ENV_DIR:-}"
fi

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-hidden_states}"
<<<<<<< HEAD
RESULTS_DIR="${RESULTS_DIR:-ablation_results}"
=======
RESULTS_DIR="${RESULTS_DIR:-full_ablation_results}"
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"
if [[ -n "${MUTANTS_BENCHMARKS:-}" ]]; then
    read -r -a MUTANTS_BENCHMARKS <<< "${MUTANTS_BENCHMARKS}"
else
    MUTANTS_BENCHMARKS=("bcb_based")
fi

MODE="submit"
DRY_RUN=0
<<<<<<< HEAD
=======
SKIP_EXISTING=1
MUTANTS_ONLY=0
MUTANTS_CONFOUNDS=0
PAPER_CONFIGS=0
VAL_ON_TEST_DATASET=0
FIT_SCORE_SCRIPT="${REPO_ROOT}/scripts/experiments/fit_and_score_from_cache.py"
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
for arg in "$@"; do
    case "${arg}" in
        --local) MODE="local" ;;
        --dry-run) DRY_RUN=1 ;;
<<<<<<< HEAD
        *)
            echo "Unknown option: ${arg}" >&2
            echo "Usage: $0 [--local | --dry-run]" >&2
=======
        --skip-existing) SKIP_EXISTING=1 ;;
        --no-skip-existing) SKIP_EXISTING=0 ;;
        --mutants-only|--mutant-only) MUTANTS_ONLY=1 ;;
        --mutants-confounds) MUTANTS_CONFOUNDS=1 ;;
        --paper-configs) PAPER_CONFIGS=1 ;;
        --val-on-test-dataset) VAL_ON_TEST_DATASET=1 ;;
        *)
            echo "Unknown option: ${arg}" >&2
            echo "Usage: $0 [--local | --dry-run | --mutants-only | --mutants-confounds | --paper-configs | --val-on-test-dataset | --skip-existing | --no-skip-existing]" >&2
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
            exit 1
            ;;
    esac
done

<<<<<<< HEAD
submit() {
    local model="$1"
    local safe_model="${model//\//__}"

    local -a sbatch_args=(
        --job-name="ablation_fit_score_${safe_model}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]] && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]     && sbatch_args+=(--qos="${QOS}")

    local -a env_vars=(
        "MODEL=${model}"
        "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
        "RESULTS_DIR=${RESULTS_DIR}"
        "DATA_ROOT=${DATA_ROOT}"
        "ENV_DIR=${ENV_DIR}"
        "PYTHON_MODULE=${PYTHON_MODULE}"
        "REPO_ROOT=${REPO_ROOT}"
    )
    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    echo "${full_cmd[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${full_cmd[@]}"
    fi
}

run_sweep_for_model() {
    local model="$1"
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local REPRESENTATIONS=("response_avg" "sequence_last")
    local METHODS=("mean_diff" "pca")
    local FIT_FRAMINGS=("function" "neutral" "concept")
    local VAL_FRAMINGS=("neutral" "concept")
    local TEST_DATASETS=("bigcodebench" "humaneval")
    local FIT_SOURCES=("in-domain" "mbpp" "synthetic")

    local n=0
    local REP METHOD FIT_FRAMING VAL_FRAMING FIT_SOURCE TEST_DATASET
=======
if [[ -n "${VAL_DATASET:-}" ]]; then
    VAL_ON_TEST_DATASET=0
fi

if [[ "${MUTANTS_ONLY}" -eq 1 && "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
    echo "Error: --mutants-only and --mutants-confounds are mutually exclusive." >&2
    exit 1
fi

if [[ -n "${MUTANT_STRATEGIES:-}" ]]; then
    read -r -a MUTANT_STRATEGIES_ARR <<< "${MUTANT_STRATEGIES}"
elif [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
    MUTANT_STRATEGIES_ARR=("${ABLATION_MUTANTS_CONFOUNDS_STRATEGIES[@]}")
else
    MUTANT_STRATEGIES_ARR=("${ABLATION_MUTANTS_STRATEGIES[@]}")
fi
if [[ -n "${MUTANT_TEST_DATASETS:-}" ]]; then
    read -r -a MUTANT_TEST_DATASETS_ARR <<< "${MUTANT_TEST_DATASETS}"
elif [[ -n "${TEST_DATASET:-}" && ( "${MUTANTS_ONLY}" -eq 1 || "${MUTANTS_CONFOUNDS}" -eq 1 ) ]]; then
    MUTANT_TEST_DATASETS_ARR=("${TEST_DATASET}")
elif [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
    MUTANT_TEST_DATASETS_ARR=("${ABLATION_MUTANTS_CONFOUNDS_TEST_DATASETS[@]}")
else
    MUTANT_TEST_DATASETS_ARR=("${ABLATION_MUTANTS_TEST_DATASETS[@]}")
fi

is_mutant_strategy() {
    local strategy="$1"
    local allowed
    for allowed in "${ABLATION_MUTANTS_STRATEGIES[@]}"; do
        if [[ "${strategy}" == "${allowed}" ]]; then
            return 0
        fi
    done
    return 1
}

is_confounds_strategy() {
    local strategy="$1"
    local allowed
    for allowed in "${ABLATION_MUTANTS_CONFOUNDS_STRATEGIES[@]}"; do
        if [[ "${strategy}" == "${allowed}" ]]; then
            return 0
        fi
    done
    return 1
}

is_mutant_test_dataset() {
    local test_dataset="$1"
    local allowed
    for allowed in "${ABLATION_MUTANTS_TEST_DATASETS[@]}" "${ABLATION_MUTANTS_CONFOUNDS_TEST_DATASETS[@]}"; do
        if [[ "${test_dataset}" == "${allowed}" ]]; then
            return 0
        fi
    done
    return 1
}

is_selected_mutant_strategy() {
    local strategy="$1"
    local allowed
    for allowed in "${MUTANT_STRATEGIES_ARR[@]}"; do
        if [[ "${strategy}" == "${allowed}" ]]; then
            return 0
        fi
    done
    return 1
}

is_icse26_benchmark() {
    local test_dataset="$1"
    [[ "${test_dataset}" == "bigcodebench" || "${test_dataset}" == "humaneval" ]]
}

val_dataset_args_for_spec() {
    local fit_source="$1"
    local test_dataset="$2"
    if [[ -n "${VAL_DATASET:-}" ]]; then
        printf '%s\n' "--val-dataset" "${VAL_DATASET}"
        return
    fi
    if [[ "${VAL_ON_TEST_DATASET}" -eq 1 ]] \
        && is_mutant_strategy "${fit_source}" \
        && is_icse26_benchmark "${test_dataset}"; then
        printf '%s\n' "--val-dataset" "${test_dataset}"
    fi
}

validate_mutant_filters() {
    local strategy test
    for strategy in "${MUTANT_STRATEGIES_ARR[@]}"; do
        if [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
            if ! is_confounds_strategy "${strategy}"; then
                echo "Error: invalid MUTANT_STRATEGIES entry '${strategy}' for --mutants-confounds; expected one of: ${ABLATION_MUTANTS_CONFOUNDS_STRATEGIES[*]}" >&2
                exit 1
            fi
        elif ! is_mutant_strategy "${strategy}"; then
            echo "Error: invalid MUTANT_STRATEGIES entry '${strategy}'; expected one of: ${ABLATION_MUTANTS_STRATEGIES[*]}" >&2
            exit 1
        fi
    done
    for test in "${MUTANT_TEST_DATASETS_ARR[@]}"; do
        if [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
            local allowed
            for allowed in "${ABLATION_MUTANTS_CONFOUNDS_TEST_DATASETS[@]}"; do
                if [[ "${test}" == "${allowed}" ]]; then
                    continue 2
                fi
            done
            echo "Error: invalid MUTANT_TEST_DATASETS entry '${test}' for --mutants-confounds; expected one of: ${ABLATION_MUTANTS_CONFOUNDS_TEST_DATASETS[*]}" >&2
            exit 1
        elif ! is_mutant_test_dataset "${test}"; then
            echo "Error: invalid MUTANT_TEST_DATASETS entry '${test}'; expected one of: ${ABLATION_MUTANTS_TEST_DATASETS[*]}" >&2
            exit 1
        fi
    done
}

validate_mutant_filters

submit() {
    local job_name="$1"; shift
    local -a env_vars=("$@")
    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a sbatch_args=(
        --job-name="${job_name}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]] && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]     && sbatch_args+=(--qos="${QOS}")
    if [[ -n "${GPU_TYPE}" ]]; then
        sbatch_args+=(--gres="gpu:${GPU_TYPE}:${GPUS:-1}")
    elif [[ -n "${GPUS}" ]]; then
        sbatch_args+=(--gpus="${GPUS}")
    fi

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    echo "${full_cmd[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${full_cmd[@]}"
    fi
}

run_sweep_for_spec() {
    local model="$1"
    local fit_source="$2"
    local test_dataset="$3"
    local mutants_benchmark="${4:-bcb_based}"

    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local REPRESENTATIONS=("${ABLATION_REPRESENTATIONS[@]}")
    local METHODS=("${ABLATION_METHODS[@]}")
    local FIT_FRAMINGS VAL_FRAMINGS TEST_FRAMINGS
    if [[ "${PAPER_CONFIGS}" -eq 1 ]]; then
        FIT_FRAMINGS=("${ABLATION_PAPER_FIT_FRAMINGS[@]}")
    else
        FIT_FRAMINGS=("${ABLATION_FIT_FRAMINGS[@]}")
    fi
    VAL_FRAMINGS=("${ABLATION_VAL_FRAMINGS[@]}")
    TEST_FRAMINGS=("${ABLATION_TEST_FRAMINGS[@]}")

    local n=0
    local skipped=0
    local REP METHOD FIT_FRAMING VAL_FRAMING TEST_FRAMING
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
    for REP in "${REPRESENTATIONS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for FIT_FRAMING in "${FIT_FRAMINGS[@]}"; do
                for VAL_FRAMING in "${VAL_FRAMINGS[@]}"; do
<<<<<<< HEAD
                    for FIT_SOURCE in "${FIT_SOURCES[@]}"; do
                        for TEST_DATASET in "${TEST_DATASETS[@]}"; do
                            n=$((n + 1))
                            local -a cmd=(python -u scripts/experiments/fit_and_score_from_cache.py
                                --model "${model}"
                                --representation "${REP}"
                                --method "${METHOD}"
                                --fit-framing "${FIT_FRAMING}"
                                --val-framing "${VAL_FRAMING}"
                                --fit-source "${FIT_SOURCE}"
                                --test-dataset "${TEST_DATASET}"
                                --hidden-states-root "${HIDDEN_STATES_ROOT}"
                                --results-dir "${RESULTS_DIR}"
                                --data-root "${DATA_ROOT}")
                            echo "[${n}] ${cmd[*]}"
                            if [[ "${DRY_RUN}" -eq 0 ]]; then
                                "${cmd[@]}"
                            fi
                        done
=======
                    for TEST_FRAMING in "${TEST_FRAMINGS[@]}"; do
                        if [[ "${PAPER_CONFIGS}" -eq 1 ]] && ablation_skip_non_paper_combo \
                            "${FIT_FRAMING}" "${VAL_FRAMING}" "${TEST_FRAMING}"; then
                            skipped=$((skipped + 1))
                            continue
                        fi
                        if ablation_skip_irrelevant_combo \
                            "${REP}" "${METHOD}" "${FIT_FRAMING}" \
                            "${VAL_FRAMING}" "${TEST_FRAMING}"; then
                            skipped=$((skipped + 1))
                            continue
                        fi
                        n=$((n + 1))
                        local -a cmd=(python -u "${FIT_SCORE_SCRIPT}"
                            --model "${model}"
                            --representation "${REP}"
                            --method "${METHOD}"
                            --fit-framing "${FIT_FRAMING}"
                            --val-framing "${VAL_FRAMING}"
                            --test-framing "${TEST_FRAMING}"
                            --fit-source "${fit_source}"
                            --test-dataset "${test_dataset}"
                            --hidden-states-root "${HIDDEN_STATES_ROOT}"
                            --results-dir "${RESULTS_DIR}"
                            --data-root "${DATA_ROOT}")
                        if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
                            cmd+=(--skip-existing)
                        fi
                        if [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
                            cmd+=(--mutants-confounds)
                        fi
                        if is_mutant_strategy "${fit_source}"; then
                            cmd+=(--mutants-benchmark "${mutants_benchmark}")
                        fi
                        while IFS= read -r val_arg; do
                            [[ -n "${val_arg}" ]] && cmd+=("${val_arg}")
                        done < <(val_dataset_args_for_spec "${fit_source}" "${test_dataset}")
                        echo "[${n}] ${cmd[*]}"
                        if [[ "${DRY_RUN}" -eq 0 ]]; then
                            "${cmd[@]}"
                        fi
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
                    done
                done
            done
        done
    done
<<<<<<< HEAD
    echo "Model ${model}: ${n} combinations"
}

run_sweep() {
=======
    echo "Model ${model} fit_source=${fit_source} test_dataset=${test_dataset}: ${n} combinations (${skipped} skipped as irrelevant)"
}

run_mutants_sweep_for_model() {
    local model="$1"
    local mutants_benchmark strategy test
    for mutants_benchmark in "${MUTANTS_BENCHMARKS[@]}"; do
        for strategy in "${MUTANT_STRATEGIES_ARR[@]}"; do
            for test in "${MUTANT_TEST_DATASETS_ARR[@]}"; do
                run_sweep_for_spec "${model}" "${strategy}" "${test}" "${mutants_benchmark}"
            done
        done
    done
}

run_sweep_for_model() {
    local model="$1"
    local benchmark ood test
    if [[ "${MUTANTS_ONLY}" -eq 1 || "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
        run_mutants_sweep_for_model "${model}"
        return
    fi
    for benchmark in "${BENCHMARKS[@]}"; do
        run_sweep_for_spec "${model}" "in-domain" "${benchmark}"
    done
    for ood in "${OOD_SOURCES[@]}"; do
        for test in "${TEST_DATASETS[@]}"; do
            run_sweep_for_spec "${model}" "${ood}" "${test}"
        done
    done
    run_mutants_sweep_for_model "${model}"
}

run_sweep() {
    if [[ -n "${MODEL:-}" && -n "${FIT_SOURCE:-}" && -n "${TEST_DATASET:-}" ]]; then
        if [[ "${MUTANTS_ONLY}" -eq 1 || "${MUTANTS_CONFOUNDS}" -eq 1 ]] && ! is_selected_mutant_strategy "${FIT_SOURCE}"; then
            echo "Error: --mutants-only/--mutants-confounds requires FIT_SOURCE in: ${MUTANT_STRATEGIES_ARR[*]} (got '${FIT_SOURCE}')" >&2
            exit 1
        fi
        if [[ "${MUTANTS_ONLY}" -eq 1 || "${MUTANTS_CONFOUNDS}" -eq 1 ]] && ! is_mutant_test_dataset "${TEST_DATASET}"; then
            if [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
                echo "Error: --mutants-confounds requires TEST_DATASET in: ${ABLATION_MUTANTS_CONFOUNDS_TEST_DATASETS[*]} (got '${TEST_DATASET}')" >&2
            else
                echo "Error: --mutants-only requires TEST_DATASET in: ${ABLATION_MUTANTS_TEST_DATASETS[*]} (got '${TEST_DATASET}')" >&2
            fi
            exit 1
        fi
        run_sweep_for_spec "${MODEL}" "${FIT_SOURCE}" "${TEST_DATASET}" "${MUTANTS_BENCHMARK:-bcb_based}"
        return
    fi

>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
    if [[ -n "${MODEL:-}" ]]; then
        run_sweep_for_model "${MODEL}"
        return
    fi

<<<<<<< HEAD
=======
    local model
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
    for model in "${MODELS[@]}"; do
        run_sweep_for_model "${model}"
    done
}

<<<<<<< HEAD
=======
common_export_vars() {
    local model="$1"
    printf '%s\n' \
        "MODEL=${model}" \
        "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}" \
        "RESULTS_DIR=${RESULTS_DIR}" \
        "DATA_ROOT=${DATA_ROOT}" \
        "ENV_DIR=${ENV_DIR}" \
        "PYTHON_MODULE=${PYTHON_MODULE}" \
        "REPO_ROOT=${REPO_ROOT}" \
        "SKIP_EXISTING=${SKIP_EXISTING}" \
        "PAPER_CONFIGS=${PAPER_CONFIGS}" \
        "VAL_ON_TEST_DATASET=${VAL_ON_TEST_DATASET}" \
        "VAL_DATASET=${VAL_DATASET:-}"
}

>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
if [[ "${MODE}" == "submit" ]]; then
    if [[ -z "${SCRATCH:-}" ]]; then
        echo "Error: SCRATCH is not set." >&2
        exit 1
    fi
    if [[ -z "${ENV_DIR}" ]]; then
        echo "Error: ENV_DIR is not set." >&2
        exit 1
    fi

    mkdir -p "${REPO_ROOT}/logs/slurm"
    cd "${REPO_ROOT}"

    for model in "${MODELS[@]}"; do
<<<<<<< HEAD
        submit "${model}"
=======
        safe_model="${model//\//__}"
        mapfile -t base_vars < <(common_export_vars "${model}")
        if [[ "${MUTANTS_ONLY}" -eq 0 && "${MUTANTS_CONFOUNDS}" -eq 0 ]]; then
            for benchmark in "${BENCHMARKS[@]}"; do
                submit "ablation_fit_score_id_${benchmark}_${safe_model}" \
                    "${base_vars[@]}" \
                    "FIT_SOURCE=in-domain" "TEST_DATASET=${benchmark}"
            done
            for ood in "${OOD_SOURCES[@]}"; do
                for test in "${TEST_DATASETS[@]}"; do
                    submit "ablation_fit_score_ood_${ood}_${test}_${safe_model}" \
                        "${base_vars[@]}" \
                        "FIT_SOURCE=${ood}" "TEST_DATASET=${test}"
                done
            done
        fi
        for mutants_benchmark in "${MUTANTS_BENCHMARKS[@]}"; do
            for strategy in "${MUTANT_STRATEGIES_ARR[@]}"; do
                for test in "${MUTANT_TEST_DATASETS_ARR[@]}"; do
                    job_tag="mut"
                    export_extras=("MUTANTS_BENCHMARK=${mutants_benchmark}")
                    if [[ "${MUTANTS_CONFOUNDS}" -eq 1 ]]; then
                        job_tag="mutconf"
                        export_extras+=("MUTANTS_CONFOUNDS=1")
                    fi
                    submit "ablation_fit_score_${job_tag}_${mutants_benchmark}_${strategy}_${test}_${safe_model}" \
                        "${base_vars[@]}" \
                        "${export_extras[@]}" \
                        "FIT_SOURCE=${strategy}" "TEST_DATASET=${test}"
                done
            done
        done
>>>>>>> 72a481761d3ed9f7c2b5e2e55e6cf321a1c8c815
    done
else
    run_sweep
fi
