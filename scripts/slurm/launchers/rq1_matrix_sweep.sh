#!/bin/bash
# Submit the RQ1 matrix fit+score sweep in batches.
#
# The full matrix has 384 rows. Run a slice at a time so each SLURM job
# stays short and you can pause between batches on the cluster.
#
# Usage:
#   # Submit batch 0 (rows 0-49)
#   BATCH_SIZE=50 BATCH_INDEX=0 bash scripts/slurm/launchers/rq1_matrix_sweep.sh
#
#   # Submit batch 1 (rows 50-99) after the first finishes
#   BATCH_SIZE=50 BATCH_INDEX=1 bash scripts/slurm/launchers/rq1_matrix_sweep.sh
#
#   # Submit ALL batches at once (parallel on the cluster)
#   BATCH_SIZE=50 BATCH_INDEX=0 BATCH_END=7 bash scripts/slurm/launchers/rq1_matrix_sweep.sh
#
#   # Submit ALL batches as a dependency chain (batch N+1 starts after batch N)
#   BATCH_SIZE=50 bash scripts/slurm/launchers/rq1_matrix_sweep.sh --chain
#
#   # Chain only batches 1-3 (e.g. resume after batch 0 finished manually)
#   BATCH_SIZE=50 BATCH_START=1 BATCH_END=3 bash scripts/slurm/launchers/rq1_matrix_sweep.sh --chain
#
#   # List all batches without submitting
#   BATCH_SIZE=50 bash scripts/slurm/launchers/rq1_matrix_sweep.sh --list-batches
#
#   # Run one batch locally (no sbatch)
#   BATCH_SIZE=10 BATCH_INDEX=0 bash scripts/slurm/launchers/rq1_matrix_sweep.sh --local
#
#   # Refresh the full CSV from accumulated results (no new runs)
#   bash scripts/slurm/launchers/rq1_matrix_sweep.sh --aggregate-only
#
#   # Dry-run the sbatch command for batch 2
#   BATCH_SIZE=50 BATCH_INDEX=2 bash scripts/slurm/launchers/rq1_matrix_sweep.sh --dry-run
#
# Env vars:
#   BATCH_SIZE     rows per batch (default: 50)
#   BATCH_INDEX    zero-based batch to submit/run (default: 0)
#   BATCH_START    alias for BATCH_INDEX
#   BATCH_END      if set with BATCH_START, submit every batch in [START, END]
#   HIDDEN_STATES_ROOT / RESULTS_DIR / DATA_ROOT / OUTPUT / JOBS / SKIP_COMPLETED
#   PARTITION / ACCOUNT / QOS / CPUS / MEM / TIME / ENV_DIR / PYTHON_MODULE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/rq1_matrix_sweep.sbatch"

BATCH_SIZE="${BATCH_SIZE:-50}"
BATCH_INDEX="${BATCH_INDEX:-${BATCH_START:-0}}"
BATCH_END="${BATCH_END:-}"

PARTITION="${PARTITION:-compute}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-04:00:00}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.3}"

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${ENV_DIR:-${SCRATCH}/envs/llm-code-correctness}"
else
    ENV_DIR="${ENV_DIR:-}"
fi

HIDDEN_STATES_ROOT="${HIDDEN_STATES_ROOT:-/scratch/re2230/hidden_states_sohaila}"
RESULTS_DIR="${RESULTS_DIR:-rq1_results}"
DATA_ROOT="${DATA_ROOT:-dataset/icse26}"
OUTPUT="${OUTPUT:-analysis/ablation/rq1_results.csv}"
JOBS="${JOBS:-2}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

MODE="submit"
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --local) MODE="local" ;;
        --dry-run) DRY_RUN=1 ;;
        --aggregate-only) MODE="aggregate" ;;
        --list-batches) MODE="list" ;;
        --chain) MODE="chain" ;;
        *)
            echo "Unknown option: ${arg}" >&2
            echo "Usage: $0 [--local | --dry-run | --aggregate-only | --list-batches | --chain]" >&2
            exit 1
            ;;
    esac
done

python_bin() {
    if [[ -n "${ENV_DIR:-}" && -x "${ENV_DIR}/bin/python" ]]; then
        echo "${ENV_DIR}/bin/python"
    else
        echo "python3"
    fi
}

total_rows() {
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    "$(python_bin)" -c "from scripts.analysis.rq1_matrix_spec import matrix_row_count; print(matrix_row_count())"
}

num_batches() {
    local total="$1"
    echo $(( (total + BATCH_SIZE - 1) / BATCH_SIZE ))
}

batch_offset() {
    local idx="$1"
    echo $(( idx * BATCH_SIZE ))
}

batch_limit() {
    local idx="$1"
    local total="$2"
    local offset
    offset=$(batch_offset "${idx}")
    local remaining=$(( total - offset ))
    if [[ "${remaining}" -lt "${BATCH_SIZE}" ]]; then
        echo "${remaining}"
    else
        echo "${BATCH_SIZE}"
    fi
}

list_batches() {
    local total n_batches idx offset lim end_row
    total=$(total_rows)
    n_batches=$(num_batches "${total}")
    echo "RQ1 matrix: ${total} rows, batch_size=${BATCH_SIZE}, num_batches=${n_batches}"
    for (( idx=0; idx<n_batches; idx++ )); do
        offset=$(batch_offset "${idx}")
        lim=$(batch_limit "${idx}" "${total}")
        end_row=$(( offset + lim - 1 ))
        printf "  batch %2d: rows %3d-%3d (offset=%d limit=%d)\n" \
            "${idx}" "${offset}" "${end_row}" "${offset}" "${lim}"
    done
}

run_local_batch() {
    local idx="$1"
    local total offset lim
    total=$(total_rows)
    offset=$(batch_offset "${idx}")
    lim=$(batch_limit "${idx}" "${total}")
    if [[ "${offset}" -ge "${total}" ]]; then
        echo "Error: batch ${idx} is out of range (total rows=${total})." >&2
        exit 1
    fi

    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local py
    py="$(python_bin)"

    local -a cmd=(
        "${py}" -u scripts/experiments/run_rq1_matrix_sweep.py
        --hidden-states-root "${HIDDEN_STATES_ROOT}"
        --results-dir "${RESULTS_DIR}"
        --data-root "${DATA_ROOT}"
        --output "${OUTPUT}"
        --offset "${offset}"
        --limit "${lim}"
        --jobs "${JOBS}"
    )
    echo "[batch ${idx}] ${cmd[*]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${cmd[@]}"
    fi
}

run_aggregate_only() {
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    local py
    py="$(python_bin)"

    local -a cmd=(
        "${py}" -u scripts/experiments/run_rq1_matrix_sweep.py
        --hidden-states-root "${HIDDEN_STATES_ROOT}"
        --results-dir "${RESULTS_DIR}"
        --data-root "${DATA_ROOT}"
        --output "${OUTPUT}"
        --aggregate-only
    )
    echo "${cmd[*]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${cmd[@]}"
    fi
}

submit_batch() {
    local idx="$1"
    local dependency="${2:-}"
    local total offset lim
    total=$(total_rows)
    offset=$(batch_offset "${idx}")
    lim=$(batch_limit "${idx}" "${total}")
    if [[ "${offset}" -ge "${total}" ]]; then
        echo "Error: batch ${idx} is out of range (total rows=${total})." >&2
        exit 1
    fi

    local end_row=$(( offset + lim - 1 ))
    local -a sbatch_args=(
        --job-name="rq1_matrix_b${idx}_${offset}-${end_row}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
    )
    [[ -n "${ACCOUNT}" ]] && sbatch_args+=(--account="${ACCOUNT}")
    [[ -n "${QOS}" ]]     && sbatch_args+=(--qos="${QOS}")
    [[ -n "${dependency}" ]] && sbatch_args+=(--dependency="${dependency}")

    local -a env_vars=(
        "OFFSET=${offset}"
        "LIMIT=${lim}"
        "BATCH_INDEX=${idx}"
        "BATCH_SIZE=${BATCH_SIZE}"
        "HIDDEN_STATES_ROOT=${HIDDEN_STATES_ROOT}"
        "RESULTS_DIR=${RESULTS_DIR}"
        "DATA_ROOT=${DATA_ROOT}"
        "OUTPUT=${OUTPUT}"
        "JOBS=${JOBS}"
        "SKIP_COMPLETED=${SKIP_COMPLETED}"
        "ENV_DIR=${ENV_DIR}"
        "PYTHON_BIN=${ENV_DIR}/bin/python"
        "PYTHON_MODULE=${PYTHON_MODULE}"
        "REPO_ROOT=${REPO_ROOT}"
    )
    local export_str
    export_str=$(IFS=,; echo "${env_vars[*]}")

    local -a full_cmd=(sbatch "${sbatch_args[@]}" --export="${export_str}" "${JOB_SCRIPT}")
    if [[ -n "${dependency}" ]]; then
        echo "[batch ${idx}] rows ${offset}-${end_row} (after ${dependency}): ${full_cmd[*]}" >&2
    else
        echo "[batch ${idx}] rows ${offset}-${end_row}: ${full_cmd[*]}" >&2
    fi
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        local out
        out=$("${full_cmd[@]}")
        echo "${out}" >&2
        # sbatch prints: "Submitted batch job 12345"
        echo "${out}" | awk '{print $NF}'
    fi
}

submit_range() {
    local start="$1"
    local end="$2"
    local idx
    for (( idx=start; idx<=end; idx++ )); do
        submit_batch "${idx}"
    done
}

default_batch_end() {
    local total n_batches
    total=$(total_rows)
    n_batches=$(num_batches "${total}")
    echo $(( n_batches - 1 ))
}

submit_chain() {
    local start="$1"
    local end="$2"
    local idx prev_id="" job_id dep=""
    echo "Chaining batches ${start}-${end} (each starts after the previous completes)."
    for (( idx=start; idx<=end; idx++ )); do
        dep=""
        if [[ -n "${prev_id}" ]]; then
            dep="afterok:${prev_id}"
        fi
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            submit_batch "${idx}" "${dep}"
            prev_id="<job_${idx}>"
        else
            job_id=$(submit_batch "${idx}" "${dep}")
            if [[ -n "${job_id}" ]]; then
                prev_id="${job_id}"
                echo "  -> job ${prev_id}" >&2
            fi
        fi
    done
    if [[ "${DRY_RUN}" -eq 0 && -n "${prev_id}" ]]; then
        echo "Chain submitted. Last job id: ${prev_id}" >&2
        echo "Monitor: squeue -u \"\$USER\" -n rq1_matrix" >&2
    fi
}

case "${MODE}" in
    list)
        list_batches
        ;;
    aggregate)
        run_aggregate_only
        ;;
    local)
        if [[ -n "${BATCH_END}" ]]; then
            for (( idx=BATCH_INDEX; idx<=BATCH_END; idx++ )); do
                run_local_batch "${idx}"
            done
        else
            run_local_batch "${BATCH_INDEX}"
        fi
        ;;
    chain)
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
        if [[ -z "${BATCH_END}" ]]; then
            BATCH_END=$(default_batch_end)
        fi
        submit_chain "${BATCH_INDEX}" "${BATCH_END}"
        ;;
    submit)
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
        if [[ -n "${BATCH_END}" ]]; then
            submit_range "${BATCH_INDEX}" "${BATCH_END}"
        else
            submit_batch "${BATCH_INDEX}"
        fi
        ;;
esac
