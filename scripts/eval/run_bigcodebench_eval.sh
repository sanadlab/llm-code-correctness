#!/usr/bin/env bash
# scripts/eval/run_bigcodebench_eval.sh
#
# Thin wrapper that launches BigCodeBench evaluation inside a container.
# Supports Docker (default) and Singularity (--runtime singularity).
# All benchmark code runs inside the container — never on the host.
#
# Usage:
#   scripts/eval/run_bigcodebench_eval.sh \
#     --samples /abs/path/to/samples.jsonl \
#     --split complete \
#     --subset full \
#     [--image bigcodebench/bigcodebench-evaluate:latest] \
#     [--sif /path/to/bigcodebench-evaluate.sif] \
#     [--runtime docker|singularity] \
#     [--parallel 8] \
#     [--min-time-limit 1] \
#     [--no-gt] \
#     [--selective-evaluate 'BigCodeBench/0,BigCodeBench/1,...']
#
# The script mounts the directory containing --samples into /app inside the
# container, runs bigcodebench.evaluate with --execution local, and writes
# the *_eval_results.json next to the input samples file.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────
IMAGE="bigcodebench/bigcodebench-evaluate:latest"
SIF=""
RUNTIME="docker"
SPLIT="complete"
SUBSET="full"
PARALLEL=8
MIN_TIME_LIMIT=10
SAMPLES=""
NO_GT=""
SELECTIVE=""
CONTAINER_NAME=""
SANITIZE=false
CALIBRATED=true
NO_PATCH=false

# ── parse args ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples)      SAMPLES="$2";       shift 2 ;;
        --split)        SPLIT="$2";         shift 2 ;;
        --subset)       SUBSET="$2";        shift 2 ;;
        --image)        IMAGE="$2";         shift 2 ;;
        --sif)          SIF="$2";           shift 2 ;;
        --runtime)      RUNTIME="$2";       shift 2 ;;
        --parallel)     PARALLEL="$2";      shift 2 ;;
        --min-time-limit) MIN_TIME_LIMIT="$2"; shift 2 ;;
        --no-gt)        NO_GT="--no-gt";    shift ;;
        --selective-evaluate) SELECTIVE="$2"; shift 2 ;;
        --container-name) CONTAINER_NAME="$2"; shift 2 ;;
        --sanitize)     SANITIZE=true;      shift ;;
        --no-calibrate) CALIBRATED=false;   shift ;;
        --no-patch)     NO_PATCH=true;      shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SAMPLES" ]]; then
    echo "ERROR: --samples is required" >&2
    exit 1
fi

# Resolve to absolute path
SAMPLES="$(realpath "$SAMPLES")"
SAMPLES_DIR="$(dirname "$SAMPLES")"
SAMPLES_NAME="$(basename "$SAMPLES")"

if [[ ! -f "$SAMPLES" ]]; then
    echo "ERROR: samples file not found: $SAMPLES" >&2
    exit 1
fi

# ── common eval arguments ────────────────────────────────
EVAL_ARGS=(
    --execution local
    --split "$SPLIT"
    --subset "$SUBSET"
    --samples "$SAMPLES_NAME"
    --parallel "$PARALLEL"
    --min-time-limit "$MIN_TIME_LIMIT"
)
# fire parses --calibrated / --nocalibrated as True / False booleans
if [[ "$CALIBRATED" == "true" ]]; then
    EVAL_ARGS+=(--calibrated)
else
    EVAL_ARGS+=(--nocalibrated)
fi
[[ -n "$NO_GT" ]]    && EVAL_ARGS+=("$NO_GT")
[[ -n "$SELECTIVE" ]] && EVAL_ARGS+=(--selective_evaluate "$SELECTIVE")

# ── launch container ──────────────────────────────────────
echo "Running BigCodeBench evaluation (runtime=$RUNTIME)..."
echo "  Samples:   $SAMPLES"
echo "  Split:     $SPLIT"
echo "  Subset:    $SUBSET"
echo "  Parallel:  $PARALLEL"
if [[ -n "$SELECTIVE" ]]; then
    echo "  Selective: ${SELECTIVE:0:80}..."
fi

if [[ "$RUNTIME" == "singularity" ]]; then
    # ── Singularity / Apptainer path ──────────────────────
    if [[ -z "$SIF" ]]; then
        echo "ERROR: --sif is required when --runtime singularity" >&2
        exit 1
    fi
    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: SIF image not found: $SIF" >&2
        exit 1
    fi
    echo "  SIF:       $SIF"

    # Bind the samples directory to /app (mimicking Docker -v).
    # --writable-tmpfs allows the container to write to its own rootfs
    # without actually modifying the SIF.  --pwd sets the working directory.
    #
    # BIGCODEBENCH_OVERRIDE_PATH: skip HuggingFace download by pointing
    # directly at a pre-cached dataset JSONL.  Avoids rate-limit errors
    # when many containers launch in parallel.
    SINGULARITY_ENV_ARGS=()
    if [[ -n "${BIGCODEBENCH_OVERRIDE_PATH:-}" ]]; then
        SINGULARITY_ENV_ARGS+=(--env "BIGCODEBENCH_OVERRIDE_PATH=${BIGCODEBENCH_OVERRIDE_PATH}")
    fi
    # Inject our patched bigcodebench via PYTHONPATH so the container picks
    # up timeout fixes (GT-calibrated per-task timeouts).
    REPO_ROOT_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
    PATCHED_BCB="${REPO_ROOT_GUESS}/third_party/bigcodebench"
    BIND_ARGS="${SAMPLES_DIR}:/app"
    if [[ "$NO_PATCH" == false ]] && [[ -d "${PATCHED_BCB}/bigcodebench" ]]; then
        BIND_ARGS+=,${PATCHED_BCB}:/patched_bcb
        SINGULARITY_ENV_ARGS+=(--env "PYTHONPATH=/patched_bcb")
    fi
    # Bind HOME so the GT timing cache (~/.cache/bigcodebench/*.pkl) is visible.
    # When --no-patch is used, bind only the cache directory (not all of HOME)
    # to avoid leaking host-side patched code into the container.
    if [[ "$NO_PATCH" == true ]]; then
        GT_CACHE_DIR="${HOME}/.cache/bigcodebench"
        if [[ -d "${GT_CACHE_DIR}" ]]; then
            BIND_ARGS+=,${GT_CACHE_DIR}:${GT_CACHE_DIR}
        fi
    else
        BIND_ARGS+=,${HOME}:${HOME}
    fi
    # Ensure the override dataset is reachable inside the container.
    if [[ -n "${BIGCODEBENCH_OVERRIDE_PATH:-}" ]] && [[ -f "${BIGCODEBENCH_OVERRIDE_PATH}" ]]; then
        OVERRIDE_DIR="$(dirname "${BIGCODEBENCH_OVERRIDE_PATH}")"
        BIND_ARGS+=,${OVERRIDE_DIR}:${OVERRIDE_DIR}
    fi
    SINGULARITY_ENV_ARGS+=(--env "BIGCODEBENCH_GT_TIMEOUT_MULTIPLIER=${BIGCODEBENCH_GT_TIMEOUT_MULTIPLIER:-2}")

    # ── optional sanitize step (tree-sitter extraction) ─────
    if [[ "$SANITIZE" == true ]]; then
        echo "  Running bigcodebench.sanitize --inplace ..."
        singularity exec \
            --bind "${BIND_ARGS}" \
            --pwd /app \
            --writable-tmpfs \
            "${SINGULARITY_ENV_ARGS[@]}" \
            "$SIF" \
            python -m bigcodebench.sanitize --samples "$SAMPLES_NAME" --inplace
        echo "  Sanitization complete."
    fi

    # Redirect evaluator stdout/stderr to a log file so that even if the
    # Singularity-spawned Python process outlives the wrapper shell, it won't
    # hold our parent's capture-pipes open (preventing subprocess.run from
    # blocking indefinitely).
    EVAL_LOG="${SAMPLES_DIR}/.eval_log_$$.txt"
    singularity exec \
        --bind "${BIND_ARGS}" \
        --pwd /app \
        --writable-tmpfs \
        "${SINGULARITY_ENV_ARGS[@]}" \
        "$SIF" \
        python -m bigcodebench.evaluate "${EVAL_ARGS[@]}" \
        > "$EVAL_LOG" 2>&1
    EVAL_EXIT=$?
    cat "$EVAL_LOG"
    rm -f "$EVAL_LOG"
    if [[ $EVAL_EXIT -ne 0 ]]; then
        exit $EVAL_EXIT
    fi
else
    # ── Docker path (original) ────────────────────────────
    echo "  Image:     $IMAGE"

    DOCKER_RUN_ARGS=(--rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf_cache)
    [[ -n "$CONTAINER_NAME" ]] && DOCKER_RUN_ARGS+=(--name "$CONTAINER_NAME")

    # ── optional sanitize step (tree-sitter extraction) ─────
    if [[ "$SANITIZE" == true ]]; then
        echo "  Running bigcodebench.sanitize --inplace ..."
        docker run "${DOCKER_RUN_ARGS[@]}" \
            -v "${SAMPLES_DIR}:/app" \
            --entrypoint python \
            "$IMAGE" \
            -m bigcodebench.sanitize --samples "$SAMPLES_NAME" --inplace
        echo "  Sanitization complete."
    fi

    docker run "${DOCKER_RUN_ARGS[@]}" \
        -v "${SAMPLES_DIR}:/app" \
        "$IMAGE" \
        "${EVAL_ARGS[@]}"
fi

# The eval container writes results next to the samples file:
# <samples_name>_eval_results.json
RESULT_FILE="${SAMPLES%.jsonl}_eval_results.json"

if [[ -f "$RESULT_FILE" ]]; then
    echo "Evaluation results written to: $RESULT_FILE"
else
    echo "WARNING: expected result file not found: $RESULT_FILE" >&2
    # List what was actually written
    ls -la "${SAMPLES_DIR}"/*.json 2>/dev/null || true
fi
