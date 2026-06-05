#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/phase1_mutate.sbatch"

CONFIG="configs/phase1/bigcodebench.yaml"
STEP="mutate"
JOB_NAME="phase1_mutate"
PARTITION="cs"
ACCOUNT="condo_cs_nadi"
QOS=""
DEPENDENCY=""
CPUS=32
PARALLEL=""
MEM="32G"
TIME="01:00:00"
PYTHON_MODULE="python/3.11.3"
FORCE=0
DRY_RUN=0

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR_DEFAULT="${SCRATCH}/envs/llm-code-correctness"
else
    ENV_DIR_DEFAULT=""
fi

ENV_DIR="${ENV_DIR_DEFAULT}"

usage() {
    cat <<'EOF'
Usage: scripts/slurm/launchers/phase1_mutate.sh [options]

Submit the Phase 1 mutate step to SLURM.

Options:
  --config PATH           YAML config (default: configs/phase1/bigcodebench.yaml)
  --env-dir PATH          Virtualenv path on the compute node
  --job-name NAME         SLURM job name (default: phase1_mutate)
  --partition NAME        SLURM partition (default: auto-select)
  --account NAME          SLURM account (default: none)
  --qos NAME              SLURM QOS (default: none)
  --dependency SPEC       SLURM dependency spec, e.g. afterok:12345
  --cpus N                CPU count (default: 32)
  --parallel N            Mutate worker count override (default: use cpus-per-task)
  --mem SIZE              Memory (default: 32G)
  --time HH:MM:SS         Wall-clock limit (default: 01:00:00)
  --python-module NAME    Module to load (default: python/3.11.3)
  --force                 Re-run even if output exists
  --dry-run               Print sbatch command without submitting
  --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)         CONFIG="$2";         shift 2 ;;
        --env-dir)        ENV_DIR="$2";        shift 2 ;;
        --job-name)       JOB_NAME="$2";       shift 2 ;;
        --partition)      PARTITION="$2";      shift 2 ;;
        --account)        ACCOUNT="$2";        shift 2 ;;
        --qos)            QOS="$2";            shift 2 ;;
        --dependency)     DEPENDENCY="$2";     shift 2 ;;
        --cpus)           CPUS="$2";           shift 2 ;;
        --parallel)       PARALLEL="$2";       shift 2 ;;
        --mem)            MEM="$2";            shift 2 ;;
        --time)           TIME="$2";           shift 2 ;;
        --python-module)  PYTHON_MODULE="$2";  shift 2 ;;
        --force)          FORCE=1;              shift ;;
        --dry-run)        DRY_RUN=1;            shift ;;
        --help|-h)        usage; exit 0 ;;
        *)                echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "${ENV_DIR}" ]]; then
    echo "Error: --env-dir was not provided and SCRATCH is unavailable for the default path."
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

SBATCH_EXPORT="ALL,CONFIG=${CONFIG},STEP=${STEP},ENV_DIR=${ENV_DIR},PYTHON_MODULE=${PYTHON_MODULE},FORCE=${FORCE},REPO_ROOT=${REPO_ROOT}"

if [[ -n "${PARALLEL}" ]]; then
    SBATCH_EXPORT+=" ,MUTATE_PARALLEL=${PARALLEL}"
    SBATCH_EXPORT="${SBATCH_EXPORT/ ,/,}"
fi

cmd=(
    sbatch
    --job-name "${JOB_NAME}"
    --output "logs/slurm/${JOB_NAME}.%j.out"
    --error "logs/slurm/${JOB_NAME}.%j.err"
    --cpus-per-task "${CPUS}"
    --mem "${MEM}"
    --time "${TIME}"
    --export "${SBATCH_EXPORT}"
)

if [[ -n "${PARTITION}" ]]; then
    cmd+=(--partition "${PARTITION}")
fi

if [[ -n "${ACCOUNT}" ]]; then
    cmd+=(--account "${ACCOUNT}")
fi

if [[ -n "${QOS}" ]]; then
    cmd+=(--qos "${QOS}")
fi

if [[ -n "${DEPENDENCY}" ]]; then
    cmd+=(--dependency "${DEPENDENCY}")
fi

cmd+=("${JOB_SCRIPT}")

printf 'Submitting SLURM job with command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

"${cmd[@]}"
