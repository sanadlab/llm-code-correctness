#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../jobs/phase1_confound.sbatch"

CONFIG="configs/phase1/bigcodebench.yaml"
STEP="confound"
JOB_NAME="phase1_confound"
PARTITION="nvidia"
ACCOUNT=""
GPUS=1
CPUS=8
MEM="16G"
TIME="04:00:00"
PYTHON_MODULE="python/3.11.3"
CUDA_MODULE="cuda/12.2.0"
FORCE=0
DRY_RUN=0

if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR_DEFAULT="${SCRATCH}/envs/project-abu-dhabi"
    HF_HOME_DEFAULT="${SCRATCH}/hf-models"
else
    ENV_DIR_DEFAULT=""
    HF_HOME_DEFAULT=""
fi

ENV_DIR="${ENV_DIR_DEFAULT}"
HF_HOME="${HF_HOME_DEFAULT}"

usage() {
    cat <<'EOF'
Usage: scripts/slurm/launchers/phase1_confound.sh [options]

Submit the GPU-backed Phase 1 local confound step to SLURM.

Options:
  --config PATH           YAML config to run (default: configs/phase1/bigcodebench.yaml)
  --step NAME             Phase 1 step to run on GPU (default: confound)
  --env-dir PATH          Virtualenv to activate on the compute node
  --hf-home PATH          Hugging Face cache/model directory
  --job-name NAME         SLURM job name (default: phase1_confound)
    --partition NAME        SLURM partition (default: nvidia)
    --account NAME          SLURM account (default: none)
    --gpus N                Number of GPUs to request (default: 1)
    --cpus N                Number of CPUs to request (default: 4)
    --mem SIZE              Memory request (default: 16G)
    --time HH:MM:SS         Wall-clock limit (default: 00:10:00)
  --python-module NAME    Python module to load (default: python/3.11.3)
  --cuda-module NAME      CUDA module to load (default: cuda/12.2.0)
  --force                 Pass --force to the phase1 runner
  --dry-run               Print the sbatch command without submitting
  --help                  Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --env-dir)
            ENV_DIR="$2"
            shift 2
            ;;
        --hf-home)
            HF_HOME="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --python-module)
            PYTHON_MODULE="$2"
            shift 2
            ;;
        --cuda-module)
            CUDA_MODULE="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${SCRATCH:-}" ]]; then
    echo "Error: SCRATCH is not set."
    exit 1
fi

if [[ -z "${ENV_DIR}" ]]; then
    echo "Error: --env-dir was not provided and SCRATCH is unavailable for the default path."
    exit 1
fi

if [[ -z "${HF_HOME}" ]]; then
    echo "Error: --hf-home was not provided and SCRATCH is unavailable for the default path."
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

SBATCH_EXPORT="ALL,CONFIG=${CONFIG},STEP=${STEP},ENV_DIR=${ENV_DIR},HF_HOME=${HF_HOME},PYTHON_MODULE=${PYTHON_MODULE},CUDA_MODULE=${CUDA_MODULE},FORCE=${FORCE},REPO_ROOT=${REPO_ROOT}"

cmd=(
    sbatch
    --job-name "${JOB_NAME}"
    --output "logs/slurm/${JOB_NAME}.%j.out"
    --error "logs/slurm/${JOB_NAME}.%j.err"
    --partition "${PARTITION}"
    --gres "gpu:${GPUS}"
    --cpus-per-task "${CPUS}"
    --mem "${MEM}"
    --time "${TIME}"
    --export "${SBATCH_EXPORT}"
    "${JOB_SCRIPT}"
)

if [[ -n "${ACCOUNT}" ]]; then
    cmd=("${cmd[@]:0:5}" --account "${ACCOUNT}" "${cmd[@]:5}")
fi

printf 'Submitting SLURM job with command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

"${cmd[@]}"