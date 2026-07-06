#!/bin/bash
#SBATCH --job-name=build_frozen_he
#SBATCH --output=logs/slurm/build_frozen_he.%j.out
#SBATCH --error=logs/slurm/build_frozen_he.%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_ROOT}"

module purge
module load python/3.11.3
source "${SCRATCH}/envs/llm-code-correctness/bin/activate"

CONFIG="${CONFIG:-configs/phase1/humaneval.yaml}"
OUTPUT_ROOT="$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['output_root'])")"
INTERMEDIATE="${OUTPUT_ROOT}/intermediate"

if compgen -G "${INTERMEDIATE}/validated_confounded.shard_*_of_*.jsonl" > /dev/null; then
    python scripts/merge_validated_confounds.py --intermediate-dir "${INTERMEDIATE}"
fi

python -m src.phase1.run --config "${CONFIG}" --step freeze --force

python scripts/report_frozen.py \
    --frozen "${OUTPUT_ROOT}/frozen.jsonl" \
    --metadata "${OUTPUT_ROOT}/metadata.json" \
    --output-json "${OUTPUT_ROOT}/frozen_report.json"

echo "Done freezing HumanEval."
