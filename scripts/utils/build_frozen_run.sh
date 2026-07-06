#!/bin/bash
# Merge validated confounds (if sharded), freeze, report, and sample.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/phase1/2_isolated_run.yaml}"
SAMPLE_N="${SAMPLE_N:-270}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
REFERENCE_ROOT="${REFERENCE_ROOT:-dataset/isolated_bigcodebench}"

OUTPUT_ROOT="$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG}"))
print(cfg["output_root"])
PY
)"
INTERMEDIATE="${OUTPUT_ROOT}/intermediate"
FROZEN="${OUTPUT_ROOT}/frozen.jsonl"
METADATA="${OUTPUT_ROOT}/metadata.json"
REPORT_JSON="${OUTPUT_ROOT}/frozen_report.json"
SAMPLE_OUT="${OUTPUT_ROOT}/sample_${SAMPLE_N}_seed${SAMPLE_SEED}.jsonl"

echo "Output root: ${OUTPUT_ROOT}"

if compgen -G "${INTERMEDIATE}/validated_confounded.shard_*_of_*.jsonl" > /dev/null; then
    python scripts/merge_validated_confounds.py --intermediate-dir "${INTERMEDIATE}"
fi

if [[ ! -f "${INTERMEDIATE}/validated_confounded.jsonl" ]]; then
    echo "Error: ${INTERMEDIATE}/validated_confounded.jsonl not found."
    echo "Run validate_confounds first."
    exit 1
fi

python -m src.phase1.run --config "${CONFIG}" --step freeze --force

python scripts/report_frozen.py \
    --frozen "${FROZEN}" \
    --metadata "${METADATA}" \
    --output-json "${REPORT_JSON}"

python scripts/sample_frozen.py \
    --frozen "${FROZEN}" \
    --output "${SAMPLE_OUT}" \
    --n "${SAMPLE_N}" \
    --seed "${SAMPLE_SEED}" \
    --reference-root "${REFERENCE_ROOT}"

TRIPLET_OUT="${OUTPUT_ROOT}/sample_${SAMPLE_N}_seed${SAMPLE_SEED}_triplets.jsonl"
TRIPLET_FLAT="${OUTPUT_ROOT}/sample_${SAMPLE_N}_seed${SAMPLE_SEED}_triplets_flat.jsonl"
python scripts/sample_frozen_triplets.py \
    --frozen "${FROZEN}" \
    --output "${TRIPLET_OUT}" \
    --flat-output "${TRIPLET_FLAT}" \
    --n "${SAMPLE_N}" \
    --seed "${SAMPLE_SEED}" \
    --reference-root "${REFERENCE_ROOT}"

echo "Done."
echo "  frozen:  ${FROZEN}"
echo "  report:  ${REPORT_JSON}"
echo "  sample:  ${SAMPLE_OUT}"
echo "  triplets:      ${TRIPLET_OUT}"
echo "  triplets flat: ${TRIPLET_FLAT}"
