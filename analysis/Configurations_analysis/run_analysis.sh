#!/usr/bin/env bash
# Run RQ1 configuration analysis from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${SCRATCH:-/scratch/re2230}/envs/llm-code-correctness/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

"$PYTHON" -m analysis.Configurations_analysis.run_analysis "$@"
