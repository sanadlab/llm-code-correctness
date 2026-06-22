#!/usr/bin/env python3
"""
Generate RQ1 experiment matrix CSV with all combinations.

Usage:
    python generate_rq1_matrix.py > matrix.csv
    python generate_rq1_matrix.py | python csv_to_sheets.py /dev/stdin

Outputs CSV to stdout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.rq1_matrix_spec import MATRIX_FIELDNAMES, iter_rq1_rows  # noqa: E402


def main() -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=MATRIX_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(iter_rq1_rows())


if __name__ == "__main__":
    main()
