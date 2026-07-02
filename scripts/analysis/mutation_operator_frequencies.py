"""Report mutation-operator frequencies per fold and split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset.operator_weights import (  # noqa: E402
    SPLITS,
    SPLIT_TO_JSONL,
    collect_fold_split_frequencies,
    count_operators_in_file,
    format_operator_report,
    operator_code_to_name,
)


def _report_dataset(
    label: str,
    data_root: Path,
    folds: list[int],
    splits: list[str],
) -> dict[int, dict[str, dict[str, int]]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing data root: {data_root}")

    names = operator_code_to_name()
    counts_by_fold_split: dict[int, dict[str, dict[str, int]]] = {}

    print(f"\n{'#' * 60}")
    print(f"# {label} ({data_root})")
    print(f"{'#' * 60}")

    for fold in folds:
        print(f"\n--- fold_{fold} ---")
        counts_by_fold_split[fold] = {}
        for split in splits:
            path = data_root / f"fold_{fold}" / SPLIT_TO_JSONL[split]
            if not path.is_file():
                raise FileNotFoundError(f"missing split file: {path}")
            counts = count_operators_in_file(path)
            counts_by_fold_split[fold][split] = dict(counts)
            print()
            print(format_operator_report(split, counts, names=names))

    return counts_by_fold_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report mutation-operator frequencies per fold/split"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/mutants_only_full/S1-M"),
        help="Dataset subfolder with fold_* directories",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=list(range(10)),
        help="Fold indices (default: 0..9)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to report (default: fit validate test)",
    )
    parser.add_argument(
        "--also-s2",
        action="store_true",
        help="Also report mutants_only_full/S2-M-BCB",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write counts + frequencies as JSON",
    )
    args = parser.parse_args()

    counts_s1 = _report_dataset("S1-M", args.data_root, args.folds, args.splits)

    counts_s2 = None
    if args.also_s2:
        s2_root = args.data_root.parent / "S2-M-BCB"
        counts_s2 = _report_dataset("S2-M-BCB", s2_root, args.folds, args.splits)

    if args.json_out:
        freqs_s1 = collect_fold_split_frequencies(args.data_root, args.folds, args.splits)
        payload: dict = {
            "data_root": str(args.data_root),
            "folds": args.folds,
            "splits": args.splits,
            "counts": counts_s1,
            "frequencies": freqs_s1,
        }
        if counts_s2 is not None:
            s2_root = args.data_root.parent / "S2-M-BCB"
            payload["s2_m_bcb"] = {
                "data_root": str(s2_root),
                "counts": counts_s2,
                "frequencies": collect_fold_split_frequencies(s2_root, args.folds, args.splits),
            }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
