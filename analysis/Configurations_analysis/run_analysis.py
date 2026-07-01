"""Main entry point for RQ1 configuration analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR, TOP_N
from .load_data import aggregate_by_config, load_results
from .plots import generate_all_plots
from .rankings import rank_configurations, rank_rows, write_rankings
from .summaries import build_all_summaries, write_summaries


def _print_overview(row_df, config_df) -> None:
    print(f"Loaded {len(row_df)} cell-level rows across {config_df['ConfigID'].nunique()} configurations.")
    print(f"Selected accuracy: mean={row_df['avg'].mean():.4f}, std={row_df['avg'].std():.4f}")
    print(f"Best-layer accuracy: mean={row_df['avg (best, 1of4)'].mean():.4f}")
    print(f"Gap: mean={row_df['gap'].mean():.4f}, median={row_df['gap'].median():.4f}, max={row_df['gap'].max():.4f}")

    best = config_df.nlargest(3, "selected_avg")[["ConfigID", "config_label", "selected_avg", "gap"]]
    worst = config_df.nsmallest(3, "selected_avg")[["ConfigID", "config_label", "selected_avg", "gap"]]
    print("\nTop 3 configurations (by mean selected accuracy):")
    print(best.to_string(index=False))
    print("\nBottom 3 configurations:")
    print(worst.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RQ1 configuration results: rankings, gaps, patterns, and plots.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to rq1_results_with_best_layer.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV summaries, rankings, and plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of top/bottom configurations to include in rankings and bar charts",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (CSVs only)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    row_df = load_results(args.input_csv)
    config_df = aggregate_by_config(row_df)

    # Persist enriched tables.
    row_df.to_csv(args.output_dir / "enriched_rows.csv", index=False)
    config_df.to_csv(args.output_dir / "aggregated_by_config.csv", index=False)

    config_rankings = rank_configurations(config_df, top_n=args.top_n)
    row_rankings = rank_rows(row_df, top_n=args.top_n)
    write_rankings({**config_rankings, **row_rankings}, args.output_dir)

    summaries = build_all_summaries(row_df)
    write_summaries(summaries, args.output_dir)

    if not args.no_plots:
        generate_all_plots(
            row_df,
            config_df,
            args.output_dir,
            top_n=args.top_n,
            summaries=summaries,
            row_rankings=row_rankings,
        )

    _print_overview(row_df, config_df)
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
