"""Phase 1 pipeline CLI.

Usage:
    python -m src.phase1.run --config configs/phase1/bigcodebench.yaml --step load
    python -m src.phase1.run --config configs/phase1/bigcodebench.yaml --step all
    python -m src.phase1.run --config configs/phase1/bigcodebench.yaml --step all --force

Steps: load, validate_ground_truth, mutate, validate_mutants, confound,
       validate_confounds, freeze, all
"""

import argparse
import os
import sys
from pathlib import Path

from src.phase1.config import load_config


STEPS_ORDER = [
    "load",
    "validate_ground_truth",
    "mutate",
    "validate_mutants",
    "confound",
    "validate_confounds",
    "freeze",
]

# Map step -> required input files (relative to config paths)
STEP_DEPS = {
    "load": [],
    "validate_ground_truth": ["tasks_path"],
    "mutate": ["validated_tasks_path"],
    "validate_mutants": ["mutants_path", "validated_tasks_path"],
    "confound": ["validated_mutants_path"],
    "validate_confounds": ["confound_candidates_path", "validated_tasks_path", "validated_mutants_path"],
    "freeze": ["validated_mutants_path", "validated_confounded_path", "validated_tasks_path", "tasks_path"],
}

# Map step -> output file (for resume check)
STEP_OUTPUTS = {
    "load": "tasks_path",
    "validate_ground_truth": "validated_tasks_path",
    "mutate": "mutants_path",
    "validate_mutants": "validated_mutants_path",
    "confound": "confound_candidates_path",
    "validate_confounds": "validated_confounded_path",
    "freeze": "frozen_path",
}


def _check_deps(cfg, step: str) -> None:
    """Error out if required inputs don't exist."""
    for dep in STEP_DEPS.get(step, []):
        path = getattr(cfg, dep)
        if not path.exists():
            print(f"ERROR: step '{step}' requires {path} which does not exist.")
            print(f"  Run the preceding step first.")
            sys.exit(1)


def _should_skip(cfg, step: str, force: bool) -> bool:
    """Check if step output already exists (resume semantics)."""
    if force:
        return False
    if step == "validate_mutants" and os.environ.get("VALIDATE_RESUME_MODE") in {"retry_timeouts", "retry_unresolved"}:
        return False
    output_attr = STEP_OUTPUTS.get(step)
    if output_attr:
        path = getattr(cfg, output_attr)
        if path.exists():
            print(f"  [{step}] output exists at {path}, skipping (use --force to re-run)")
            return True
    return False


def _run_step(cfg, step: str) -> None:
    """Execute a single step."""
    print(f"\n{'='*60}")
    print(f"  Step: {step}")
    print(f"{'='*60}")

    if step == "load":
        from src.phase1.load import load_tasks
        load_tasks(cfg)

    elif step == "validate_ground_truth":
        from src.phase1.validate_gt import validate_ground_truth
        validate_ground_truth(cfg)

    elif step == "mutate":
        from src.phase1.mutate import mutate
        mutate(cfg)

    elif step == "validate_mutants":
        from src.phase1.validate_mutants import validate_mutants
        validate_mutants(cfg)

    elif step == "confound":
        from src.phase1.confound import confound
        confound(cfg)

    elif step == "validate_confounds":
        from src.phase1.validate_confounds import validate_confounds
        validate_confounds(cfg)

    elif step == "freeze":
        from src.phase1.freeze import freeze
        freeze(cfg)

    else:
        print(f"ERROR: unknown step '{step}'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: BigCodeBench mutation dataset generation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--step", required=True,
                       choices=STEPS_ORDER + ["all"],
                       help="Step to run (or 'all')")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run even if output exists")
    parser.add_argument(
        "--reason",
        action="store_true",
        help="When running validate_mutants (or all), print details about failing mutants",
    )
    args = parser.parse_args()

    if args.reason:
        # Read by src.phase1.validate_mutants to emit extra debugging output.
        os.environ["VALIDATE_SHOW_REASON"] = "1"

    cfg = load_config(args.config)

    # Ensure output directories exist
    cfg.intermediate_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)

    if args.step == "all":
        for step in STEPS_ORDER:
            _check_deps(cfg, step)
            if not _should_skip(cfg, step, args.force):
                _run_step(cfg, step)
    else:
        _check_deps(cfg, args.step)
        if not _should_skip(cfg, args.step, args.force):
            _run_step(cfg, args.step)

    print("\nDone.")


if __name__ == "__main__":
    main()
