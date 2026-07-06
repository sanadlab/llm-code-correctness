# Dataset Utilities

This directory contains utility scripts for generating, analyzing, validating, and packaging the synthetic correctness datasets (mutants and confounds).

## Dataset Splitting & Packaging
- `build_final_deliverables.py`: The definitive script that generates the final `deliverables` directory and matrices.
- `build_phase2_splits.py`: Formats the raw phase 1 triplets/isolated runs into the phase 2 partitions for evaluation.
- `update_splits.py` & `build_new_splits.py`: Helper scripts for regenerating specific data folds.

## Data Sampling
- `sample_frozen_triplets.py`: Samples strict triplets (canonical, mutant, confound) from the dataset.
- `sample_frozen_triplets_allow_unconfounded.py`: Samples triplets but relaxes the strict requirement for confounds.
- `sample_frozen.py`: General sampling utility for frozen datasets.

## Analytics & Validation
- `analyze_rejected_confounds.py`: Deep dive analytics into why LLM-generated confounds failed validation (e.g. failure fingerprint mismatches).
- `sanity_check.py`: Validates the distribution of task kinds (canonical, mutant, confound).
- `error_frequencies.py`: Computes the frequencies of different mutation operators and failure fingerprints.
- `merge_validated_confounds.py`: Merges validated confounds into a unified pool.
- `report_frozen.py`: Generates summary statistics over a frozen dataset JSONL.

## Testing & Automation
- `test_type_aware_operators.py`: Verification script to test AST type-aware mutations.
- `build_frozen_run.sh`: Shell automation script to freeze a dataset build.
- `download_models.py`: Helper script to download local HF models if needed.
