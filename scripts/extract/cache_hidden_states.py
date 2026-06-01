"""Extract and cache per-layer pooled hidden states to disk.

This script reuses:
  - `extract_bundle()` for forward + pooling
  - `REPRESENTATIONS` registry for pooling strategy instances
  - `HiddenStateBundle.save()/load()` for persistence
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from scripts.experiments._common import (
    REPRESENTATIONS,
    extract_bundle,
    load_model_and_tokenizer,
    load_split,
)
from src.hidden_states import (
    HiddenStateBundle,
    SPLIT_TO_JSONL,
    VerifyConfig,
    build_bundle_metadata,
    bundle_is_complete,
    bundle_output_dir,
    framing_dispatcher,
    git_commit,
    verify_standard_fold,
    write_bundle_metadata,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset source and its fold structure."""

    split_family: str
    benchmark: str
    splits_dir: Path
    distribution: str = "id"
    fold_prefix: str = "fold"
    ood_source: Optional[str] = None
    test_dataset: Optional[str] = None
    outer_fold: Optional[int] = None


@dataclass(frozen=True)
class ExtractionJob:
    model_name: str
    dataset: DatasetSpec
    fold: int
    data_split: str
    framing: str
    jsonl_path: Path
    output_dir: Path


@dataclass
class ExtractionConfig:
    models: List[str]
    datasets: List[DatasetSpec]
    data_splits: List[str]
    framings: Dict[str, str]
    representations: List[str]
    folds: Optional[List[int]]
    output_dir: Path
    batch_size: int
    fit_max_length: int
    eval_max_length: int
    dtype: str
    force: bool
    verify: Optional[VerifyConfig]
    verify_split_families: List[str]
    verify_only: bool = False


def _parse_framings(pairs: Sequence[str], data_splits: Sequence[str]) -> Dict[str, str]:
    framings: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid --framings entry {pair!r}, expected split=framing")
        split, framing = pair.split("=", 1)
        split = split.strip()
        framing = framing.strip()
        if split not in SPLIT_TO_JSONL:
            raise ValueError(f"invalid split {split!r} in --framings")
        framings[split] = framing
    missing = [s for s in data_splits if s not in framings]
    if missing:
        raise ValueError(f"missing framing for splits: {missing}")
    return framings


def _protocol_defaults(protocol: str) -> Dict[str, str]:
    if protocol == "function":
        return {"fit": "function", "validate": "neutral", "test": "neutral"}
    if protocol == "pca":
        return {"fit": "raw", "validate": "concept", "test": "concept"}
    raise ValueError(f"unknown protocol: {protocol}")


def resolve_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    """Resolve one or more dataset specs from CLI flags."""
    out: List[DatasetSpec] = []
    data_root = Path(args.data_root)

    if args.dataset_source == "icse26":
        splits_dir = data_root / args.benchmark
        out.append(
            DatasetSpec(
                split_family=args.split_family,
                benchmark=args.benchmark,
                splits_dir=splits_dir,
            )
        )
    elif args.dataset_source == "mutation":
        splits_dir = data_root / args.split_family
        out.append(
            DatasetSpec(
                split_family=args.split_family,
                benchmark="bigcodebench",
                splits_dir=splits_dir,
            )
        )
    else:  # ood
        # Inner OOD fit/validate source
        out.append(
            DatasetSpec(
                split_family=f"ood_{args.ood_source}",
                benchmark=args.test_dataset,
                splits_dir=data_root / args.ood_source,
                fold_prefix="inner",
                ood_source=args.ood_source,
                test_dataset=args.test_dataset,
            )
        )
        # Outer ID test source
        out.append(
            DatasetSpec(
                split_family=f"ood_{args.ood_source}",
                benchmark=args.test_dataset,
                splits_dir=data_root / args.test_dataset,
                fold_prefix="fold",
                ood_source=args.ood_source,
                test_dataset=args.test_dataset,
            )
        )
    return out


def _dataset_fold_indices(spec: DatasetSpec, requested: Optional[List[int]]) -> List[int]:
    manifest = json.loads((spec.splits_dir / "manifest.json").read_text())
    if requested is None:
        return list(range(manifest["num_folds"]))
    return requested


def iter_extraction_jobs(config: ExtractionConfig) -> Iterable[ExtractionJob]:
    """Expand model × dataset × fold × split into concrete extraction jobs."""
    for model_name in config.models:
        for dataset in config.datasets:
            folds = _dataset_fold_indices(dataset, config.folds)
            for fold in folds:
                for data_split in config.data_splits:
                    if data_split not in SPLIT_TO_JSONL:
                        continue
                    if not _should_extract_split(dataset, data_split):
                        continue
                    framing = config.framings[data_split]
                    split_dir = dataset.splits_dir / f"fold_{fold}"
                    jsonl_path = split_dir / SPLIT_TO_JSONL[data_split]
                    out_dir = bundle_output_dir(
                        output_root=config.output_dir,
                        model_name=model_name,
                        split_family=dataset.split_family,
                        benchmark=dataset.benchmark,
                        distribution=dataset.distribution,
                        data_split=data_split,
                        framing=framing,
                        fold=fold,
                        fold_prefix=dataset.fold_prefix,
                    )
                    yield ExtractionJob(
                        model_name=model_name,
                        dataset=dataset,
                        fold=fold,
                        data_split=data_split,
                        framing=framing,
                        jsonl_path=jsonl_path,
                        output_dir=out_dir,
                    )


def _should_extract_split(dataset: DatasetSpec, data_split: str) -> bool:
    # OOD inner source only has fit/validate; OOD outer source used for test.
    if dataset.ood_source is None:
        return True
    if dataset.fold_prefix == "inner":
        return data_split in {"fit", "validate"}
    return data_split == "test"


def max_length_for_split(data_split: str, config: ExtractionConfig) -> int:
    return config.fit_max_length if data_split == "fit" else config.eval_max_length


def extract_one_job(
    *,
    job: ExtractionJob,
    model,
    tokenizer,
    config: ExtractionConfig,
    git_sha: str,
) -> None:
    """Run extraction + save one cache cell."""
    if bundle_is_complete(job.output_dir, config.force):
        logger.info("skip existing: %s", job.output_dir)
        return
    if not job.jsonl_path.is_file():
        raise FileNotFoundError(f"missing split file: {job.jsonl_path}")

    logger.info(
        "extract model=%s split_family=%s benchmark=%s split=%s framing=%s fold=%d",
        job.model_name,
        job.dataset.split_family,
        job.dataset.benchmark,
        job.data_split,
        job.framing,
        job.fold,
    )
    records = load_split(job.jsonl_path)
    representations = [REPRESENTATIONS[name] for name in config.representations]
    bundle = extract_bundle(
        framing_dispatcher(job.framing, job.model_name),
        records,
        model,
        tokenizer,
        representations,
        batch_size=config.batch_size,
        max_length=max_length_for_split(job.data_split, config),
    )
    bundle = HiddenStateBundle(
        sample_ids=bundle.sample_ids,
        task_ids=bundle.task_ids,
        labels=bundle.labels,
        acts={name: tensor.float() for name, tensor in bundle.acts.items()},
    )
    bundle_path = job.output_dir / "bundle.pt"
    bundle.save(bundle_path)
    meta = build_bundle_metadata(
        model=job.model_name,
        dataset=job.dataset.benchmark,
        split_family=job.dataset.split_family,
        distribution=job.dataset.distribution,
        fold=job.fold,
        data_split=job.data_split,
        framing=job.framing,
        representations=config.representations,
        dtype=config.dtype,
        max_length=max_length_for_split(job.data_split, config),
        n_samples=bundle.num_samples,
        source_jsonl=job.jsonl_path,
        bundle_path=bundle_path,
        sample_ids=bundle.sample_ids,
        experiment_script="scripts/extract/cache_hidden_states.py",
        git_sha=git_sha,
        ood_source=job.dataset.ood_source,
        test_dataset=job.dataset.test_dataset,
        outer_fold=job.dataset.outer_fold,
        fold_prefix=job.dataset.fold_prefix,
    )
    write_bundle_metadata(job.output_dir / "metadata.json", meta)


def run_for_model(model_name: str, config: ExtractionConfig, jobs: List[ExtractionJob]) -> None:
    logger.info("loading model: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name, config.dtype)
    git_sha = git_commit()

    for job in jobs:
        extract_one_job(job=job, model=model, tokenizer=tokenizer, config=config, git_sha=git_sha)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_verification(config: ExtractionConfig) -> None:
    """Validate cached scores against existing detailed.json artifacts."""
    if config.verify is None:
        return

    rep = REPRESENTATIONS[config.verify.representation]
    for split_family in config.verify_split_families:
        if split_family.startswith("ood_"):
            logger.warning("verification skipped for %s (no detailed.json in ood flow)", split_family)
            continue
        for dataset in config.datasets:
            if dataset.split_family != split_family:
                continue
            folds = _dataset_fold_indices(dataset, config.folds)
            for fold in folds:
                verify_standard_fold(
                    model_name=config.models[0],
                    split_family=dataset.split_family,
                    dataset=dataset.benchmark,
                    distribution=dataset.distribution,
                    fold=fold,
                    hidden_states_root=config.output_dir,
                    verify=config.verify,
                    representation=rep,
                )


def parse_args() -> ExtractionConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset-source", choices=["icse26", "mutation", "ood"], required=True)
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], default="bigcodebench")
    parser.add_argument("--split-family", default="icse26")
    parser.add_argument("--ood-source", choices=["synthetic", "mbpp"], default=None)
    parser.add_argument("--test-dataset", choices=["bigcodebench", "humaneval"], default=None)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root (default: dataset/icse26 or dataset/bigcodebench/phase2).",
    )
    parser.add_argument("--data-splits", nargs="+", default=["fit", "validate", "test"])
    parser.add_argument("--protocol", choices=["pca", "function"], default=None)
    parser.add_argument("--framing", default=None)
    parser.add_argument("--framings", nargs="*", default=[])
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["prompt_avg", "prompt_last", "response_avg", "sequence_last"],
        choices=list(REPRESENTATIONS),
    )
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default="hidden_states")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fit-max-length", type=int, default=2048)
    parser.add_argument("--eval-max-length", type=int, default=4096)
    parser.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip extraction; only run verification against existing caches.",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--vectors-dir", default="vectors")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--verify-method", choices=["mean_diff", "pca"], default=None)
    parser.add_argument("--verify-fit-framing", default=None)
    parser.add_argument("--verify-representation", choices=list(REPRESENTATIONS), default=None)
    parser.add_argument("--verify-val-framing", default=None)
    parser.add_argument("--verify-test-framing", default=None)
    # Verification compares two independent (non-bitwise-reproducible) float16
    # forward passes, so it checks scale-free, decision-level invariants rather
    # than per-cell tolerances. Defaults kept in sync with VerifyConfig.
    parser.add_argument("--verify-corr-min", type=float, default=0.999)
    parser.add_argument("--verify-acc-atol", type=float, default=0.02)
    args = parser.parse_args()

    if args.dataset_source == "ood":
        if args.ood_source is None or args.test_dataset is None:
            raise ValueError("--ood-source and --test-dataset are required for --dataset-source ood")

    if args.verify_only and not args.verify:
        raise ValueError("--verify-only requires --verify")

    if args.data_root is None:
        if args.dataset_source == "mutation":
            args.data_root = "dataset/bigcodebench/phase2"
        else:
            args.data_root = "dataset/icse26"

    data_splits = list(dict.fromkeys(args.data_splits))
    if args.framing is not None:
        framings = {split: args.framing for split in data_splits}
    elif args.framings:
        framings = _parse_framings(args.framings, data_splits)
    elif args.protocol is not None:
        defaults = _protocol_defaults(args.protocol)
        framings = {split: defaults[split] for split in data_splits}
    else:
        raise ValueError("Provide one of --framing, --framings, or --protocol")

    verify: Optional[VerifyConfig] = None
    if args.verify:
        if args.protocol == "function":
            method = "mean_diff"
            fit_framing = "function"
            rep = "response_avg"
            val_framing = "neutral"
            test_framing = "neutral"
        elif args.protocol == "pca":
            method = "pca"
            fit_framing = "raw"
            rep = "sequence_last"
            val_framing = "concept"
            test_framing = "concept"
        else:
            method = args.verify_method
            fit_framing = args.verify_fit_framing
            rep = args.verify_representation
            val_framing = args.verify_val_framing
            test_framing = args.verify_test_framing
        if not all([method, fit_framing, rep, val_framing, test_framing]):
            raise ValueError(
                "--verify requires protocol defaults or explicit "
                "--verify-method/--verify-fit-framing/--verify-representation/"
                "--verify-val-framing/--verify-test-framing"
            )
        verify = VerifyConfig(
            vectors_dir=Path(args.vectors_dir),
            results_dir=Path(args.results_dir),
            method=method,
            fit_framing=fit_framing,
            representation=rep,
            val_framing=val_framing,
            test_framing=test_framing,
            corr_min=args.verify_corr_min,
            acc_atol=args.verify_acc_atol,
        )

    datasets = resolve_dataset_specs(args)
    return ExtractionConfig(
        models=args.models,
        datasets=datasets,
        data_splits=data_splits,
        framings=framings,
        representations=args.representations,
        folds=args.folds,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        fit_max_length=args.fit_max_length,
        eval_max_length=args.eval_max_length,
        dtype=args.dtype,
        force=args.force,
        verify=verify,
        verify_split_families=list({d.split_family for d in datasets}),
        verify_only=args.verify_only,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = parse_args()
    jobs = list(iter_extraction_jobs(config))
    if not jobs:
        raise RuntimeError("No extraction jobs generated from the current config.")

    if not config.verify_only:
        for model_name in config.models:
            model_jobs = [j for j in jobs if j.model_name == model_name]
            run_for_model(model_name, config, model_jobs)

    if config.verify is not None:
        run_verification(config)


if __name__ == "__main__":
    main()
