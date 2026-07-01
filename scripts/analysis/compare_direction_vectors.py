"""Refit direction vectors from cache and compare against vectors/ on disk.

``fit_and_score_from_cache.py`` fits a per-layer direction but only persists
scores (``results.json`` / ``detailed.json``). This script repeats that fit
step from cached **fit** bundles, optionally writes the recomputed
``direction.pt`` files, and compares them to existing vectors with matching
configuration keys.

Typical usage (scan all configs under vectors/):

    python scripts/analysis/compare_direction_vectors.py \\
        --hidden-states-root /scratch/re2230/hidden_states_re2230 \\
        --vectors-root vectors \\
        --output-vectors-root vectors_refit_from_cache \\
        --summary-csv analysis/direction_compare/summary.csv

Single configuration (same knobs as fit_and_score_from_cache in-domain):

    python scripts/analysis/compare_direction_vectors.py \\
        --model codellama/CodeLlama-7b-Instruct-hf \\
        --split-family icse26 \\
        --benchmark bigcodebench \\
        --method mean_diff --representation response_avg \\
        --fit-framing function \\
        --hidden-states-root /scratch/re2230/hidden_states_sohaila \\
        --folds 0

Or use the bash wrapper for icse26 + hidden_states_sohaila:

    ./scripts/analysis/Bash_Scripts/run_compare_direction_icse26.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments._common import METHODS, REPRESENTATIONS
from scripts.experiments.fit_vectors_from_cache import (
    _assert_representation,
    _common_path,
    _fit_direction,
    _load_bundle,
)
from src.hidden_states import git_commit, safe_model_dir
from src.vectors.direction import Direction

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN_STATES_ROOT = _REPO_ROOT.parent / "hidden_states_re2230"
DEFAULT_VECTORS_ROOT = _REPO_ROOT / "vectors"
EXPERIMENT_SCRIPT = "scripts/analysis/compare_direction_vectors.py"

_REL_PATH_RE = re.compile(
    r"^(?P<model>[^/]+)/(?P<split_family>[^/]+)/(?P<benchmark>[^/]+)/"
    r"(?P<distribution>[^/]+)/(?P<method>[^/]+)/(?P<fit_framing>[^/]+)/"
    r"(?P<representation>[^/]+)/(?P<fold_dir>fold|inner)_(?P<fold>\d+)/direction\.pt$"
)


@dataclass(frozen=True)
class VectorConfig:
    model: str
    split_family: str
    benchmark: str
    distribution: str
    method: str
    fit_framing: str
    representation: str
    fold: int
    fold_prefix: str
    reference_path: Path

    @property
    def common_path(self) -> str:
        return _common_path(
            self.model,
            self.split_family,
            self.benchmark,
            self.distribution,
            self.method,
            self.fit_framing,
            self.representation,
        )

    def match_key(self) -> Tuple[str, ...]:
        return (
            self.model,
            self.split_family,
            self.benchmark,
            self.distribution,
            self.method,
            self.fit_framing,
            self.representation,
            str(self.fold),
        )


@dataclass
class LayerComparison:
    layer: int
    cosine: float
    signed_cosine: float
    rel_l2_diff: float
    max_abs_diff: float
    allclose: bool


@dataclass
class FoldComparison:
    config: VectorConfig
    n_layers: int
    mean_cosine: float
    min_cosine: float
    max_rel_l2_diff: float
    max_max_abs_diff: float
    all_layers_allclose: bool
    missing_reference: bool
    missing_cache: bool
    error: Optional[str]
    refitted_path: Optional[Path]
    per_layer: List[LayerComparison]


def _parse_model_from_metadata(meta: dict, rel_parts: Sequence[str]) -> str:
    if meta.get("model"):
        return str(meta["model"])
    # safe_model_dir is reversible only in the sense that we store the slug;
    # metadata is the source of truth when present.
    return rel_parts[0].replace("__", "/")


def _parse_config_from_path(direction_path: Path, vectors_root: Path) -> Optional[VectorConfig]:
    rel = direction_path.relative_to(vectors_root).as_posix()
    match = _REL_PATH_RE.match(rel)
    if not match:
        return None

    fold_dir = direction_path.parent
    meta: dict = {}
    meta_path = fold_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())

    parts = rel.split("/")
    model = _parse_model_from_metadata(meta, parts)
    benchmark = str(meta.get("dataset") or meta.get("benchmark") or match.group("benchmark"))
    return VectorConfig(
        model=model,
        split_family=str(meta.get("split_family") or match.group("split_family")),
        benchmark=benchmark,
        distribution=str(meta.get("distribution") or match.group("distribution")),
        method=str(meta.get("method") or match.group("method")),
        fit_framing=str(meta.get("fit_framing") or match.group("fit_framing")),
        representation=str(meta.get("representation") or match.group("representation")),
        fold=int(meta.get("fold", match.group("fold"))),
        fold_prefix=str(meta.get("fold_prefix") or match.group("fold_dir")),
        reference_path=direction_path,
    )


def discover_vector_configs(vectors_root: Path) -> List[VectorConfig]:
    configs: List[VectorConfig] = []
    for direction_path in sorted(vectors_root.rglob("direction.pt")):
        cfg = _parse_config_from_path(direction_path, vectors_root)
        if cfg is None:
            logger.warning("Skipping unrecognized path layout: %s", direction_path)
            continue
        configs.append(cfg)
    return configs


def _filter_configs(
    configs: Iterable[VectorConfig],
    *,
    model: Optional[str],
    split_family: Optional[str],
    benchmark: Optional[str],
    distribution: Optional[str],
    method: Optional[str],
    fit_framing: Optional[str],
    representation: Optional[str],
    folds: Optional[Sequence[int]],
) -> List[VectorConfig]:
    selected: List[VectorConfig] = []
    for cfg in configs:
        if model is not None and cfg.model != model:
            continue
        if split_family is not None and cfg.split_family != split_family:
            continue
        if benchmark is not None and cfg.benchmark != benchmark:
            continue
        if distribution is not None and cfg.distribution != distribution:
            continue
        if method is not None and cfg.method != method:
            continue
        if fit_framing is not None and cfg.fit_framing != fit_framing:
            continue
        if representation is not None and cfg.representation != representation:
            continue
        if folds is not None and cfg.fold not in folds:
            continue
        selected.append(cfg)
    return selected


def _build_config_from_args(args: argparse.Namespace) -> List[VectorConfig]:
    if args.model is None:
        raise ValueError("--model is required when not using --scan-vectors-root")
    folds = args.folds if args.folds is not None else list(range(10))
    common = _common_path(
        args.model,
        args.split_family,
        args.benchmark,
        args.distribution,
        args.method,
        args.fit_framing,
        args.representation,
    )
    configs: List[VectorConfig] = []
    for fold in folds:
        ref_path = args.vectors_root / common / f"fold_{fold}" / "direction.pt"
        configs.append(
            VectorConfig(
                model=args.model,
                split_family=args.split_family,
                benchmark=args.benchmark,
                distribution=args.distribution,
                method=args.method,
                fit_framing=args.fit_framing,
                representation=args.representation,
                fold=fold,
                fold_prefix=args.fold_prefix,
                reference_path=ref_path,
            )
        )
    return configs


def fit_direction_for_config(
    config: VectorConfig,
    hidden_states_root: Path,
    *,
    fold_prefix: str = "fold",
) -> Direction:
    representation = REPRESENTATIONS[config.representation]
    method = METHODS[config.method]
    bundle_fit, fit_dir = _load_bundle(
        hidden_states_root,
        config.model,
        config.split_family,
        config.benchmark,
        config.distribution,
        "fit",
        config.fit_framing,
        config.fold,
        fold_prefix,
    )
    _assert_representation(bundle_fit, fit_dir, representation)
    return _fit_direction(bundle_fit, representation, method)


def _compare_layer(
    ref_vec: torch.Tensor,
    new_vec: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> LayerComparison:
    v_ref = ref_vec.detach().float().cpu()
    v_new = new_vec.detach().float().cpu()
    signed_cos = torch.nn.functional.cosine_similarity(
        v_new.unsqueeze(0), v_ref.unsqueeze(0)
    ).item()
    cosine = abs(signed_cos)
    diff = v_new - v_ref
    denom = max(float(v_ref.norm().item()), 1e-12)
    rel_l2 = float(diff.norm().item() / denom)
    max_abs = float(diff.abs().max().item())
    close = torch.allclose(v_new, v_ref, rtol=rtol, atol=atol)
    return LayerComparison(
        layer=-1,
        cosine=cosine,
        signed_cosine=signed_cos,
        rel_l2_diff=rel_l2,
        max_abs_diff=max_abs,
        allclose=bool(close),
    )


def compare_directions(
    reference: Direction,
    candidate: Direction,
    *,
    rtol: float,
    atol: float,
) -> List[LayerComparison]:
    shared_layers = sorted(set(reference.layers) & set(candidate.layers))
    if not shared_layers:
        raise ValueError(
            f"No shared layers between reference ({reference.layers}) "
            f"and candidate ({candidate.layers})"
        )

    only_ref = sorted(set(reference.layers) - set(candidate.layers))
    only_new = sorted(set(candidate.layers) - set(reference.layers))
    if only_ref or only_new:
        logger.warning(
            "Layer mismatch: only_in_reference=%s only_in_candidate=%s",
            only_ref,
            only_new,
        )

    comparisons: List[LayerComparison] = []
    for layer in shared_layers:
        layer_cmp = _compare_layer(
            reference.vectors[layer],
            candidate.vectors[layer],
            rtol=rtol,
            atol=atol,
        )
        comparisons.append(
            LayerComparison(
                layer=layer,
                cosine=layer_cmp.cosine,
                signed_cosine=layer_cmp.signed_cosine,
                rel_l2_diff=layer_cmp.rel_l2_diff,
                max_abs_diff=layer_cmp.max_abs_diff,
                allclose=layer_cmp.allclose,
            )
        )
    return comparisons


def _persist_refitted(
    output_vectors_root: Path,
    config: VectorConfig,
    direction: Direction,
    *,
    n_fit: Optional[int] = None,
) -> Path:
    out_fold = output_vectors_root / config.common_path / f"fold_{config.fold}"
    out_fold.mkdir(parents=True, exist_ok=True)
    out_path = out_fold / "direction.pt"
    direction.save(out_path)
    meta = {
        "model": config.model,
        "dataset": config.benchmark,
        "split_family": config.split_family,
        "distribution": config.distribution,
        "fold": config.fold,
        "method": config.method,
        "representation": config.representation,
        "fit_framing": config.fit_framing,
        "reference_vector_path": str(config.reference_path),
        "n_fit": n_fit,
        "experiment_script": EXPERIMENT_SCRIPT,
        "git_commit": git_commit(),
    }
    with open(out_fold / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_path


def compare_fold(
    config: VectorConfig,
    *,
    hidden_states_root: Path,
    output_vectors_root: Optional[Path],
    rtol: float,
    atol: float,
) -> FoldComparison:
    refitted_path: Optional[Path] = None
    try:
        candidate = fit_direction_for_config(
            config, hidden_states_root, fold_prefix=config.fold_prefix
        )
    except FileNotFoundError as exc:
        return FoldComparison(
            config=config,
            n_layers=0,
            mean_cosine=float("nan"),
            min_cosine=float("nan"),
            max_rel_l2_diff=float("nan"),
            max_max_abs_diff=float("nan"),
            all_layers_allclose=False,
            missing_reference=not config.reference_path.is_file(),
            missing_cache=True,
            error=str(exc),
            refitted_path=None,
            per_layer=[],
        )
    except Exception as exc:  # noqa: BLE001 - surface fit failures per fold
        return FoldComparison(
            config=config,
            n_layers=0,
            mean_cosine=float("nan"),
            min_cosine=float("nan"),
            max_rel_l2_diff=float("nan"),
            max_max_abs_diff=float("nan"),
            all_layers_allclose=False,
            missing_reference=not config.reference_path.is_file(),
            missing_cache=False,
            error=str(exc),
            refitted_path=None,
            per_layer=[],
        )

    if output_vectors_root is not None:
        refitted_path = _persist_refitted(output_vectors_root, config, candidate)

    if not config.reference_path.is_file():
        return FoldComparison(
            config=config,
            n_layers=len(candidate.layers),
            mean_cosine=float("nan"),
            min_cosine=float("nan"),
            max_rel_l2_diff=float("nan"),
            max_max_abs_diff=float("nan"),
            all_layers_allclose=False,
            missing_reference=True,
            missing_cache=False,
            error="reference direction.pt not found",
            refitted_path=refitted_path,
            per_layer=[],
        )

    reference = Direction.load(config.reference_path)
    per_layer = compare_directions(reference, candidate, rtol=rtol, atol=atol)
    cosines = [row.cosine for row in per_layer]
    return FoldComparison(
        config=config,
        n_layers=len(per_layer),
        mean_cosine=sum(cosines) / len(cosines),
        min_cosine=min(cosines),
        max_rel_l2_diff=max(row.rel_l2_diff for row in per_layer),
        max_max_abs_diff=max(row.max_abs_diff for row in per_layer),
        all_layers_allclose=all(row.allclose for row in per_layer),
        missing_reference=False,
        missing_cache=False,
        error=None,
        refitted_path=refitted_path,
        per_layer=per_layer,
    )


def write_summary_csv(path: Path, rows: List[FoldComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "split_family",
        "benchmark",
        "distribution",
        "method",
        "fit_framing",
        "representation",
        "fold",
        "n_layers",
        "mean_cosine",
        "min_cosine",
        "max_rel_l2_diff",
        "max_max_abs_diff",
        "all_layers_allclose",
        "missing_reference",
        "missing_cache",
        "error",
        "reference_path",
        "refitted_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cfg = row.config
            writer.writerow(
                {
                    "model": cfg.model,
                    "split_family": cfg.split_family,
                    "benchmark": cfg.benchmark,
                    "distribution": cfg.distribution,
                    "method": cfg.method,
                    "fit_framing": cfg.fit_framing,
                    "representation": cfg.representation,
                    "fold": cfg.fold,
                    "n_layers": row.n_layers,
                    "mean_cosine": row.mean_cosine,
                    "min_cosine": row.min_cosine,
                    "max_rel_l2_diff": row.max_rel_l2_diff,
                    "max_max_abs_diff": row.max_max_abs_diff,
                    "all_layers_allclose": row.all_layers_allclose,
                    "missing_reference": row.missing_reference,
                    "missing_cache": row.missing_cache,
                    "error": row.error or "",
                    "reference_path": str(cfg.reference_path),
                    "refitted_path": str(row.refitted_path) if row.refitted_path else "",
                }
            )


def write_per_layer_csv(path: Path, rows: List[FoldComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "split_family",
        "benchmark",
        "method",
        "fit_framing",
        "representation",
        "fold",
        "layer",
        "cosine",
        "signed_cosine",
        "rel_l2_diff",
        "max_abs_diff",
        "allclose",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fold_row in rows:
            cfg = fold_row.config
            for layer_row in fold_row.per_layer:
                writer.writerow(
                    {
                        "model": cfg.model,
                        "split_family": cfg.split_family,
                        "benchmark": cfg.benchmark,
                        "method": cfg.method,
                        "fit_framing": cfg.fit_framing,
                        "representation": cfg.representation,
                        "fold": cfg.fold,
                        "layer": layer_row.layer,
                        "cosine": layer_row.cosine,
                        "signed_cosine": layer_row.signed_cosine,
                        "rel_l2_diff": layer_row.rel_l2_diff,
                        "max_abs_diff": layer_row.max_abs_diff,
                        "allclose": layer_row.allclose,
                    }
                )


def _print_fold_summary(row: FoldComparison) -> None:
    cfg = row.config
    prefix = (
        f"[{safe_model_dir(cfg.model)} | {cfg.split_family} | {cfg.benchmark} | "
        f"{cfg.method}/{cfg.fit_framing}/{cfg.representation} | fold {cfg.fold}]"
    )
    if row.error:
        logger.warning("%s ERROR: %s", prefix, row.error)
        return
    status = "MATCH" if row.all_layers_allclose else "DIFFER"
    logger.info(
        "%s %s mean_cos=%.8f min_cos=%.8f max_rel_l2=%.2e max_abs=%.2e",
        prefix,
        status,
        row.mean_cosine,
        row.min_cosine,
        row.max_rel_l2_diff,
        row.max_max_abs_diff,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit directions from cache and compare to vectors/."
    )
    parser.add_argument(
        "--hidden-states-root",
        type=Path,
        default=DEFAULT_HIDDEN_STATES_ROOT,
        help=f"Root with cached bundle.pt files (default: {DEFAULT_HIDDEN_STATES_ROOT}).",
    )
    parser.add_argument(
        "--vectors-root",
        type=Path,
        default=DEFAULT_VECTORS_ROOT,
        help=f"Reference vectors root (default: {DEFAULT_VECTORS_ROOT}).",
    )
    parser.add_argument(
        "--output-vectors-root",
        type=Path,
        default=None,
        help="Optional directory to write refitted direction.pt files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=_REPO_ROOT / "analysis" / "direction_compare" / "summary.csv",
        help="Per-fold comparison summary CSV.",
    )
    parser.add_argument(
        "--per-layer-csv",
        type=Path,
        default=None,
        help="Optional per-layer detail CSV (can be large).",
    )
    parser.add_argument(
        "--scan-vectors-root",
        action="store_true",
        help="Discover all direction.pt configs under --vectors-root (default when --model omitted).",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--split-family", default="icse26_val_1_4")
    parser.add_argument("--filter-split-family", action="store_true")
    parser.add_argument("--benchmark", default="bigcodebench", choices=["bigcodebench", "humaneval"])
    parser.add_argument("--filter-benchmark", action="store_true")
    parser.add_argument("--distribution", default="id")
    parser.add_argument("--filter-distribution", action="store_true")
    parser.add_argument("--method", default="mean_diff", choices=list(METHODS))
    parser.add_argument("--filter-method", action="store_true")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--filter-fit-framing", action="store_true")
    parser.add_argument("--representation", default="response_avg", choices=list(REPRESENTATIONS))
    parser.add_argument("--filter-representation", action="store_true")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--fold-prefix",
        default="fold",
        help="Fold directory prefix in hidden_states/ (use 'inner' for OOD inner folds).",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def resolve_configs(args: argparse.Namespace) -> List[VectorConfig]:
    scan = args.scan_vectors_root or args.model is None
    if scan:
        configs = discover_vector_configs(args.vectors_root)
        # In scan mode, only apply non-fold filters when --model is set.
        return _filter_configs(
            configs,
            model=args.model,
            split_family=args.split_family if args.filter_split_family else None,
            benchmark=args.benchmark if args.filter_benchmark else None,
            distribution=args.distribution if args.filter_distribution else None,
            method=args.method if args.filter_method else None,
            fit_framing=args.fit_framing if args.filter_fit_framing else None,
            representation=args.representation if args.filter_representation else None,
            folds=args.folds,
        )

    return _build_config_from_args(args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = parse_args()
    configs = resolve_configs(args)
    if not configs:
        raise SystemExit("No vector configurations matched the filters.")

    logger.info("Hidden states root: %s", args.hidden_states_root.resolve())
    logger.info("Reference vectors root: %s", args.vectors_root.resolve())
    if args.output_vectors_root is not None:
        logger.info("Output vectors root: %s", args.output_vectors_root.resolve())
    logger.info("Comparing %d fold-level configuration(s)", len(configs))

    rows: List[FoldComparison] = []
    for cfg in configs:
        rows.append(
            compare_fold(
                cfg,
                hidden_states_root=args.hidden_states_root,
                output_vectors_root=args.output_vectors_root,
                rtol=args.rtol,
                atol=args.atol,
            )
        )
        _print_fold_summary(rows[-1])

    write_summary_csv(args.summary_csv, rows)
    logger.info("Wrote summary CSV: %s", args.summary_csv.resolve())

    if args.per_layer_csv is not None:
        write_per_layer_csv(args.per_layer_csv, rows)
        logger.info("Wrote per-layer CSV: %s", args.per_layer_csv.resolve())

    ok = [r for r in rows if r.error is None and not r.missing_reference]
    if ok:
        mean_of_means = sum(r.mean_cosine for r in ok) / len(ok)
        n_match = sum(1 for r in ok if r.all_layers_allclose)
        logger.info(
            "[aggregate] compared=%d exact_match=%d mean_cosine=%.8f",
            len(ok),
            n_match,
            mean_of_means,
        )


if __name__ == "__main__":
    main()
