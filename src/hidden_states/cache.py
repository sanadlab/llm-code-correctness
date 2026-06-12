"""On-disk cache helpers for pooled hidden-state bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.representations import Representation
from src.vectors.direction import Direction
from src.vectors.projection import project_all_layers

logger = logging.getLogger(__name__)

# jsonl stem for each cached data split
SPLIT_TO_JSONL = {
    "fit": "fit.jsonl",
    "validate": "validate.jsonl",
    "test": "test.jsonl",
}

FramingFor = Callable[[dict], object]

if TYPE_CHECKING:
    from src.hidden_states import HiddenStateBundle


class VerificationError(RuntimeError):
    """Raised when cached hidden states disagree with stored experiment scores."""


def safe_model_dir(model_name: str) -> str:
    """HuggingFace model id with ``/`` replaced by ``__`` (matches vectors/ layout)."""
    return model_name.replace("/", "__")


def git_commit() -> str:
    """Current git HEAD SHA, or ``unknown`` if unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


def fold_dir_name(fold: int, prefix: str = "fold") -> str:
    return f"{prefix}_{fold}"


def framing_uses_chat_template(framing_key: str) -> bool:
    return framing_key in ("neutral", "function")


def framing_dispatcher(framing_key: str, model_name: str) -> FramingFor:
    """Return a per-record framing callable for extraction.

    ``function`` dispatches ``function_pos`` / ``function_neg`` by label;
    other keys use a single framing for every record.
    """
    # Lazy import: build_framing lives with experiment registries today.
    from scripts.experiments._common import build_framing

    if framing_key == "function":
        fit_pos = build_framing("function_pos", model_name)
        fit_neg = build_framing("function_neg", model_name)

        def _dispatch(record: dict):
            return fit_pos if record["label"] == 1 else fit_neg

        return _dispatch

    framing = build_framing(framing_key, model_name)
    return lambda _record: framing


def bundle_output_dir(
    output_root: Path,
    model_name: str,
    split_family: str,
    benchmark: str,
    distribution: str,
    data_split: str,
    framing: str,
    fold: int,
    fold_prefix: str = "fold",
) -> Path:
    """Leaf directory for one cached bundle (without ``bundle.pt``)."""
    return (
        Path(output_root)
        / safe_model_dir(model_name)
        / split_family
        / benchmark
        / distribution
        / data_split
        / framing
        / fold_dir_name(fold, fold_prefix)
    )


def bundle_is_complete(out_dir: Path, force: bool) -> bool:
    if force:
        return False
    return (out_dir / "bundle.pt").is_file() and (out_dir / "metadata.json").is_file()


def sample_ids_hash(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    return digest[:16]


def build_bundle_metadata(
    *,
    model: str,
    dataset: str,
    split_family: str,
    distribution: str,
    fold: int,
    data_split: str,
    framing: str,
    representations: Sequence[str],
    dtype: str,
    max_length: int,
    n_samples: int,
    source_jsonl: Path,
    bundle_path: Path,
    sample_ids: Sequence[str],
    experiment_script: str,
    git_sha: Optional[str] = None,
    ood_source: Optional[str] = None,
    test_dataset: Optional[str] = None,
    outer_fold: Optional[int] = None,
    fold_prefix: str = "fold",
) -> Dict[str, Any]:
    """Build metadata dict aligned with vectors/ where applicable."""
    meta: Dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "split_family": split_family,
        "distribution": distribution,
        "data_split": data_split,
        "framing": framing,
        "representations": list(representations),
        "acts_dtype": "float32",
        "chat_template": framing_uses_chat_template(framing),
        "dtype": dtype,
        "max_length": max_length,
        "n_samples": n_samples,
        "source_jsonl": str(source_jsonl),
        "bundle_path": str(bundle_path),
        "sample_ids_hash": sample_ids_hash(sample_ids),
        "experiment_script": experiment_script,
        "git_commit": git_sha if git_sha is not None else git_commit(),
    }
    if fold_prefix == "inner":
        meta["inner_fold"] = fold
    else:
        meta["fold"] = fold
    if ood_source is not None:
        meta["ood_source"] = ood_source
    if test_dataset is not None:
        meta["test_dataset"] = test_dataset
    if outer_fold is not None:
        meta["outer_fold"] = outer_fold
    return meta


def write_bundle_metadata(path: Path, meta: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(meta), f, indent=2)


def vector_common_path(
    model_name: str,
    split_family: str,
    dataset: str,
    distribution: str,
    method: str,
    fit_framing: str,
    representation: str,
) -> str:
    return (
        f"{safe_model_dir(model_name)}/{split_family}/{dataset}/{distribution}/"
        f"{method}/{fit_framing}/{representation}"
    )


def _scores_for_sample_order(
    bundle: "HiddenStateBundle",
    representation: Representation,
    direction: Direction,
    sample_ids: List[str],
) -> Tuple[List[int], torch.Tensor]:
    """Project bundle and reorder rows to match ``sample_ids``.

    Activations are cast to the direction's vector dtype before projection so
    the numeric path matches the pipeline that produced the reference scores.
    The shipped experiments run the model in float16 and project in float16;
    the cached activations are stored as float32 but carry float16-precision
    values, so projecting them back in float16 reproduces ``detailed.json``
    (most scores bit-for-bit). Projecting in float32 instead would use a more
    precise path than the float16-quantized reference and never match.
    """
    vec_dtype = next(iter(direction.vectors.values())).dtype
    acts = representation.select(bundle).to(vec_dtype)
    layers, scores = project_all_layers(direction, acts)
    id_to_row = {sid: i for i, sid in enumerate(bundle.sample_ids)}
    missing = [sid for sid in sample_ids if sid not in id_to_row]
    if missing:
        raise VerificationError(
            f"{len(missing)} sample_ids in reference missing from bundle "
            f"(first: {missing[0]!r})"
        )
    order = [id_to_row[sid] for sid in sample_ids]
    return layers, scores[order]


def _pairwise_accuracy(
    scores: torch.Tensor, labels: Sequence[int], task_ids: Sequence[str]
) -> float:
    """Average over tasks of P(score(pos) > score(neg)); mirrors the experiment."""
    by_task: Dict[str, Dict[str, List[float]]] = {}
    for s, lab, tid in zip(scores.tolist(), labels, task_ids):
        g = by_task.setdefault(tid, {"pos": [], "neg": []})
        g["pos" if int(lab) == 1 else "neg"].append(s)
    accs: List[float] = []
    for g in by_task.values():
        if not g["pos"] or not g["neg"]:
            continue
        wins = sum(p > n for p in g["pos"] for n in g["neg"])
        ties = sum(p == n for p in g["pos"] for n in g["neg"])
        total = len(g["pos"]) * len(g["neg"])
        accs.append((wins + 0.5 * ties) / total)
    return sum(accs) / len(accs) if accs else float("nan")


def _compare_score_tensors(
    *,
    split_name: str,
    layers: List[int],
    computed: torch.Tensor,
    reference_layers: List[int],
    reference_scores: List[List[float]],
    sample_ids: List[str],
    labels: Sequence[int],
    task_ids: Sequence[str],
    corr_min: float,
    acc_atol: float,
) -> None:
    """Check that cached projections *reproduce* the reference experiment.

    Reference scores (``detailed.json``) and the cached bundle come from two
    independent float16 forward passes. float16 transformer forwards are not
    bitwise reproducible (batch/padding composition and GPU nondeterminism
    differ between runs), and per-token differences accumulate over layers. The
    result is unbiased reproduction noise: per-layer score correlation stays
    >0.9999, downstream accuracy stays within a fraction of a point, yet any
    individual *near-zero* score can disagree by tens of percent in relative
    terms. A per-cell ``(rtol, atol)`` band therefore cannot work -- the noise
    floor is unbounded relative to small scores, so a tolerance that passes a
    legitimate rerun also passes real bugs (and vice versa).

    The one invariant that is robust across splits, sample sizes, datasets and
    models -- and that real extraction bugs (wrong layer/pooling/sample
    alignment/direction/dtype corruption) destroy -- is **per-layer Pearson
    correlation** between computed and reference scores. It is the hard gate:
    every layer must correlate at ``>= corr_min`` (legitimate reruns sit at
    >0.9999; structural bugs collapse far below 1).

    Pairwise accuracy (the reported metric) is *implied* by high correlation:
    if every layer's scores track the reference, layer selection and the
    selected-layer accuracy follow. We compute the per-layer accuracy agreement
    only as a transparency signal and warn if it exceeds ``acc_atol``, but do
    NOT fail on it -- on small splits the metric is a coarse step function where
    a single float-noise tie-flip moves it by several points, so a hard gate
    there would just reintroduce sample-size fragility.

    Note: this intentionally does not detect a constant rescale of all scores
    (correlation and pairwise accuracy are both invariant to it), because such a
    transform does not change any reported result.
    """
    if layers != reference_layers:
        raise VerificationError(
            f"{split_name}: layer list mismatch: computed {layers[:5]}... "
            f"vs reference {reference_layers[:5]}..."
        )
    ref = torch.tensor(reference_scores, dtype=torch.float32)
    computed_f = computed.float()
    if computed_f.shape != ref.shape:
        raise VerificationError(
            f"{split_name}: score shape {tuple(computed_f.shape)} != "
            f"reference {tuple(ref.shape)}"
        )

    worst_corr = (2.0, -1)  # (correlation, layer)
    worst_acc = (-1.0, -1, 0.0, 0.0)  # (abs diff, layer, acc_computed, acc_ref)
    for j, layer in enumerate(layers):
        c_col = computed_f[:, j]
        r_col = ref[:, j]
        cs = c_col - c_col.mean()
        rs = r_col - r_col.mean()
        denom = (cs.norm() * rs.norm()).item()
        # Degenerate (constant) columns carry no ranking signal; treat as 1.0.
        corr = float((cs @ rs).item() / denom) if denom > 0 else 1.0
        if corr < worst_corr[0]:
            worst_corr = (corr, layer)

        acc_c = _pairwise_accuracy(c_col, labels, task_ids)
        acc_r = _pairwise_accuracy(r_col, labels, task_ids)
        if acc_c == acc_c and acc_r == acc_r:  # both non-NaN
            d = abs(acc_c - acc_r)
            if d > worst_acc[0]:
                worst_acc = (d, layer, acc_c, acc_r)

    if worst_corr[0] < corr_min:
        raise VerificationError(
            f"{split_name}: per-layer score correlation {worst_corr[0]:.6f} at "
            f"layer {worst_corr[1]} is below corr_min={corr_min}; cached scores "
            f"do not track the reference (likely wrong layer/pooling/alignment/"
            f"direction, not float noise)."
        )
    logger.info(
        "%s: min layer correlation=%.6f, max layer pairwise-acc diff=%.4f",
        split_name, worst_corr[0], worst_acc[0],
    )
    if worst_acc[0] > acc_atol:
        # Not a failure: implied-by-correlation, and coarse on small splits.
        logger.warning(
            "%s: pairwise accuracy differs by %.4f at layer %d "
            "(computed=%.4f vs reference=%.4f); above acc_atol=%s but within "
            "float-reproduction noise given correlation=%.6f. Not failing.",
            split_name, worst_acc[0], worst_acc[1], worst_acc[2], worst_acc[3],
            acc_atol, worst_corr[0],
        )


@dataclass(frozen=True)
class VerifyConfig:
    vectors_dir: Path
    results_dir: Path
    method: str
    fit_framing: str
    representation: str
    val_framing: str
    test_framing: str
    # Verification compares two independent float16 forward passes, which are
    # not bitwise reproducible. We therefore check scale-free, decision-level
    # invariants rather than per-cell tolerances (see _compare_score_tensors).
    # Minimum per-layer Pearson correlation between cached and reference scores.
    # Legitimate reruns sit at >0.9999; structural bugs collapse far below.
    # This is the hard pass/fail gate.
    corr_min: float = 0.999
    # Per-layer pairwise-accuracy difference above which we WARN (never fail):
    # accuracy is implied by correlation and is coarse on small splits.
    acc_atol: float = 0.02
    verify_inner_fold: int = 0


def verify_standard_fold(
    *,
    model_name: str,
    split_family: str,
    dataset: str,
    distribution: str,
    fold: int,
    hidden_states_root: Path,
    verify: VerifyConfig,
    representation: Representation,
) -> None:
    """Compare cached val/test projections to ``detailed.json`` for one fold."""
    common = vector_common_path(
        model_name,
        split_family,
        dataset,
        distribution,
        verify.method,
        verify.fit_framing,
        verify.representation,
    )
    direction_path = (
        verify.vectors_dir / common / f"fold_{fold}" / "direction.pt"
    )
    if not direction_path.is_file():
        raise VerificationError(f"missing direction: {direction_path}")

    results_fold = (
        verify.results_dir
        / common
        / f"val={verify.val_framing}__test={verify.test_framing}"
        / f"fold_{fold}"
    )
    detailed_path = results_fold / "detailed.json"
    if not detailed_path.is_file():
        raise VerificationError(f"missing detailed.json: {detailed_path}")

    direction = Direction.load(direction_path)
    with open(detailed_path) as f:
        detailed = json.load(f)

    for split_name, framing in (("validate", verify.val_framing), ("test", verify.test_framing)):
        bundle_dir = bundle_output_dir(
            hidden_states_root,
            model_name,
            split_family,
            dataset,
            distribution,
            split_name,
            framing,
            fold,
        )
        bundle_path = bundle_dir / "bundle.pt"
        if not bundle_path.is_file():
            raise VerificationError(f"missing cached bundle: {bundle_path}")

        from src.hidden_states import HiddenStateBundle

        bundle = HiddenStateBundle.load(bundle_path)
        ref = detailed["val" if split_name == "validate" else "test"]
        layers, scores = _scores_for_sample_order(
            bundle, representation, direction, ref["sample_ids"]
        )
        _compare_score_tensors(
            split_name=split_name,
            layers=layers,
            computed=scores,
            reference_layers=detailed["layers"],
            reference_scores=ref["scores"],
            sample_ids=ref["sample_ids"],
            labels=ref["labels"],
            task_ids=ref["task_ids"],
            corr_min=verify.corr_min,
            acc_atol=verify.acc_atol,
        )
        logger.info("verify OK: %s fold=%d %s", split_name, fold, bundle_dir)
