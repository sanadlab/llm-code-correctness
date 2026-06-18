"""Cache-only correlation of the residual-PCA y-axis against surface code metrics.

For each task (a set of k implementations, expected k=4: 1 correct + 3 incorrect)
we build a per-task scatter where the x-axis is the fold-specific reading
vector (correctness direction) and the y-axis is the top principal component of
the *residuals* (hidden states with the reading-vector component projected out).
This script measures **what the y-axis represents beyond correctness** by
correlating that y-axis against surface-level code metrics, and how the tracking
changes **across layers**.

Reads only precomputed artifacts (no model inference):
  - hidden states from ``hidden_states/.../bundle.pt`` (``response_avg`` acts)
  - per-fold reading vectors from ``vectors/.../direction.pt``
  - validation-selected layer from ``results/.../results.json``
  - implementation source code from the dataset fold JSONL (``load_split``)
  - fold count from ``manifest.json`` (via ``--folds all``)

Aggregation is two-level because the reading vector is per fold: a task's r-squared
is only defined relative to a fold's reading vector, so we average within fold
first (Level 1), then macro-average across folds (Level 2).

Interpretation note: with k=4 implementations a metric with no real relationship
to the y-axis still averages r-squared ~= 1/3 (analytic n=4 chance level), so curves
must be read against the 1/3 chance line, not against 0. Only values clearly
above 1/3 indicate the y-axis tracks that surface metric.

Surface metrics do not depend on layer/fold/model and are computed once per task
and cached (``surface_metrics.csv``).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import sys
import textwrap
import warnings
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from scripts.analysis.viz_common import (
    parse_fold_selector,
    read_selected_layer,
    save_figure,
    splits_dir,
    viz_output_path_parts,
)
from scripts.experiments._common import load_split
from src.hidden_states.cache import bundle_output_dir, safe_model_dir, vector_common_path
from src.vectors.direction import Direction

logger = logging.getLogger(__name__)

# Analytic chance level for r-squared with k=4 points (1 correct + 3 incorrect).
NULL_BASELINE = 1.0 / 3.0

METRICS: Tuple[str, ...] = (
    "levenshtein_to_correct",
    "ast_dist_to_correct",
    "loc",
    "cyclomatic",
)

SURFACE_COLUMNS: Tuple[str, ...] = (
    "task_id",
    "sample_id",
    "label",
    "fold",
    "error_type",
    *METRICS,
)

STAGES: Tuple[str, ...] = ("metrics", "correlate", "aggregate", "plot", "all")

# Mutation operator -> error-type category (see dataset/.../mutant_ds_explained.tsv).
# [TODO] confirm this is the canonical error-type source for Mutation_DS plots.
OPERATOR_CATEGORY: Dict[str, str] = {
    "AOR": "arithmetic", "AOI": "arithmetic", "AOD": "arithmetic",
    "BOR": "bitwise", "BSOR": "bitwise", "BSOD": "bitwise",
    "COR": "logical", "COD": "logical", "NEG": "logical",
    "ROR": "relational",
    "SD": "statement", "RVM": "statement", "TR": "statement",
    "FR": "statement", "RC": "statement", "FCD": "statement",
    "UOI": "unary", "UOD": "unary",
    "CR": "value_replace", "VVR": "value_replace", "VCR": "value_replace",
    "CVR": "value_replace", "CRN": "value_replace",
}

# Fixed qualitative palette so a metric keeps the same color across figures.
_TAB10 = plt.get_cmap("tab10").colors
METRIC_COLORS: Dict[str, Any] = {metric: _TAB10[i] for i, metric in enumerate(METRICS)}


# ---------------------------------------------------------------------------
# Lazy imports for optional surface-metric dependencies
# ---------------------------------------------------------------------------

def _import_levenshtein():
    try:
        import Levenshtein  # type: ignore
    except ImportError as exc:  # pragma: no cover - guidance path
        raise ImportError(
            "The 'Levenshtein' package is required for levenshtein_to_correct. "
            "Install it with: pip install Levenshtein"
        ) from exc
    return Levenshtein


def _import_zss():
    try:
        import zss  # type: ignore
    except ImportError as exc:  # pragma: no cover - guidance path
        raise ImportError(
            "The 'zss' package is required for ast_dist_to_correct with --ast-backend zss. "
            "Install it with: pip install zss"
        ) from exc
    return zss


def _import_apted():
    try:
        from apted import APTED, Config  # type: ignore
    except ImportError as exc:  # pragma: no cover - guidance path
        raise ImportError(
            "The 'apted' package is required for ast_dist_to_correct with --ast-backend apted. "
            "Install it with: pip install apted"
        ) from exc
    return APTED, Config


def _import_radon():
    try:
        from radon import complexity as radon_complexity  # type: ignore
        from radon import raw as radon_raw  # type: ignore
    except ImportError as exc:  # pragma: no cover - guidance path
        raise ImportError(
            "The 'radon' package is required for loc/cyclomatic. "
            "Install it with: pip install radon"
        ) from exc
    return radon_raw, radon_complexity


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def dataset_label(split_family: str) -> str:
    """Human dataset name used in the ``dataset`` column and figure titles."""
    return "ICSE26_DS" if split_family == "icse26" else "Mutation_DS"


def load_bundle(bundle_path: str, representation: str = "response_avg") -> Dict[str, Any]:
    """Load a cached test bundle.

    Returns ``{sample_ids, task_ids, labels, acts}`` where ``acts`` is the
    ``[N, L+1, H]`` float32 tensor for *representation* (index 0 = embedding).
    """
    import torch

    blob = torch.load(Path(bundle_path), map_location="cpu", weights_only=False)
    acts_map = blob["acts"]
    if representation not in acts_map:
        raise KeyError(
            f"representation {representation!r} missing in bundle acts "
            f"({sorted(acts_map)}) at {bundle_path}"
        )
    return {
        "sample_ids": list(blob["sample_ids"]),
        "task_ids": list(blob["task_ids"]),
        "labels": blob["labels"].detach().cpu().numpy().astype(np.int64),
        "acts": acts_map[representation].detach().cpu().numpy().astype(np.float32),
    }


def _bundle_path_for_fold(
    *,
    hidden_states_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    fold: int,
    test_framing: str,
) -> Path:
    return (
        bundle_output_dir(
            output_root=hidden_states_root,
            model_name=model,
            split_family=split_family,
            benchmark=benchmark,
            distribution="id",
            data_split="test",
            framing=test_framing,
            fold=fold,
        )
        / "bundle.pt"
    )


def load_reading_vectors(
    *,
    vectors_root: Path,
    model: str,
    split_family: str,
    benchmark: str,
    method: str,
    fit_framing: str,
    representation: str,
    folds: Sequence[int],
    distribution: str = "id",
) -> Dict[int, Dict[int, np.ndarray]]:
    """Per-fold, per-layer reading vectors, unit-normalized.

    Returns ``reading_vectors[fold][layer] -> np.ndarray[H]`` (unit norm).
    Layer indexing matches the ``acts`` layer axis (0 = embedding).

    """
    common = vector_common_path(
        model, split_family, benchmark, distribution, method, fit_framing, representation
    )
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for fold in folds:
        direction_path = vectors_root / common / f"fold_{fold}" / "direction.pt"
        if not direction_path.is_file():
            raise FileNotFoundError(f"missing direction vector file: {direction_path}")
        direction = Direction.load(direction_path)
        if direction.representation and direction.representation != representation:
            raise ValueError(
                f"direction representation mismatch in {direction_path}: "
                f"expected {representation!r}, got {direction.representation!r}"
            )
        layer_map: Dict[int, np.ndarray] = {}
        for layer, vec in direction.vectors.items():
            signed = float(direction.signs[layer]) * vec.detach().cpu().numpy().astype(np.float64)
            norm = float(np.linalg.norm(signed))
            if norm == 0.0:
                raise ValueError(f"zero-norm reading vector at {direction_path} layer {layer}")
            layer_map[int(layer)] = signed / norm  # defensive unit-normalization
        out[int(fold)] = layer_map
    return out


def _fold_split_path(
    *, split_family: str, benchmark: str, data_root: Path, fold: int, data_split: str = "test"
) -> Path:
    return (
        splits_dir(split_family=split_family, benchmark=benchmark, data_root=data_root)
        / f"fold_{fold}"
        / f"{data_split}.jsonl"
    )


def load_fold_assignments(
    *,
    split_family: str,
    benchmark: str,
    data_root: Path,
    folds: Sequence[int],
    data_split: str = "test",
) -> Dict[int, List[str]]:
    """``{fold_idx: [held_out_task_ids]}`` from each fold's test split JSONL.

    """
    out: Dict[int, List[str]] = {}
    for fold in folds:
        path = _fold_split_path(
            split_family=split_family,
            benchmark=benchmark,
            data_root=data_root,
            fold=fold,
            data_split=data_split,
        )
        seen: List[str] = []
        seen_set = set()
        for rec in load_split(path):
            tid = rec["task_id"]
            if tid not in seen_set:
                seen_set.add(tid)
                seen.append(tid)
        out[int(fold)] = seen
    return out


# ---------------------------------------------------------------------------
# Surface metrics
# ---------------------------------------------------------------------------

def _task_error_type(sample_ids: Sequence[str], labels: Sequence[int], split_family: str) -> str:
    """Error-type category for a Mutation_DS task from its incorrect mutant ids.

    Returns "" for ICSE26 (no mutation operators), the shared category when all
    incorrect implementations share one, or "mixed" otherwise.
    """
    if split_family == "icse26":
        return ""
    cats = set()
    for sid, lab in zip(sample_ids, labels):
        if int(lab) == 1:
            continue
        parts = str(sid).split("__")
        if len(parts) >= 2:
            op = parts[1]
            cat = OPERATOR_CATEGORY.get(op)
            if cat is not None:
                cats.add(cat)
    if not cats:
        return "unknown"
    if len(cats) == 1:
        return next(iter(cats))
    return "mixed"


def _ast_label(node: ast.AST, normalize_identifiers: bool) -> str:
    """Node label: AST type only when normalizing, else with identifier payload."""
    label = type(node).__name__
    if not normalize_identifiers:
        if isinstance(node, ast.Name):
            label += f":{node.id}"
        elif isinstance(node, ast.arg):
            label += f":{node.arg}"
        elif isinstance(node, ast.Attribute):
            label += f":{node.attr}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            label += f":{node.name}"
        elif isinstance(node, ast.Constant):
            label += f":{node.value!r}"
    return label


class _AptedNode:
    """Minimal labeled tree node for APTED."""

    __slots__ = ("label", "children")

    def __init__(self, label: str) -> None:
        self.label = label
        self.children: List["_AptedNode"] = []


def _ast_to_apted(node: ast.AST, normalize_identifiers: bool) -> "_AptedNode":
    anode = _AptedNode(_ast_label(node, normalize_identifiers))
    for child in ast.iter_child_nodes(node):
        anode.children.append(_ast_to_apted(child, normalize_identifiers))
    return anode


def _ast_to_zss(node: ast.AST, normalize_identifiers: bool):
    zss = _import_zss()
    znode = zss.Node(_ast_label(node, normalize_identifiers))
    for child in ast.iter_child_nodes(node):
        znode.addkid(_ast_to_zss(child, normalize_identifiers))
    return znode


def _prepare_source(code: str, wrap_body: bool) -> str:
    """Make a source string parseable as a module.

    Mutation_DS stores bare function *bodies* (indented statements, often with
    top-level ``return``) rather than full modules, so they fail ``ast.parse``
    and ``radon.cc_visit`` directly. When ``wrap_body`` is set we dedent the body
    and nest it under a synthetic ``def`` so it parses. ICSE26 implementations are
    already complete modules and are returned unchanged.
    """
    if not wrap_body:
        return code
    return "def __wrap__():\n" + textwrap.indent(textwrap.dedent(code), "    ")


def _ast_distance(
    code_a: str,
    code_b: str,
    *,
    normalize_identifiers: bool,
    backend: str,
    max_nodes: int,
) -> float:
    """Tree edit distance between two source strings' ASTs.

    Sources must already be parseable modules (Mutation_DS bodies are wrapped by
    the caller via :func:`_prepare_source`). Returns ``nan`` if either source
    fails to parse, or if either AST exceeds ``max_nodes`` (tree edit distance is
    super-quadratic, so a node cap bounds worst-case runtime; pathologically
    large implementations are skipped).
    """
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
    except SyntaxError as exc:
        logger.warning("ast.parse failed (ast_dist_to_correct -> nan): %s", exc)
        return math.nan

    n_a = sum(1 for _ in ast.walk(tree_a))
    n_b = sum(1 for _ in ast.walk(tree_b))
    if max_nodes > 0 and (n_a > max_nodes or n_b > max_nodes):
        logger.warning(
            "ast too large (%d/%d > %d nodes); ast_dist_to_correct -> nan",
            n_a,
            n_b,
            max_nodes,
        )
        return math.nan

    if backend == "apted":
        APTED, Config = _import_apted()

        class _Cfg(Config):
            def children(self, node):  # noqa: D401
                return node.children

            def rename(self, a, b):
                return 0 if a.label == b.label else 1

        a = _ast_to_apted(tree_a, normalize_identifiers)
        b = _ast_to_apted(tree_b, normalize_identifiers)
        return float(APTED(a, b, _Cfg()).compute_edit_distance())

    zss = _import_zss()
    return float(
        zss.simple_distance(
            _ast_to_zss(tree_a, normalize_identifiers),
            _ast_to_zss(tree_b, normalize_identifiers),
        )
    )


def _loc(code: str) -> float:
    radon_raw, _ = _import_radon()
    try:
        return float(radon_raw.analyze(code).sloc)
    except Exception as exc:  # noqa: BLE001 - radon raises broad/parse errors
        logger.warning("radon.raw.analyze failed (loc -> nan): %s", exc)
        return math.nan


def _cyclomatic(code: str) -> float:
    """McCabe cyclomatic complexity (max over functions/blocks; [TODO] agg choice)."""
    _, radon_complexity = _import_radon()
    try:
        blocks = radon_complexity.cc_visit(code)
    except Exception as exc:  # noqa: BLE001 - radon raises broad/parse errors
        logger.warning("radon.cc_visit failed (cyclomatic -> nan): %s", exc)
        return math.nan
    if not blocks:
        return 0.0
    return float(max(b.complexity for b in blocks))


def compute_task_surface_metrics(
    records: Sequence[Dict[str, Any]],
    *,
    fold: int,
    split_family: str,
    ast_normalize_identifiers: bool,
    ast_backend: str = "apted",
    ast_max_nodes: int = 300,
) -> Optional[List[Dict[str, Any]]]:
    """Surface metrics for one task's implementations.

    Returns one row per implementation, or ``None`` (logged) if the task has
    < 3 implementations or is missing a correct (label==1) implementation.
    """
    if len(records) < 3:
        logger.warning(
            "task %s: only %d implementation(s) (< 3); skipping",
            records[0]["task_id"] if records else "?",
            len(records),
        )
        return None
    correct_recs = [r for r in records if int(r["label"]) == 1]
    if not correct_recs:
        logger.warning("task %s: no correct implementation; skipping", records[0]["task_id"])
        return None
    correct_code = correct_recs[0]["answer"]

    Levenshtein = _import_levenshtein()
    error_type = _task_error_type(
        [r["sample_id"] for r in records], [r["label"] for r in records], split_family
    )

    # Mutation_DS stores bare function bodies; wrap them in a synthetic ``def`` so
    # the code parses as a module. ICSE26 implementations are already full modules.
    # All metrics are computed on the wrapped form so they are mutually consistent.
    wrap_body = split_family != "icse26"
    correct_prep = _prepare_source(correct_code, wrap_body)

    rows: List[Dict[str, Any]] = []
    for rec in records:
        code = rec["answer"]
        code_prep = _prepare_source(code, wrap_body)
        is_correct = int(rec["label"]) == 1
        rows.append(
            {
                "task_id": rec["task_id"],
                "sample_id": rec["sample_id"],
                "label": int(rec["label"]),
                "fold": int(fold),
                "error_type": error_type,
                "levenshtein_to_correct": 0.0
                if is_correct
                else float(Levenshtein.distance(code_prep, correct_prep)),
                "ast_dist_to_correct": 0.0
                if is_correct
                else _ast_distance(
                    code_prep,
                    correct_prep,
                    normalize_identifiers=ast_normalize_identifiers,
                    backend=ast_backend,
                    max_nodes=ast_max_nodes,
                ),
                "loc": _loc(code_prep),
                "cyclomatic": _cyclomatic(code_prep),
            }
        )
    return rows


def get_surface_metrics(
    *,
    split_family: str,
    benchmark: str,
    data_root: Path,
    folds: Sequence[int],
    surface_metrics_path: Optional[Path],
    ast_normalize_identifiers: bool,
    ast_backend: str,
    ast_max_nodes: int,
    cache_path: Path,
) -> pd.DataFrame:
    """Per-(task, implementation) surface metrics.

    Columns: ``task_id, sample_id, label, fold, error_type, levenshtein_to_correct,
    ast_dist_to_correct, loc, cyclomatic``. Reuses *surface_metrics_path* if given;
    otherwise computes from each fold's ``test.jsonl`` and writes ``cache_path``.
    Metrics do not depend on layer/fold/model, so each task is computed once.
    """
    if surface_metrics_path is not None:
        logger.info("Loading cached surface metrics from %s", surface_metrics_path)
        df = pd.read_csv(surface_metrics_path)
        df["task_id"] = df["task_id"].astype(str)
        df["sample_id"] = df["sample_id"].astype(str)
        return df

    rows: List[Dict[str, Any]] = []
    seen_tasks = set()
    for fold in folds:
        path = _fold_split_path(
            split_family=split_family, benchmark=benchmark, data_root=data_root, fold=fold
        )
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in load_split(path):
            by_task[rec["task_id"]].append(rec)
        for tid in sorted(by_task):
            if tid in seen_tasks:
                continue  # a task is held out in exactly one fold's test split
            seen_tasks.add(tid)
            task_rows = compute_task_surface_metrics(
                by_task[tid],
                fold=fold,
                split_family=split_family,
                ast_normalize_identifiers=ast_normalize_identifiers,
                ast_backend=ast_backend,
                ast_max_nodes=ast_max_nodes,
            )
            if task_rows is not None:
                rows.extend(task_rows)

    df = pd.DataFrame(rows, columns=list(SURFACE_COLUMNS))
    df["task_id"] = df["task_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Wrote surface metrics (%d rows) to %s", len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def residual_pca_y(
    acts_task: np.ndarray, reading_vec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Residual-PCA y-axis for one task at one layer.

    ``acts_task`` is ``[k, H]`` and ``reading_vec`` is a unit ``[H]`` vector.
    Projects out the reading vector, centers, and runs PCA on the residuals.

    Returns ``(y[k], v1[H], v1_explained_var_ratio)`` where ``v1`` is the top
    residual component and ``y`` are the projections onto it. The sign of
    ``v1`` (hence ``y``) is arbitrary, so only ``r-squared`` downstream is meaningful.
    """
    if acts_task.ndim != 2:
        raise ValueError(f"expected 2D acts_task, got shape={acts_task.shape}")
    k, h = acts_task.shape
    if k < 2:
        raise ValueError("need at least 2 implementations for residual PCA")

    r_hat = np.asarray(reading_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r_hat))
    if r_norm == 0.0:
        raise ValueError("reading vector has zero norm")
    r_hat = r_hat / r_norm  # defensive

    feats = acts_task.astype(np.float64)
    proj = feats @ r_hat
    resid = feats - np.outer(proj, r_hat)  # orthogonal to r_hat
    resid = resid - resid.mean(axis=0, keepdims=True)  # center

    if np.allclose(resid, 0.0) or float(np.std(resid)) < 1e-12:
        return np.zeros(k, dtype=np.float64), np.zeros(h, dtype=np.float64), 0.0

    pca = PCA(n_components=1)
    y = pca.fit_transform(resid)[:, 0]
    v1 = pca.components_[0]
    evr = float(pca.explained_variance_ratio_[0])
    return y.astype(np.float64), v1.astype(np.float64), evr


def pearson_r2(y: np.ndarray, f: np.ndarray) -> Tuple[float, float]:
    """Pearson ``r`` and ``r-squared`` over length-k vectors.

    Returns ``(nan, nan)`` if either vector is constant.
    """
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if y.shape != f.shape:
        raise ValueError("y and f must have the same shape")
    if not np.isfinite(y).all() or not np.isfinite(f).all():
        return math.nan, math.nan
    if np.std(y) < 1e-12 or np.std(f) < 1e-12:
        return math.nan, math.nan
    r = float(np.corrcoef(y, f)[0, 1])
    return r, float(r * r)


def permutation_null_r2(
    y: np.ndarray,
    f: np.ndarray,
    *,
    n_perms: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Permutation null for ``r-squared``.

    For ``k <= 7`` enumerates all ``k!`` permutations of ``f`` against ``y`` exactly.
    For larger ``k`` it samples ``n_perms`` random permutations.

    Returns ``(null_mean_r2, p_value)`` where ``p_value`` is the fraction of
    permutations with ``r-squared >= observed``. Returns ``(nan, nan)`` if the
    observed ``r-squared`` is undefined (constant ``y`` or ``f``).
    """
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    _, observed_r2 = pearson_r2(y, f)
    if math.isnan(observed_r2):
        return math.nan, math.nan

    k = y.shape[0]
    r2_values: List[float] = []
    if k <= 7:
        for perm in permutations(range(k)):
            _, r2 = pearson_r2(y, f[list(perm)])
            r2_values.append(0.0 if math.isnan(r2) else r2)
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        for _ in range(n_perms):
            _, r2 = pearson_r2(y, f[rng.permutation(k)])
            r2_values.append(0.0 if math.isnan(r2) else r2)

    arr = np.asarray(r2_values, dtype=np.float64)
    null_mean = float(arr.mean())
    p_value = float(np.mean(arr >= observed_r2 - 1e-12))
    return null_mean, p_value


# ---------------------------------------------------------------------------
# Stage: correlation analysis (raw long table)
# ---------------------------------------------------------------------------

def _layer_indices(num_layers_total: int, include_embedding: bool) -> List[int]:
    start = 0 if include_embedding else 1
    return list(range(start, num_layers_total))


def run_correlation_analysis(
    *,
    hidden_states_root: Path,
    vectors_root: Path,
    data_root: Path,
    split_family: str,
    benchmark: str,
    model: str,
    folds: Sequence[int],
    method: str,
    fit_framing: str,
    representation: str,
    test_framing: str,
    surface_df: pd.DataFrame,
    include_embedding: bool,
    n_perms: int,
    seed: int,
) -> pd.DataFrame:
    """Iterate folds -> that fold's held-out tasks -> layers -> metrics.

    Uses the fold's reading vector to compute the residual-PCA y-axis and
    correlates it against each surface metric. Returns the raw long-format table
    (one row per ``(dataset, model, fold, layer, task, error_type, metric)``).
    """
    ds_label = dataset_label(split_family)
    rng = np.random.default_rng(seed)

    reading_vectors = load_reading_vectors(
        vectors_root=vectors_root,
        model=model,
        split_family=split_family,
        benchmark=benchmark,
        method=method,
        fit_framing=fit_framing,
        representation=representation,
        folds=folds,
    )

    metrics_by_sample: Dict[str, Dict[str, float]] = {
        row["sample_id"]: {m: row[m] for m in METRICS} for _, row in surface_df.iterrows()
    }
    error_type_by_task: Dict[str, str] = {
        str(row["task_id"]): str(row["error_type"]) for _, row in surface_df.iterrows()
    }

    rows: List[Dict[str, Any]] = []
    for fold in folds:
        bundle_path = _bundle_path_for_fold(
            hidden_states_root=hidden_states_root,
            model=model,
            split_family=split_family,
            benchmark=benchmark,
            fold=fold,
            test_framing=test_framing,
        )
        if not bundle_path.is_file():
            raise FileNotFoundError(f"missing cached test bundle: {bundle_path}")
        bundle = load_bundle(str(bundle_path), representation=representation)
        acts = bundle["acts"]  # [N, L+1, H]
        sample_ids = bundle["sample_ids"]
        task_ids = bundle["task_ids"]
        labels = bundle["labels"]

        layer_vectors = reading_vectors[fold]
        any_vec = next(iter(layer_vectors.values()))
        if any_vec.shape[0] != acts.shape[2]:
            raise ValueError(
                f"fold {fold}: reading vector H={any_vec.shape[0]} != acts H={acts.shape[2]}"
            )

        layers = _layer_indices(acts.shape[1], include_embedding)
        missing = [l for l in layers if l not in layer_vectors]
        if missing:
            raise KeyError(f"fold {fold}: direction vectors missing layer(s) {missing}")

        by_task: Dict[str, List[int]] = defaultdict(list)
        for i, tid in enumerate(task_ids):
            by_task[tid].append(i)

        for tid in sorted(by_task):
            idx = by_task[tid]
            task_labels = [int(labels[i]) for i in idx]
            if len(idx) < 3:
                logger.warning("fold %d task %s: %d impls (< 3); skipping", fold, tid, len(idx))
                continue
            if sum(1 for lab in task_labels if lab == 1) < 1:
                logger.warning("fold %d task %s: no correct implementation; skipping", fold, tid)
                continue

            sids = [sample_ids[i] for i in idx]
            if any(sid not in metrics_by_sample for sid in sids):
                logger.warning(
                    "fold %d task %s: missing surface metrics for some impls; skipping",
                    fold,
                    tid,
                )
                continue

            metric_vectors = {
                m: np.asarray([metrics_by_sample[sid][m] for sid in sids], dtype=np.float64)
                for m in METRICS
            }
            error_type = error_type_by_task.get(tid, "")
            k = len(idx)

            for layer in layers:
                acts_task = acts[idx, layer, :]
                y, _v1, evr = residual_pca_y(acts_task, layer_vectors[layer])
                for metric in METRICS:
                    f = metric_vectors[metric]
                    r, r2 = pearson_r2(y, f)
                    null_mean, p_value = permutation_null_r2(
                        y, f, n_perms=n_perms, rng=rng
                    )
                    rows.append(
                        {
                            "dataset": ds_label,
                            "split_family": split_family,
                            "model": model,
                            "fold": int(fold),
                            "layer": int(layer),
                            "task_id": tid,
                            "error_type": error_type,
                            "metric": metric,
                            "pearson_r": r,
                            "r_squared": r2,
                            "null_r2_mean": null_mean,
                            "p_value": p_value,
                            "v1_explained_var": evr,
                            "n_points": k,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage: aggregation
# ---------------------------------------------------------------------------

def _macro_for_subset(raw_df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    """Two-level macro aggregation for one error-type subset (or 'ALL')."""
    valid = raw_df.dropna(subset=["r_squared"])
    if valid.empty:
        return pd.DataFrame()

    # Level 1: per (fold, layer, metric) mean r-squared over that fold's tasks.
    level1 = (
        valid.groupby(["fold", "layer", "metric"], as_index=False)["r_squared"]
        .mean()
        .rename(columns={"r_squared": "fold_mean_r2"})
    )

    # Level 2: per (layer, metric) macro-average across folds (equal fold weight).
    out_rows: List[Dict[str, Any]] = []
    ds_label = str(raw_df["dataset"].iloc[0])
    split_family = str(raw_df["split_family"].iloc[0])
    model = str(raw_df["model"].iloc[0])
    for (layer, metric), grp in level1.groupby(["layer", "metric"]):
        vals = grp["fold_mean_r2"].to_numpy(dtype=np.float64)
        task_subset = valid[(valid["layer"] == layer) & (valid["metric"] == metric)]
        frac_above = (
            float(np.mean(task_subset["r_squared"].to_numpy() > NULL_BASELINE))
            if len(task_subset)
            else math.nan
        )
        out_rows.append(
            {
                "dataset": ds_label,
                "split_family": split_family,
                "model": model,
                "error_type": error_type,
                "layer": int(layer),
                "metric": metric,
                "macro_mean_r2": float(np.mean(vals)),
                "macro_std_r2": float(np.std(vals, ddof=0)),
                "null_baseline": NULL_BASELINE,
                "n_folds": int(len(vals)),
                "frac_tasks_above_null": frac_above,
            }
        )
    return pd.DataFrame(out_rows)


def aggregate_macro(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Two-level (within-fold then macro-across-folds) aggregation.

    Produces one row per ``(dataset, model, error_type, layer, metric)``. The
    ``error_type == 'ALL'`` rows pool every task; for Mutation_DS each distinct
    error-type subset also gets its own rows.
    """
    if raw_df.empty:
        return pd.DataFrame()
    frames = [_macro_for_subset(raw_df, "ALL")]
    split_family = str(raw_df["split_family"].iloc[0])
    if split_family != "icse26":
        for error_type in sorted(et for et in raw_df["error_type"].unique() if et):
            subset = raw_df[raw_df["error_type"] == error_type]
            frames.append(_macro_for_subset(subset, error_type))
    result = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return result.sort_values(["error_type", "metric", "layer"]).reset_index(drop=True)


def aggregate_selected_layer(
    raw_df: pd.DataFrame, selected_layers: Dict[int, int]
) -> pd.DataFrame:
    """Per-fold r-squared at that fold's validation-selected layer, macro-averaged.

    Each fold contributes its own ``selected_layer`` value; results are then
    macro-averaged across folds. One row per ``(dataset, model, error_type, metric)``.
    """
    if raw_df.empty:
        return pd.DataFrame()

    def _subset_rows(df: pd.DataFrame, error_type: str) -> List[Dict[str, Any]]:
        ds_label = str(df["dataset"].iloc[0])
        split_family = str(df["split_family"].iloc[0])
        model = str(df["model"].iloc[0])
        per_fold = []
        for fold, sel_layer in selected_layers.items():
            sel = df[(df["fold"] == fold) & (df["layer"] == sel_layer)].dropna(
                subset=["r_squared"]
            )
            if sel.empty:
                continue
            fold_means = sel.groupby("metric")["r_squared"].mean()
            per_fold.append((fold, sel_layer, fold_means))
        out: List[Dict[str, Any]] = []
        for metric in METRICS:
            vals = [fm[metric] for _, _, fm in per_fold if metric in fm.index]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out.append(
                {
                    "dataset": ds_label,
                    "split_family": split_family,
                    "model": model,
                    "error_type": error_type,
                    "metric": metric,
                    "macro_mean_r2": float(np.mean(arr)),
                    "macro_std_r2": float(np.std(arr, ddof=0)),
                    "null_baseline": NULL_BASELINE,
                    "n_folds": int(arr.size),
                    "selected_layers": ";".join(
                        f"{fold}:{lay}" for fold, lay, _ in per_fold
                    ),
                }
            )
        return out

    rows = _subset_rows(raw_df, "ALL")
    split_family = str(raw_df["split_family"].iloc[0])
    if split_family != "icse26":
        for error_type in sorted(et for et in raw_df["error_type"].unique() if et):
            subset = raw_df[raw_df["error_type"] == error_type]
            if not subset.empty:
                rows.extend(_subset_rows(subset, error_type))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage: plotting
# ---------------------------------------------------------------------------

def plot_layer_curves(
    summary_df: pd.DataFrame,
    *,
    out_dir: Path,
    dataset_name: str,
    split_family: str,
    model: str,
    error_type: Optional[str] = None,
) -> Optional[Path]:
    """Layer-curve plot of macro-mean r-squared per metric.

    x = layer, y = macro_mean_r2, one line per metric with a +/-std-across-folds
    band and a dashed chance line at 1/3. Writes one PNG and returns its path.
    """
    subset_key = error_type if error_type is not None else "ALL"
    df = summary_df[summary_df["error_type"] == subset_key]
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric in METRICS:
        mdf = df[df["metric"] == metric].sort_values("layer")
        if mdf.empty:
            continue
        layers = mdf["layer"].to_numpy()
        mean = mdf["macro_mean_r2"].to_numpy()
        std = mdf["macro_std_r2"].to_numpy()
        color = METRIC_COLORS[metric]
        ax.plot(layers, mean, marker="o", markersize=3, label=metric, color=color)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(NULL_BASELINE, linestyle="--", color="gray", linewidth=1.2, label="chance (n=4)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("macro-mean $r^2$ (residual-PCA y-axis vs metric)")
    title = f"{dataset_name} — {safe_model_dir(model)}"
    if error_type is not None and error_type != "ALL":
        title += f" — error type: {error_type}"
    ax.set_title(title, fontsize="medium")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small", ncol=2, loc="upper right")

    suffix = "" if subset_key == "ALL" else f"_{subset_key}"
    out_path = out_dir / f"layer_curves{suffix}.png"
    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_output_dir(*, output_root: Path, model: str, split_family: str, benchmark: str) -> Path:
    out_dir = (
        output_root
        / safe_model_dir(model)
        / Path(*viz_output_path_parts(split_family))
        / benchmark
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_pipeline(
    *,
    stage: str,
    hidden_states_root: Path,
    vectors_root: Path,
    results_root: Path,
    data_root: Path,
    split_family: str,
    benchmark: str,
    model: str,
    folds: Sequence[int],
    method: str,
    fit_framing: str,
    representation: str,
    val_framing: str,
    test_framing: str,
    surface_metrics_path: Optional[Path],
    ast_normalize_identifiers: bool,
    ast_backend: str,
    ast_max_nodes: int,
    include_embedding: bool,
    n_perms: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    out_dir = build_output_dir(
        output_root=output_dir, model=model, split_family=split_family, benchmark=benchmark
    )
    surface_cache = out_dir / "surface_metrics.csv"
    raw_path = out_dir / "correlation_raw.csv"
    summary_csv = out_dir / "correlation_summary.csv"
    summary_json = out_dir / "correlation_summary.json"
    selected_csv = out_dir / "correlation_selected_layer.csv"

    written: Dict[str, str] = {}
    ds_label = dataset_label(split_family)

    # ---- Stage: metrics -----------------------------------------------------
    if stage in {"metrics", "correlate", "all"}:
        surface_df = get_surface_metrics(
            split_family=split_family,
            benchmark=benchmark,
            data_root=data_root,
            folds=folds,
            surface_metrics_path=surface_metrics_path,
            ast_normalize_identifiers=ast_normalize_identifiers,
            ast_backend=ast_backend,
            ast_max_nodes=ast_max_nodes,
            cache_path=surface_cache,
        )
        if surface_metrics_path is None:
            written["surface_metrics"] = str(surface_cache)
    else:
        surface_df = None  # type: ignore[assignment]

    if stage == "metrics":
        return {
            "stage": stage,
            "dataset": ds_label,
            "model": model,
            "n_surface_rows": int(len(surface_df)),
            "output_paths": written,
        }

    # ---- Stage: correlate ---------------------------------------------------
    if stage in {"correlate", "all"}:
        raw_df = run_correlation_analysis(
            hidden_states_root=hidden_states_root,
            vectors_root=vectors_root,
            data_root=data_root,
            split_family=split_family,
            benchmark=benchmark,
            model=model,
            folds=folds,
            method=method,
            fit_framing=fit_framing,
            representation=representation,
            test_framing=test_framing,
            surface_df=surface_df,
            include_embedding=include_embedding,
            n_perms=n_perms,
            seed=seed,
        )
        raw_df.to_csv(raw_path, index=False)
        written["correlation_raw"] = str(raw_path)
    else:
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"stage {stage!r} needs {raw_path}; run --stage correlate (or all) first"
            )
        raw_df = pd.read_csv(raw_path)
        raw_df["task_id"] = raw_df["task_id"].astype(str)
        raw_df["error_type"] = raw_df["error_type"].fillna("").astype(str)

    if stage == "correlate":
        return {
            "stage": stage,
            "dataset": ds_label,
            "model": model,
            "n_raw_rows": int(len(raw_df)),
            "folds": list(folds),
            "output_paths": written,
        }

    # ---- Stage: aggregate ---------------------------------------------------
    if stage in {"aggregate", "all"}:
        summary_df = aggregate_macro(raw_df)
        summary_df.to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(
                {
                    "dataset": ds_label,
                    "split_family": split_family,
                    "model": model,
                    "null_baseline": NULL_BASELINE,
                    "rows": summary_df.to_dict(orient="records"),
                },
                indent=2,
            )
        )
        written["correlation_summary_csv"] = str(summary_csv)
        written["correlation_summary_json"] = str(summary_json)

        selected_layers: Dict[int, int] = {}
        for fold in folds:
            try:
                selected_layers[int(fold)] = read_selected_layer(
                    results_root=results_root,
                    model=model,
                    split_family=split_family,
                    benchmark=benchmark,
                    method=method,
                    fit_framing=fit_framing,
                    representation=representation,
                    val_framing=val_framing,
                    test_framing=test_framing,
                    fold=fold,
                )
            except (FileNotFoundError, KeyError) as exc:
                logger.warning("fold %d: could not read selected layer: %s", fold, exc)
        if selected_layers:
            selected_df = aggregate_selected_layer(raw_df, selected_layers)
            selected_df.to_csv(selected_csv, index=False)
            written["correlation_selected_layer"] = str(selected_csv)
        else:
            logger.warning("no selected layers available; skipping selected-layer summary")
            selected_df = pd.DataFrame()
    else:
        if not summary_csv.is_file():
            raise FileNotFoundError(
                f"stage {stage!r} needs {summary_csv}; run --stage aggregate (or all) first"
            )
        summary_df = pd.read_csv(summary_csv)
        summary_df["error_type"] = summary_df["error_type"].fillna("ALL").astype(str)
        selected_df = pd.read_csv(selected_csv) if selected_csv.is_file() else pd.DataFrame()

    if stage == "aggregate":
        return {
            "stage": stage,
            "dataset": ds_label,
            "model": model,
            "n_summary_rows": int(len(summary_df)),
            "n_selected_rows": int(len(selected_df)),
            "output_paths": written,
        }

    # ---- Stage: plot --------------------------------------------------------
    plot_paths: List[str] = []
    main_plot = plot_layer_curves(
        summary_df,
        out_dir=out_dir,
        dataset_name=ds_label,
        split_family=split_family,
        model=model,
        error_type=None,
    )
    if main_plot is not None:
        plot_paths.append(str(main_plot))
    if split_family != "icse26":
        for error_type in sorted(et for et in summary_df["error_type"].unique() if et and et != "ALL"):
            p = plot_layer_curves(
                summary_df,
                out_dir=out_dir,
                dataset_name=ds_label,
                split_family=split_family,
                model=model,
                error_type=error_type,
            )
            if p is not None:
                plot_paths.append(str(p))
    written["plots"] = plot_paths  # type: ignore[assignment]

    return {
        "stage": stage,
        "dataset": ds_label,
        "model": model,
        "split_family": split_family,
        "benchmark": benchmark,
        "folds": list(folds),
        "null_baseline": NULL_BASELINE,
        "n_raw_rows": int(len(raw_df)),
        "n_summary_rows": int(len(summary_df)),
        "output_dir": str(out_dir),
        "output_paths": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate the residual-PCA y-axis against surface code metrics, per layer.",
    )
    parser.add_argument("--benchmark", choices=["bigcodebench", "humaneval"], required=True)
    parser.add_argument(
        "--data-root",
        default="dataset/icse26",
        help="ICSE26 root (dataset/icse26) or Mutation_DS phase2 dir "
        "(dataset/bigcodebench/phase2); used to locate fold JSONL and manifest.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id.")
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Either explicit fold indices (e.g. 0 3) or 'all'.",
    )
    parser.add_argument("--split-family", default="icse26")
    parser.add_argument("--hidden-states-root", default="hidden_states")
    parser.add_argument("--vectors-root", default="vectors")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--method", default="mean_diff")
    parser.add_argument("--fit-framing", default="function")
    parser.add_argument("--representation", default="response_avg")
    parser.add_argument("--val-framing", default="neutral")
    parser.add_argument("--test-framing", default="neutral")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Run a single pipeline stage (each reads/writes on-disk artifacts) or 'all'.",
    )
    parser.add_argument(
        "--surface-metrics-path",
        default=None,
        help="Reuse an existing surface_metrics.csv instead of recomputing.",
    )
    parser.add_argument(
        "--ast-normalize-identifiers",
        action="store_true",
        help="Structure-only AST distance (ignore identifier/constant names). "
        "[TODO] default OFF: identifier renames count as edits.",
    )
    parser.add_argument(
        "--ast-backend",
        choices=["apted", "zss"],
        default="apted",
        help="Tree-edit-distance backend for ast_dist_to_correct (apted is faster).",
    )
    parser.add_argument(
        "--ast-max-nodes",
        type=int,
        default=300,
        help="Skip ast_dist_to_correct (-> nan) when either AST exceeds this many "
        "nodes; bounds super-quadratic tree-edit-distance runtime. 0 disables the cap.",
    )
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Include layer 0 (embedding output) in the analysis (excluded by default).",
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=10000,
        help="Random permutations for the r^2 null when k > 7 (exact k! used for k<=7). "
        "[TODO] default 10000.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="analysis/residual_pca_correlation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = Path(args.data_root)
    folds = parse_fold_selector(
        args.folds,
        split_family=args.split_family,
        data_root=data_root,
        benchmark=args.benchmark,
    )

    summary = run_pipeline(
        stage=args.stage,
        hidden_states_root=Path(args.hidden_states_root),
        vectors_root=Path(args.vectors_root),
        results_root=Path(args.results_root),
        data_root=data_root,
        split_family=args.split_family,
        benchmark=args.benchmark,
        model=args.model,
        folds=folds,
        method=args.method,
        fit_framing=args.fit_framing,
        representation=args.representation,
        val_framing=args.val_framing,
        test_framing=args.test_framing,
        surface_metrics_path=Path(args.surface_metrics_path)
        if args.surface_metrics_path
        else None,
        ast_normalize_identifiers=args.ast_normalize_identifiers,
        ast_backend=args.ast_backend,
        ast_max_nodes=args.ast_max_nodes,
        include_embedding=args.include_embedding,
        n_perms=args.n_perms,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
