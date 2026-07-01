"""PCA-on-contrasts vector method.

Paper-faithful replication of the LAT / `repe` `PCARepReader` pipeline for
standard splits (one positive + one negative per task_id).

When a task_id has multiple positives and negatives (e.g. mutant ``paired``
fit rows, each jsonl line → one canonical + one mutant), each matched pair is
treated as its own contrast: build signed difference vectors and fit PCA on
the stack of differences.

For each contrast, simulate the paper's ``aligned_shuffle(seed=contrast_idx)``
to randomize which of (pos, neg) comes first. Sign is determined by per-contrast
argmin/argmax voting on raw projections.

We also skip layers 0 (embedding) and 1, matching the paper's
``hidden_layers = list(range(-1, -num_hidden_layers, -1))``.
"""

import random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch

from .base import VectorMethod
from .direction import Direction


# Layers to omit when fitting / using PCA directions. Matches the paper's
# `hidden_layers = range(-1, -num_hidden_layers, -1)` which excludes both
# the embedding (index 0) and the first transformer layer's output (index 1)
# of the HuggingFace `output_hidden_states` tuple.
SKIPPED_LAYERS = (0, 1)


def _aligned_shuffle_keeps_order(seed: int) -> bool:
    """Replicate `loaders.aligned_shuffle` for a 2-element list.

    The paper does `random.seed(seed); random.shuffle([(t0, l0), (t1, l1)])`.
    For n=2, `random.shuffle` calls `randrange(2)` once and swaps if it
    returns 0. This helper returns True iff the original order is preserved.
    """
    rng = random.Random(seed)
    return rng.randrange(2) == 1


def _fit_layer_direction(differences: np.ndarray):
    """Fit a 1-component PCA. Returns (direction, mean, pca_model)."""
    from sklearn.decomposition import PCA as SKPCA

    mean = differences.mean(axis=0, keepdims=True)
    centered = differences - mean
    pca = SKPCA(n_components=1, whiten=False)
    pca.fit(centered)
    return pca.components_[0], mean[0], pca


def _determine_sign(
    all_acts: np.ndarray,
    labels: np.ndarray,
    task_ids_indices,  # list of (task_idx -> list of row indices)
    direction: np.ndarray,
    mean: np.ndarray,
) -> int:
    """Sign by per-task argmin/argmax voting on raw projections.

    Replicates `repe.rep_readers.PCARepReader.get_signs`
    (representation-engineering/repe/rep_readers.py:155-181): for each task,
    take projections of its candidates after recentering, and count whether
    the correct candidate's score is the per-task max or the per-task min.
    sign = sign(p_max - p_min); ties default to +1.

    A mean-of-projections comparison (what we used before) can flip sign on
    layers where a few high-magnitude tasks vote against the majority,
    losing accuracy at borderline layers.
    """
    recentered = all_acts - mean
    projections = recentered @ direction  # [N]
    n_max_correct = 0
    n_min_correct = 0
    for indices in task_ids_indices:
        proj = projections[indices]
        lab = labels[indices]
        # locate the correct index (label==1) within this task
        correct_local = np.where(lab == 1)[0]
        if len(correct_local) == 0:
            continue
        correct_idx = correct_local[0]
        correct_score = proj[correct_idx]
        if correct_score == proj.max():
            n_max_correct += 1
        if correct_score == proj.min():
            n_min_correct += 1
    diff = n_max_correct - n_min_correct
    if diff == 0:
        return 1
    return 1 if diff > 0 else -1


def _build_contrast_pairs(
    task_groups: "OrderedDict[str, dict]",
) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """Build (first, second) index lists and per-contrast sign-vote groups.

    Each task contributes ``min(n_pos, n_neg)`` contrasts, pairing positives
    and negatives by order of appearance (sorted row indices). Multiple jsonl
    rows sharing a ``task_id`` (mutant ``paired`` fit) therefore yield one PCA
    contrast per row.
    """
    first_indices: List[int] = []
    second_indices: List[int] = []
    sign_vote_groups: List[np.ndarray] = []
    contrast_idx = 0

    for tid, g in task_groups.items():
        pos_indices = sorted(g["pos"])
        neg_indices = sorted(g["neg"])
        if not pos_indices or not neg_indices:
            raise ValueError(
                f"PCA.fit: task {tid!r} must have at least one positive and "
                f"one negative sample (got pos={len(pos_indices)}, "
                f"neg={len(neg_indices)})"
            )
        if len(pos_indices) != len(neg_indices):
            raise ValueError(
                f"PCA.fit: task {tid!r} has unequal pos/neg counts "
                f"(pos={len(pos_indices)}, neg={len(neg_indices)}); "
                "pairing requires matched contrasts"
            )
        for pos_i, neg_i in zip(pos_indices, neg_indices):
            if _aligned_shuffle_keeps_order(contrast_idx):
                first_indices.append(pos_i)
                second_indices.append(neg_i)
            else:
                first_indices.append(neg_i)
                second_indices.append(pos_i)
            sign_vote_groups.append(np.array([pos_i, neg_i]))
            contrast_idx += 1

    return first_indices, second_indices, sign_vote_groups


class PCA(VectorMethod):
    name = "pca"

    def fit(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        task_ids: List[str],
    ) -> Direction:
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)

        # Group samples by task, preserving order of first appearance, so
        # task_idx matches the enumeration in the paper's fit loop.
        task_groups: "OrderedDict[str, dict]" = OrderedDict()
        for i, tid in enumerate(task_ids):
            g = task_groups.setdefault(tid, {"pos": [], "neg": [], "all": []})
            g["all"].append(i)
            g["pos" if int(labels_np[i]) == 1 else "neg"].append(i)

        first_indices, second_indices, sign_vote_groups = _build_contrast_pairs(
            task_groups
        )
        first_indices_np = np.array(first_indices)
        second_indices_np = np.array(second_indices)

        if len(first_indices_np) < 2:
            raise ValueError("PCA.fit: need at least 2 (pos, neg) contrasts")

        num_layers = acts.shape[1]
        vectors, signs, centering, extras = {}, {}, {}, {}

        for layer in range(num_layers):
            if layer in SKIPPED_LAYERS:
                continue

            layer_acts = acts[:, layer, :].float().numpy()
            differences = layer_acts[first_indices_np] - layer_acts[second_indices_np]

            if differences.var(axis=0).sum() < 1e-12:
                continue  # dead layer; skip

            direction, mean, pca_model = _fit_layer_direction(differences)
            sign = _determine_sign(
                layer_acts, labels_np, sign_vote_groups, direction, mean
            )

            vectors[layer] = torch.from_numpy(direction).float()
            signs[layer] = int(sign)
            centering[layer] = torch.from_numpy(mean).float()
            extras[layer] = pca_model

        if not vectors:
            raise ValueError("PCA.fit: all layers were zero-variance")

        return Direction(
            method=self.name,
            representation="",  # filled in by caller
            vectors=vectors,
            signs=signs,
            centering=centering,
            extras=extras,
        )
