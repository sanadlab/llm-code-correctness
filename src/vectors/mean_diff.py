"""Mean-of-differences vector method.

For each task, take the mean of correct-sample activations minus the
mean of incorrect-sample activations. Average those per-task differences
across tasks to get one direction per layer. The resulting vector is
already signed (positive points toward "correct").
"""

from collections import defaultdict
from typing import List

import torch

from .base import VectorMethod
from .direction import Direction


class MeanDiff(VectorMethod):
    name = "mean_diff"

    def fit(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        task_ids: List[str],
    ) -> Direction:
        # acts: [N, L+1, H]
        num_layers = acts.shape[1]

        task_samples: dict = defaultdict(list)
        for i, tid in enumerate(task_ids):
            task_samples[tid].append(i)

        task_diffs = []
        for tid in sorted(task_samples):
            indices = task_samples[tid]
            pos_idx = [i for i in indices if int(labels[i]) == 1]
            neg_idx = [i for i in indices if int(labels[i]) == 0]
            if not pos_idx or not neg_idx:
                continue
            pos_mean = acts[pos_idx].mean(dim=0)  # [L+1, H]
            neg_mean = acts[neg_idx].mean(dim=0)
            task_diffs.append(pos_mean - neg_mean)

        if not task_diffs:
            raise ValueError("MeanDiff.fit: no task had both correct and incorrect samples")

        diff = torch.stack(task_diffs, dim=0).mean(dim=0)  # [L+1, H]

        vectors = {layer: diff[layer].clone() for layer in range(num_layers)}
        signs = {layer: 1 for layer in range(num_layers)}

        return Direction(
            method=self.name,
            representation="",  # filled in by caller (depends on which pool was used)
            vectors=vectors,
            signs=signs,
            centering=None,
        )
