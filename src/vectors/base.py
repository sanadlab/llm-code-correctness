"""VectorMethod ABC."""

from abc import ABC, abstractmethod
from typing import List

import torch

from .direction import Direction


class VectorMethod(ABC):
    """Abstract base class for direction-fitting strategies."""

    #: Short identifier used in paths and metadata (e.g. ``"mean_diff"``).
    name: str

    @abstractmethod
    def fit(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        task_ids: List[str],
    ) -> Direction:
        """Fit a per-layer correctness direction.

        Parameters
        ----------
        acts:
            Pooled activations of shape ``[N, num_layers + 1, hidden_dim]``.
        labels:
            Long tensor of shape ``[N]`` with values in ``{0, 1}``
            (1 = correct, 0 = incorrect).
        task_ids:
            Task identifier for each of the N samples. Used to compute
            per-task statistics so that one task does not dominate the fit.
        """
        ...
