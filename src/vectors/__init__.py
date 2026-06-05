"""Vector methods: fit a per-layer direction from pooled activations.

A :class:`VectorMethod` consumes pooled activations of shape
``[num_samples, num_layers + 1, hidden_dim]`` plus per-sample labels and
task ids, and produces a :class:`Direction`: a layer-indexed bundle of
vectors usable for projection (scoring) or steering interventions.
"""

from .base import VectorMethod
from .direction import Direction
from .mean_diff import MeanDiff
from .pca import PCA
from .projection import project_all_layers

__all__ = ["VectorMethod", "Direction", "MeanDiff", "PCA", "project_all_layers"]
