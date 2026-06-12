"""Project pooled activations onto per-layer direction vectors."""

from __future__ import annotations

from typing import List, Tuple

import torch

from src.vectors.direction import Direction


def project_all_layers(
    direction: Direction, acts: torch.Tensor
) -> Tuple[List[int], torch.Tensor]:
    """Project ``acts`` ([N, L+1, H]) onto ``direction`` at every layer.

    Returns ``(layers, scores)`` where ``scores`` has shape ``[N, len(layers)]``.
    """
    layers = direction.layers
    cols = [direction.project(acts, layer).cpu() for layer in layers]
    return layers, torch.stack(cols, dim=1)
