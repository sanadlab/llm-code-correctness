"""Concrete pooling strategies.

Each strategy slices ``layer_hs[i]`` using the per-sample indices in
:class:`Boundaries` and pools the slice. Averages use a small Python
loop over the batch — extraction batches are small enough that this is
not a hot path.
"""

import torch

from .base import Boundaries, Representation


def _stack_or_zero(rows, dim: int, dtype, device):
    """Stack a list of [hidden_dim] tensors; substitute zeros for ``None``."""
    zero = torch.zeros(dim, dtype=dtype, device=device)
    return torch.stack([r if r is not None else zero for r in rows], dim=0)


class PromptAvg(Representation):
    name = "prompt_avg"

    def pool(self, layer_hs: torch.Tensor, b: Boundaries) -> torch.Tensor:
        rows = []
        for i in range(layer_hs.shape[0]):
            end = int(b.prompt_end[i])
            rows.append(layer_hs[i, :end].mean(dim=0) if end > 0 else None)
        return _stack_or_zero(rows, layer_hs.shape[-1], layer_hs.dtype, layer_hs.device)


class ResponseAvg(Representation):
    name = "response_avg"

    def pool(self, layer_hs: torch.Tensor, b: Boundaries) -> torch.Tensor:
        rows = []
        for i in range(layer_hs.shape[0]):
            start = int(b.prompt_end[i])
            end = int(b.sequence_end[i]) + 1
            rows.append(layer_hs[i, start:end].mean(dim=0) if end > start else None)
        return _stack_or_zero(rows, layer_hs.shape[-1], layer_hs.dtype, layer_hs.device)


class PromptLast(Representation):
    name = "prompt_last"

    def pool(self, layer_hs: torch.Tensor, b: Boundaries) -> torch.Tensor:
        batch_idx = torch.arange(layer_hs.shape[0], device=layer_hs.device)
        return layer_hs[batch_idx, b.prompt_end - 1, :]


class SequenceLast(Representation):
    name = "sequence_last"

    def pool(self, layer_hs: torch.Tensor, b: Boundaries) -> torch.Tensor:
        batch_idx = torch.arange(layer_hs.shape[0], device=layer_hs.device)
        return layer_hs[batch_idx, b.sequence_end, :]
