"""Representations: how to pool per-token hidden states into one vector per sample.

A :class:`Representation` is used in two situations:

1. **At extraction time**, ``pool(layer_hidden_states, masks)`` reduces a
   single layer's ``[batch, tokens, hidden_dim]`` tensor to ``[batch,
   hidden_dim]``. The extractor calls this for every (batch, layer) pair
   and stacks the results into a :class:`HiddenStateBundle`.

2. **At downstream time**, ``select(bundle)`` retrieves the pre-pooled
   tensor by name from a bundle. This is the cheap default path.

Representations are pure functions of the per-batch tensors; they do not
need access to the model or tokenizer.
"""

from .base import Boundaries, Representation
from .pooling import PromptAvg, PromptLast, ResponseAvg, SequenceLast

__all__ = [
    "Representation",
    "Boundaries",
    "PromptAvg",
    "PromptLast",
    "ResponseAvg",
    "SequenceLast",
]
