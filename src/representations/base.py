"""Representation ABC + the Boundaries struct it consumes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch

from src.hidden_states import HiddenStateBundle


@dataclass
class Boundaries:
    """Per-sample token boundaries within a right-padded batch.

    Two integers per sample describe everything a pooling strategy
    needs to know:

    - ``prompt_end[i]``: number of tokens in the prompt portion
      (exclusive end). Equivalently, the index where the response
      starts.
    - ``sequence_end[i]``: index of the last non-pad token (inclusive).

    When the framing has no response portion (e.g. Concept), the
    response is the empty slice ``[prompt_end:sequence_end + 1)``, i.e.
    ``sequence_end + 1 == prompt_end``.

    Both tensors are 1-D ``long`` and live on the same device as the
    hidden states.
    """

    prompt_end: torch.Tensor
    sequence_end: torch.Tensor


class Representation(ABC):
    """Abstract base class for hidden-state pooling strategies."""

    #: Short identifier used as the key under ``HiddenStateBundle.acts``
    #: and in result paths (e.g. ``"response_avg"``).
    name: str

    #: Alternate keys that may appear in cached ``bundle.pt`` files.
    _ACT_ALIASES: Dict[str, tuple] = {
        "sequence_last": ("seq_last",),
        "seq_last": ("sequence_last",),
    }

    @abstractmethod
    def pool(
        self,
        layer_hidden_states: torch.Tensor,
        boundaries: Boundaries,
    ) -> torch.Tensor:
        """Pool one layer's hidden states for a batch.

        Parameters
        ----------
        layer_hidden_states:
            Tensor of shape ``[batch, tokens, hidden_dim]``.
        boundaries:
            Per-sample slice endpoints (see :class:`Boundaries`).

        Returns
        -------
        Tensor of shape ``[batch, hidden_dim]``.
        """
        ...

    def select(self, bundle: HiddenStateBundle) -> torch.Tensor:
        """Fetch the pre-pooled activations for this representation."""
        for key in (self.name, *self._ACT_ALIASES.get(self.name, ())):
            if key in bundle.acts:
                return bundle.acts[key]
        available = ", ".join(sorted(bundle.acts))
        raise KeyError(
            f"representation {self.name!r} not found in bundle.pt "
            f"(available acts keys: {available})"
        )
