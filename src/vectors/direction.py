"""Direction: per-layer vectors produced by a VectorMethod."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch


@dataclass
class Direction:
    """A layer-indexed bundle of correctness directions.

    Attributes
    ----------
    method:
        Name of the :class:`VectorMethod` that produced it
        (e.g. ``"mean_diff"``).
    representation:
        Name of the :class:`Representation` it was fit on
        (e.g. ``"response_avg"``).
    vectors:
        Mapping ``layer_idx -> [hidden_dim]``.
    signs:
        Mapping ``layer_idx -> +1 | -1``. For methods where the raw
        vector already encodes a signed direction (e.g. mean-diff), all
        signs are ``+1``.
    centering:
        Optional mapping ``layer_idx -> [hidden_dim]`` of mean vectors
        subtracted from activations before projection. ``None`` for
        methods that do not center (e.g. mean-diff).
    extras:
        Free-form per-layer artifacts a method may need to preserve
        (e.g. PCA model objects). Not used by :meth:`project`.
    """

    method: str
    representation: str
    vectors: Dict[int, torch.Tensor]
    signs: Dict[int, int]
    centering: Optional[Dict[int, torch.Tensor]] = None
    extras: Dict[int, object] = field(default_factory=dict)

    @property
    def layers(self) -> list:
        return sorted(self.vectors)

    def project(self, acts: torch.Tensor, layer: int) -> torch.Tensor:
        """Project pooled activations onto the direction for a given layer.

        Parameters
        ----------
        acts:
            Either ``[B, hidden_dim]`` (one layer's activations) or
            ``[B, num_layers + 1, hidden_dim]`` (full stack — the layer
            slice is taken internally).
        layer:
            Layer index to project against.

        Returns
        -------
        Tensor of shape ``[B]`` of signed scalar scores.
        """
        if acts.dim() == 3:
            x = acts[:, layer, :]
        elif acts.dim() == 2:
            x = acts
        else:
            raise ValueError(f"acts must be 2D or 3D, got shape {tuple(acts.shape)}")

        vec = self.vectors[layer].to(x.dtype).to(x.device)
        if self.centering is not None and layer in self.centering:
            x = x - self.centering[layer].to(x.dtype).to(x.device)
        return self.signs[layer] * (x @ vec)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": self.method,
                "representation": self.representation,
                "vectors": self.vectors,
                "signs": self.signs,
                "centering": self.centering,
                "extras": self.extras,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "Direction":
        blob = torch.load(Path(path), weights_only=False)
        return cls(**blob)
