"""Hidden-state bundle.

A small container holding the per-sample, per-layer activations that have
already been pooled by :class:`Representation` objects, alongside the
metadata needed to compute per-task statistics downstream.

Representations are the things that *produce* these activations from raw
per-token hidden states during extraction; downstream consumers
(:class:`~src.vectors.VectorMethod`, scoring) only need the pooled
result, keyed by representation name.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class HiddenStateBundle:
    """Cache of pooled activations for one split (e.g. one fold's fit).

    Attributes
    ----------
    sample_ids, task_ids, labels:
        Aligned per-sample metadata. ``labels`` is a long tensor of shape
        ``[N]``.
    acts:
        Mapping from representation name (e.g. ``"response_avg"``) to a
        tensor of shape ``[N, num_layers + 1, hidden_dim]``.
    """

    sample_ids: List[str]
    task_ids: List[str]
    labels: torch.Tensor
    acts: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, rep_name: str) -> torch.Tensor:
        return self.acts[rep_name]

    def __contains__(self, rep_name: str) -> bool:
        return rep_name in self.acts

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_ids": self.sample_ids,
                "task_ids": self.task_ids,
                "labels": self.labels,
                "acts": self.acts,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "HiddenStateBundle":
        blob = torch.load(Path(path), weights_only=False)
        return cls(
            sample_ids=blob["sample_ids"],
            task_ids=blob["task_ids"],
            labels=blob["labels"],
            acts=blob["acts"],
        )
