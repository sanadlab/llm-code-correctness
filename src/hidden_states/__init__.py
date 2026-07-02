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
        tensor of shape ``[N, num_layers + 1, hidden_dim]``. On disk,
        :meth:`save` always persists ``acts`` as ``float32``.
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
                "acts": {name: tensor.float() for name, tensor in self.acts.items()},
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


from .cache import (
    SPLIT_TO_JSONL,
    MUTANTS_BENCHMARKS,
    MUTANTS_CONFOUNDS_STRATEGIES,
    MUTANTS_STRATEGIES,
    MUTANTS_TEST_STRATEGY,
    VerificationError,
    VerifyConfig,
    build_bundle_metadata,
    bundle_is_complete,
    bundle_output_dir,
    cache_path_distribution,
    fold_dir_name,
    framing_dispatcher,
    framing_uses_chat_template,
    git_commit,
    is_mutants_confounds_split_family,
    is_mutants_split_family,
    is_mutants_variant_split_family,
    mutants_confounds_split_family,
    mutants_confounds_test_split_family,
    mutants_split_family,
    mutants_test_split_family,
    safe_model_dir,
    verify_standard_fold,
    write_bundle_metadata,
)


__all__ = [
    "HiddenStateBundle",
    "SPLIT_TO_JSONL",
    "MUTANTS_BENCHMARKS",
    "MUTANTS_CONFOUNDS_STRATEGIES",
    "MUTANTS_STRATEGIES",
    "MUTANTS_TEST_STRATEGY",
    "VerificationError",
    "VerifyConfig",
    "safe_model_dir",
    "git_commit",
    "fold_dir_name",
    "framing_uses_chat_template",
    "framing_dispatcher",
    "bundle_output_dir",
    "cache_path_distribution",
    "is_mutants_confounds_split_family",
    "is_mutants_split_family",
    "is_mutants_variant_split_family",
    "mutants_confounds_split_family",
    "mutants_confounds_test_split_family",
    "mutants_split_family",
    "mutants_test_split_family",
    "bundle_is_complete",
    "build_bundle_metadata",
    "write_bundle_metadata",
    "verify_standard_fold",
]
