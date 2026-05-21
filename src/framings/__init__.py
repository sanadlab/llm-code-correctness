"""Framings: how to turn (task, code) into a model input.

A `Framing` knows how to build the user/assistant turns that go to the
tokenizer's chat template, and where the prompt/response boundary lies.

The four axes of the system (framing, representation, vector method, step)
compose freely. A framing only knows about (task, code); it does not know
about labels, hidden states, or fitting.
"""

from .base import Framing, Prompt
from .neutral import Neutral
from .concept import Concept
from .function import Function

__all__ = ["Framing", "Prompt", "Neutral", "Concept", "Function"]
