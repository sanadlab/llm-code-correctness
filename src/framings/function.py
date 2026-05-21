"""Function framing: an explicit instruction biases the assistant turn.

The instruction (baked in at construction) tells the model to produce
e.g. a *correct* or *incorrect* answer. The caller decides which one to
use based on the sample's label, by instantiating two ``Function``
objects per instruction pair and dispatching accordingly.

Fit-only: validate/test should use Neutral or Concept, since Function
leaks the label.
"""

from .base import DEFAULT_ANSWER_LABEL, DEFAULT_QUESTION_LABEL, Framing, Prompt


class Function(Framing):
    name = "function"

    def __init__(
        self,
        instruction: str,
        question_label: str = DEFAULT_QUESTION_LABEL,
        answer_label: str = DEFAULT_ANSWER_LABEL,
    ):
        self.instruction = instruction
        self.question_label = question_label
        self.answer_label = answer_label

    def build(self, question: str, answer: str) -> Prompt:
        user = (
            f"{self.instruction}\n\n"
            f"{self.question_label}:\n{question}"
        )
        return Prompt(
            user_text=user,
            assistant_text=answer,
            assistant_is_response=True,
        )
