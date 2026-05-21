"""Neutral framing: stimulus only, no bias.

The question goes in the user turn; the answer goes in the assistant
turn. Hidden states of interest typically come from inside the answer.
"""

from .base import DEFAULT_ANSWER_LABEL, DEFAULT_QUESTION_LABEL, Framing, Prompt


class Neutral(Framing):
    name = "neutral"

    def __init__(
        self,
        question_label: str = DEFAULT_QUESTION_LABEL,
        answer_label: str = DEFAULT_ANSWER_LABEL,
    ):
        self.question_label = question_label
        self.answer_label = answer_label

    def build(self, question: str, answer: str) -> Prompt:
        return Prompt(
            user_text=f"{self.question_label}:\n{question}",
            assistant_text=answer,
            assistant_is_response=True,
        )
