"""Raw framing: no chat template, paper-exact stimulus.

The original repe paper (`evaluators.py::pair_data` + `LATEvaluator`) fits
the probe on a plain string of the form::

    Task:
    <task>
    Code:
    ```python
    <code>
    ```

with **no** chat-template role markers (no ``<|im_start|>user``, no
``[INST]``). Hidden states are read at the last token of this raw string.

This framing reproduces that input verbatim. It is the appropriate fit-time
framing when targeting paper replication.
"""

from .base import (
    DEFAULT_ANSWER_LABEL,
    DEFAULT_QUESTION_LABEL,
    Framing,
    Prompt,
    format_stimulus,
)


class Raw(Framing):
    name = "raw"

    def __init__(
        self,
        question_label: str = DEFAULT_QUESTION_LABEL,
        answer_label: str = DEFAULT_ANSWER_LABEL,
        fence_answer: bool = False,
        fence_language: str = "python",
    ):
        self.question_label = question_label
        self.answer_label = answer_label
        self.fence_answer = fence_answer
        self.fence_language = fence_language

    def build(self, question: str, answer: str) -> Prompt:
        if self.fence_answer:
            answer = f"```{self.fence_language}\n{answer.strip()}\n```"
        text = format_stimulus(
            question.strip(), answer, self.question_label, self.answer_label
        )
        # Entire text is the prompt; no response portion.
        return Prompt(
            user_text=text,
            assistant_text="",
            assistant_is_response=False,
            use_chat_template=False,
        )
