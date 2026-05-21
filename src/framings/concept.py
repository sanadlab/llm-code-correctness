"""Concept framing: ask the model to evaluate a property of a stimulus.

The template is::

    Consider the amount of {concept} in the following:
    {stimulus}
    The amount of {concept} is␣

where ``{stimulus}`` is a ``Question/Answer`` block (defaults named for
code: ``Task`` / ``Code``).

The model is not asked to produce a continuation — we just want the
hidden state at the end of the stem. The stem ``"The amount of
{concept} is "`` is placed in the assistant turn so that the chat
template's role markers end up *before* the stem, leaving the trailing
space as the final token.
"""

from .base import (
    DEFAULT_ANSWER_LABEL,
    DEFAULT_QUESTION_LABEL,
    Framing,
    Prompt,
    format_stimulus,
)


class Concept(Framing):
    name = "concept"

    def __init__(
        self,
        concept: str = "correctness",
        question_label: str = DEFAULT_QUESTION_LABEL,
        answer_label: str = DEFAULT_ANSWER_LABEL,
        target_field_name: str = "answer",
        fence_answer: bool = False,
        fence_language: str = "python",
        # If provided, the chat-role markers are baked into the string and the
        # extractor is told to bypass `apply_chat_template`. This matches the
        # original repe paper, which built the role markers by raw string
        # concatenation (no system prompt, no template-side whitespace
        # rewriting). Format strings come straight from the paper's
        # `token_tags` (e.g. ``"<|im_start|>user\n"``).
        user_start=None,
        user_end=None,
        assistant_start=None,
    ):
        self.concept = concept
        self.question_label = question_label
        self.answer_label = answer_label
        self.target_field_name = target_field_name
        self.fence_answer = fence_answer
        self.fence_language = fence_language
        # If any tag kwarg is supplied (even as empty string) we activate the
        # paper-exact mode: bake markers in, skip apply_chat_template. This
        # supports BASE models for which paper sets all tags to "".
        self.paper_exact = any(t is not None for t in (user_start, user_end, assistant_start))
        self.user_start = user_start or ""
        self.user_end = user_end or ""
        self.assistant_start = assistant_start or ""

    def build(self, question: str, answer: str) -> Prompt:
        if self.fence_answer:
            answer = f"```{self.fence_language}\n{answer.strip()}\n```"
        stimulus = format_stimulus(
            question.strip(), answer, self.question_label, self.answer_label
        )
        # Paper template (repe `evaluators.py::LATEvaluator`):
        #   Consider the amount of <concept> in the following <field>.\n\n<stimulus>\n
        # Assistant stem:
        #   The amount of <concept> in the <field> is\u0020   (note trailing space)
        user_body = (
            f"Consider the amount of {self.concept} in the following "
            f"{self.target_field_name}.\n\n{stimulus}\n"
        )
        assistant_body = (
            f"The amount of {self.concept} in the {self.target_field_name} is "
        )
        if self.paper_exact:
            # Paper-exact mode: bake chat markers in, skip apply_chat_template.
            full = (
                f"{self.user_start}{user_body}{self.user_end}"
                f"{self.assistant_start}{assistant_body}"
            )
            return Prompt(
                user_text=full,
                assistant_text="",
                assistant_is_response=False,
                use_chat_template=False,
            )
        return Prompt(
            user_text=user_body,
            assistant_text=assistant_body,
            assistant_is_response=False,
        )
