"""Framing ABC + the Prompt struct it produces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


#: Default labels for the two-block stimulus. Generic on purpose so that
#: the same framings apply to any (input, output) domain. Individual
#: framings can override them via constructor arguments — usually
#: populated from a config (e.g. ``"Task"`` / ``"Code"`` for code).
DEFAULT_QUESTION_LABEL = "Question"
DEFAULT_ANSWER_LABEL = "Answer"


@dataclass
class Prompt:
    """A model input ready to be passed through a chat template.

    Attributes
    ----------
    user_text:
        Content of the user turn.
    assistant_text:
        Content of the assistant turn. May be empty.
    assistant_is_response:
        Whether ``assistant_text`` is "the thing being evaluated" (e.g.
        candidate code) or just a stylistic stem ending the prompt
        (e.g. ``"The amount of correctness is "``).

        - ``True``  : prompt ends at the start of ``assistant_text``;
                      ``assistant_text`` forms the response portion.
                      Used by Neutral, Function, and Concept framings.

        - ``False`` : the whole templated text is the prompt;
                      the response portion is empty unless
                      ``assistant_text`` is non-empty and excluded from
                      ``prompt_prefix`` by other means.

        Representations like ``response_avg`` are zero when the
        response portion is empty.
    """

    user_text: str
    assistant_text: str = ""
    assistant_is_response: bool = True
    use_chat_template: bool = True
    """If False, the extractor concatenates `user_text + assistant_text` verbatim
    and feeds that raw string to the tokenizer — no chat-template role markers.
    Used to replicate the original repe paper, whose fit/eval stimuli were
    plain strings (e.g. ``Task:\n…\nCode:\n```python\n…\n``` ``), never
    wrapped in a model's chat template."""


def format_stimulus(
    question: str,
    answer: str,
    question_label: str = DEFAULT_QUESTION_LABEL,
    answer_label: str = DEFAULT_ANSWER_LABEL,
) -> str:
    """Format a (question, answer) pair as a two-block stimulus."""
    return f"{question_label}:\n{question}\n{answer_label}:\n{answer}"


class Framing(ABC):
    """Abstract base class for prompt construction strategies."""

    #: Short identifier used in paths and metadata (e.g. ``"neutral"``).
    name: str

    @abstractmethod
    def build(self, question: str, answer: str) -> Prompt:
        """Build a :class:`Prompt` from a (question, answer) pair.

        The arguments are named generically because the same framings
        apply to any (input, output) domain — code (task, code), MCQA
        (question, answer), translation (sentence, translation), etc.
        """
        ...
