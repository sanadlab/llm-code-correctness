"""Tests that mutant body extraction preserves original formatting."""

import ast

from src.phase1.mutate import _extract_body_fragment, _mutate_task
from src.phase1.operators import get_operators


def test_sd_preserves_quotes_comments_and_blank_lines():
  canonical = (
      '    words = re.findall(r"\\b\\w+\\b", text)\n'
      "    word_freq = Counter(words)\n"
      "\n"
      "    # _, ax = plt.subplots()\n"
      "    ax = word_freq.plot(kind=\"bar\")\n"
      "    ax.set_title(\"Words\")\n"
      "    return word_freq, ax\n"
  )
  task = {
      "task_id": "test/format",
      "entry_point": "task_func",
      "complete_prompt": (
          "import re\nfrom collections import Counter\n\n"
          "def task_func(text):\n"
          '    """Count words."""\n'
      ),
      "canonical_solution": canonical,
  }
  operators = get_operators(["SD"])
  mutants = _mutate_task(task, operators, log=False)
  assert mutants, "expected at least one SD mutant"

  # SD site 2 deletes the ax = word_freq.plot(...) assignment.
  body = next(m["code"] for m in mutants if m["site_index"] == 2)

  assert 'r"\\b\\w+\\b"' in body
  assert "# _, ax = plt.subplots()" in body
  assert "\n\n    #" in body
  assert 'kind="bar"' not in body
  assert 'set_title("Words")' in body
  assert "return word_freq, ax" in body
  assert "return (word_freq, ax)" not in body


def test_modified_statement_only_reformats_changed_line():
  """Unchanged lines stay verbatim; only the mutated statement is unparsed."""
  canonical = (
      '    a = "left"\n'
      "\n"
      "    # middle comment\n"
      "    return 1 + 2\n"
  )
  task = {
      "task_id": "test/modify",
      "entry_point": "f",
      "complete_prompt": "def f():\n",
      "canonical_solution": canonical,
  }
  operators = get_operators(["AOD"])
  mutants = _mutate_task(task, operators, log=False)
  assert mutants

  for m in mutants:
      body = m["code"]
      assert 'a = "left"' in body
      assert "# middle comment" in body
      assert "\n\n    #" in body
      # Only the return line should be reformatted by unparse.
      assert "return" in body


def test_sd_diff_against_canonical_is_minimal():
  """Canonical vs mutant should differ only by the deleted statement."""
  canonical = (
      '    words = re.findall(r"\\b\\w+\\b", text)\n'
      "    word_freq = Counter(words)\n"
      "\n"
      "    # _, ax = plt.subplots()\n"
      "    ax = word_freq.plot(kind=\"bar\")\n"
      "    ax.set_title(\"Words\")\n"
      "    return word_freq, ax\n"
  )
  task = {
      "task_id": "test/diff",
      "entry_point": "task_func",
      "complete_prompt": (
          "import re\nfrom collections import Counter\n\n"
          "def task_func(text):\n"
      ),
      "canonical_solution": canonical,
  }
  body = next(
      m["code"]
      for m in _mutate_task(task, get_operators(["SD"]), log=False)
      if m["site_index"] == 2
  )
  # Removing the deleted plot line from canonical should match mutant body.
  expected = canonical.replace('    ax = word_freq.plot(kind="bar")\n', "")
  assert body == expected


def test_extract_body_fragment_preserves_unchanged_lines():
  full_source = (
      "def f():\n"
      '    x = "hello"\n'
      "\n"
      "    # keep me\n"
      "    return x\n"
  )
  orig_tree = ast.parse(full_source)
  mut_tree = ast.parse(
      "def f():\n"
      '    x = "hello"\n'
      "\n"
      "    # keep me\n"
      "    return x\n"
  )
  body = _extract_body_fragment(full_source, orig_tree, mut_tree, "f")
  assert body == '    x = "hello"\n\n    # keep me\n    return x\n'
