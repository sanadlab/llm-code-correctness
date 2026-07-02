import ast

from src.phase1.operators import ROR, COR, COD


def _parse(source: str):
    return ast.parse(source)


def _assert_all_parse(mutations):
    for m in mutations:
        ast.parse(ast.unparse(m.tree))


# -------------------------
# ROR
# -------------------------

def test_ror_replaces_less_than():
    source = """
def task_func(a, b):
    if a < b:
        return 1
"""

    muts = ROR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 5
    assert all(m.operator_code == "ROR" for m in muts)

    replacements = {m.site.replacement for m in muts}

    assert "<" not in replacements

    _assert_all_parse(muts)


def test_ror_handles_nested_if():
    source = """
def task_func(a, b):
    if True:
        if a > b:
            return 1
"""

    muts = ROR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) > 0

    _assert_all_parse(muts)


def test_ror_handles_chained_comparison():
    source = """
def task_func(x):
    if 0 < x < 10:
        return 1
"""

    muts = ROR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 10

    _assert_all_parse(muts)


# -------------------------
# COR
# -------------------------

def test_cor_replaces_and_with_or():
    source = """
def task_func(a, b):
    if a and b:
        return 1
"""

    muts = COR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 1

    mutated = ast.unparse(muts[0].tree)

    assert "a or b" in mutated

    _assert_all_parse(muts)


def test_cor_replaces_or_with_and():
    source = """
def task_func(a, b):
    if a or b:
        return 1
"""

    muts = COR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 1

    mutated = ast.unparse(muts[0].tree)

    assert "a and b" in mutated

    _assert_all_parse(muts)


def test_cor_handles_nested_boolean_expression():
    source = """
def task_func(a, b, c):
    if (a and b) or c:
        return 1
"""

    muts = COR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) >= 2

    _assert_all_parse(muts)


# -------------------------
# COD
# -------------------------

def test_cod_deletes_each_clause_from_and():
    source = """
def task_func(a, b, c):
    if a and b and c:
        return 1
"""

    muts = COD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 3

    _assert_all_parse(muts)


def test_cod_deletes_each_clause_from_or():
    source = """
def task_func(a, b, c):
    if a or b or c:
        return 1
"""

    muts = COD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 3

    _assert_all_parse(muts)


def test_cod_two_clause_expression():
    source = """
def task_func(a, b):
    if a and b:
        return 1
"""

    muts = COD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 2

    _assert_all_parse(muts)


def test_cod_nested_boolean_expression():
    source = """
def task_func(a, b, c):
    if (a and b) or c:
        return 1
"""

    muts = COD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) > 0

    _assert_all_parse(muts)


# -------------------------
# Things these operators should NOT touch
# -------------------------

def test_conditional_operators_ignore_truthiness_check():
    source = """
def task_func(numbers):
    if numbers:
        return 1
"""

    assert (
        ROR().enumerate_mutations(_parse(source), "task_func", source)
        == []
    )

    assert (
        COR().enumerate_mutations(_parse(source), "task_func", source)
        == []
    )

    assert (
        COD().enumerate_mutations(_parse(source), "task_func", source)
        == []
    )


def test_conditional_operators_ignore_method_truthiness():
    source = """
def task_func(numbers):
    if numbers.empty():
        return 1
"""

    assert (
        ROR().enumerate_mutations(_parse(source), "task_func", source)
        == []
    )

    assert (
        COR().enumerate_mutations(_parse(source), "task_func", source)
        == []
    )

    assert (
        COD().enumerate_mutations(_parse(source), "task_func", source)
        == []
    )