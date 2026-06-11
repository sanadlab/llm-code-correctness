"""
Tests for AOR, AOD, and AOI.
"""

import ast

from src.phase1.operators import AOR, AOD, AOI


def _parse(source: str):
    return ast.parse(source)


def _all_parse(mutations):
    for m in mutations:
        ast.parse(ast.unparse(m.tree))


# -------------------------
# AOR
# -------------------------

def test_aor_emits_numeric_replacements():
    source = """
def task_func(a: int, b: int):
    return a + b
"""

    muts = AOR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) > 0
    assert all(m.operator_code == "AOR" for m in muts)
    _all_parse(muts)


def test_aor_skips_string_concatenation():
    source = """
def task_func(a: str, b: str):
    return a + b
"""

    muts = AOR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aor_skips_list_repetition_literal():
    source = """
def task_func():
    return [1, 2] * 3
"""

    muts = AOR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aor_skips_list_variable():
    source = """
def task_func():
    xs = [1, 2, 3]
    return xs * 3
"""

    muts = AOR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aor_emits_for_float_operands():
    source = """
def task_func(a: float, b: float):
    return a - b
"""

    muts = AOR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) > 0
    _all_parse(muts)


def test_aor_skips_uninferable_types():
    source = """
def task_func(a, b):
    return a + b
"""

    muts = AOR().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


# -------------------------
# AOD
# -------------------------

def test_aod_emits_left_and_right_deletions():
    source = """
def task_func(a: int, b: int):
    return a + b
"""

    muts = AOD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 2

    replacements = {
        m.site.replacement
        for m in muts
    }

    assert replacements == {"a", "b"}

    _all_parse(muts)


def test_aod_skips_string_concat():
    source = """
def task_func(a: str, b: str):
    return a + b
"""

    muts = AOD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aod_skips_list_repetition_literal():
    source = """
def task_func():
    return [1, 2] * 3
"""

    muts = AOD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aod_skips_list_variable():
    source = """
def task_func():
    xs = [1, 2]
    return xs * 3
"""

    muts = AOD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aod_skips_uninferable():
    source = """
def task_func(a, b):
    return a + b
"""

    muts = AOD().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


# -------------------------
# AOI
# -------------------------

def test_aoi_emits_numeric_insertions():
    source = """
def task_func(x: int):
    return x
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 2

    replacements = {
        m.site.replacement
        for m in muts
    }

    assert replacements == {
        "x + 1",
        "x - 1",
    }

    _all_parse(muts)


def test_aoi_emits_for_float():
    source = """
def task_func(x: float):
    return x
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert len(muts) == 2


def test_aoi_skips_list_variable():
    source = """
def task_func(xs: list):
    return xs
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aoi_skips_string_variable():
    source = """
def task_func(s: str):
    return s
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aoi_skips_uninferable():
    source = """
def task_func(x):
    return x
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    assert muts == []


def test_aoi_skips_call_target():
    source = """
def task_func(nums):
    return sum(nums)
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    mutated_sources = [
        ast.unparse(m.tree)
        for m in muts
    ]

    assert all("sum + 1" not in s for s in mutated_sources)
    assert all("sum - 1" not in s for s in mutated_sources)

def test_aoi_does_not_mutate_function_name_in_call():
    source = """
def task_func(nums):
    return sum(nums)
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    for m in muts:
        mutated = ast.unparse(m.tree)

        assert "(sum + 1)(" not in mutated
        assert "(sum - 1)(" not in mutated
        assert "sum + 1(" not in mutated
        assert "sum - 1(" not in mutated

def test_aoi_does_not_mutate_attribute_call_target():
    source = """
import itertools

def task_func(items):
    return itertools.combinations(items, 2)
"""

    muts = AOI().enumerate_mutations(
        _parse(source),
        "task_func",
        source,
    )

    for m in muts:
        mutated = ast.unparse(m.tree)

        assert "itertools + 1" not in mutated
        assert "itertools - 1" not in mutated