"""Specification tests for VVR, VCR, and CVR.

Each scenario describes intended operator behavior. When a test fails, fix the
operators — do not weaken the expectation to match incorrect behavior.
"""

import ast

import pytest

from src.phase1.operators import CVR, VCR, VVR


def _mutations(op, source: str, entry_point: str = "foo"):
    tree = ast.parse(source)
    return op().enumerate_mutations(tree, entry_point, source)


def _vvr(source: str):
    return _mutations(VVR, source)


def _vcr(source: str):
    return _mutations(VCR, source)


def _cvr(source: str):
    return _mutations(CVR, source)


def _pairs(mutations):
    return {(m.site.original, m.site.replacement) for m in mutations}


def _all_parse(mutations):
    for m in mutations:
        ast.parse(ast.unparse(m.tree))


# ---------------------------------------------------------------------------
# Scenario: same-type variables (all primitive types)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lhs,rhs,use",
    [
        ("a = 1", "b = 2", "a"),
        ("a = 1.5", "b = 2.5", "a"),
        ("a = True", "b = False", "a"),
        ('a = "x"', 'b = "y"', "a"),
    ],
    ids=["int", "float", "bool", "str"],
)
def test_vvr_same_type_locals_all_primitives(lhs, rhs, use):
    source = f"""
def foo():
    {lhs}
    {rhs}
    return {use}
"""
    muts = _vvr(source)
    assert len(muts) == 1
    assert muts[0].site.original == use
    assert muts[0].site.replacement == "b"


def test_vvr_same_type_parameters():
    source = """
def foo(a: int, b: int):
    return a
"""
    muts = _vvr(source)
    assert _pairs(muts) == {("a", "b")}
    _all_parse(muts)


def test_vvr_multiple_same_type_candidates():
    source = """
def foo():
    a = 1
    b = 2
    c = 3
    return a
"""
    muts = _vvr(source)
    assert {m.site.replacement for m in muts} == {"b", "c"}


# ---------------------------------------------------------------------------
# Scenario: different types — never cross-replace
# ---------------------------------------------------------------------------


def test_vvr_different_types_not_replaced():
    source = """
def foo():
    x = 1
    y = "hello"
    return x
"""
    assert _vvr(source) == []


def test_vcr_different_types_only_uses_matching_constants():
    source = """
def foo():
    x = 1
    y = "hello"
    return x + len(y)
"""
    muts = _vcr(source)
    by_var = {}
    for m in muts:
        by_var.setdefault(m.site.original, set()).add(m.site.replacement)
    assert by_var.get("x", set()) <= {"0", "1"}
    assert by_var.get("y", set()) <= {"''"}


def test_cvr_different_types_only_replaces_with_matching_variables():
    source = """
def foo():
    x = 1
    y = "hello"
    return 42
"""
    muts = _cvr(source)
    assert len(muts) == 1
    assert muts[0].site.replacement == "x"


# ---------------------------------------------------------------------------
# Scenario: uninferable types — golden rule: skip
# ---------------------------------------------------------------------------


def test_vvr_uninferable_parameter_skipped():
    source = """
def foo(x):
    y = 1
    return x
"""
    assert _vvr(source) == []


def test_vcr_uninferable_parameter_skipped():
    source = """
def foo(x):
    y = 1
    return x
"""
    assert _vcr(source) == []


def test_vvr_uninferable_from_unknown_call_skipped():
    source = """
def foo():
    x = mystery()
    y = 1
    return x
"""
    assert _vvr(source) == []


def test_vcr_uninferable_from_unknown_call_skipped():
    source = """
def foo():
    x = mystery()
    y = 1
    return x
"""
    assert _vcr(source) == []


# ---------------------------------------------------------------------------
# Scenario: no same-type peer, but other types / uninferable vars exist
# ---------------------------------------------------------------------------


def test_vvr_no_same_type_peer_among_other_types():
    source = """
def foo():
    x = 1
    y = "abc"
    z = []
    return x
"""
    assert _vvr(source) == []


def test_cvr_no_same_type_peer_despite_other_variables():
    """Return str literal with only int/list vars in scope — no CVR candidates."""
    source = """
def foo():
    x = 1
    z = []
    return "text"
"""
    assert _cvr(source) == []


def test_cvr_no_same_type_peer_but_one_matching_variable_for_return():
    source = """
def foo():
    x = 1
    y = "abc"
    return "text"
"""
    muts = _cvr(source)
    assert len(muts) == 1
    assert muts[0].site.original == "'text'"
    assert muts[0].site.replacement == "y"


# ---------------------------------------------------------------------------
# Scenario: no variables at all
# ---------------------------------------------------------------------------


def test_vvr_no_variables():
    source = """
def foo():
    return 1
"""
    assert _vvr(source) == []


def test_vcr_no_variables():
    source = """
def foo():
    return 1
"""
    assert _vcr(source) == []


def test_cvr_no_variables():
    source = """
def foo():
    return 1
"""
    assert _cvr(source) == []


# ---------------------------------------------------------------------------
# Scenario: exactly one in-scope variable
# ---------------------------------------------------------------------------


def test_vvr_only_one_variable():
    source = """
def foo():
    x = 1
    return x
"""
    assert _vvr(source) == []


def test_vcr_only_one_variable_still_replaces_with_constants():
    source = """
def foo():
    x = 1
    return x
"""
    muts = _vcr(source)
    assert all(m.site.original == "x" for m in muts)
    assert {m.site.replacement for m in muts} <= {"0", "1"}


def test_cvr_only_one_variable_can_replace_expression_literal():
    source = """
def foo():
    x = 1
    return 0
"""
    muts = _cvr(source)
    assert len(muts) == 1
    assert muts[0].site.replacement == "x"


# ---------------------------------------------------------------------------
# Scenario: list / itertools.combinations-style inference
# ---------------------------------------------------------------------------


def test_vvr_list_variables_same_type():
    source = """
from itertools import combinations

def foo(items):
    result = list(combinations(items, 2))
    other = list(combinations(items, 2))
    return result
"""
    muts = _vvr(source)
    assert len(muts) == 1
    assert _pairs(muts) == {("result", "other")}


def test_cvr_skips_list_literal_without_list_variables():
    source = """
def foo():
    return []
"""
    assert _cvr(source) == []


def test_vcr_skips_list_type_without_matching_constants():
    source = """
def foo(items: list):
    return items
"""
    assert _vcr(source) == []


# ---------------------------------------------------------------------------
# Scenario: nested calls — callee names protected, argument vars eligible
# ---------------------------------------------------------------------------


def test_vvr_nested_calls_protect_call_targets():
    source = """
def foo(a: int, b: int):
    return abs(max(a, b))
"""
    muts = _vvr(source)
    assert "abs" not in {m.site.original for m in muts}
    assert "max" not in {m.site.original for m in muts}
    assert ("a", "b") in _pairs(muts)
    assert ("b", "a") in _pairs(muts)


def test_vcr_nested_calls_protect_call_targets():
    source = """
def foo(nums):
    return sum(nums)
"""
    muts = _vcr(source)
    assert all(m.site.original != "sum" for m in muts)


def test_cvr_nested_call_argument_literals_replaced():
    source = """
def foo():
    x = 1
    y = 2
    return max(0, 1)
"""
    muts = _cvr(source)
    assert {m.site.replacement for m in muts} == {"x", "y"}
    assert all(m.site.original in {"0", "1"} for m in muts)


# ---------------------------------------------------------------------------
# Scenario: assignment from known vs unknown calls (astroid inference)
# ---------------------------------------------------------------------------


def test_vvr_known_builtin_call_assignment_inferred():
    source = """
def foo():
    x = len([1, 2, 3])
    y = len([4, 5])
    return x
"""
    muts = _vvr(source)
    assert len(muts) == 1
    assert _pairs(muts) == {("x", "y")}


def test_vvr_unknown_call_assignment_not_inferred():
    source = """
def foo():
    x = mystery()
    y = 1
    return x
"""
    assert _vvr(source) == []


# ---------------------------------------------------------------------------
# Scenario: parameter + local, for-loop scope, multi-site, user-defined callee
# ---------------------------------------------------------------------------


def test_vvr_parameter_and_local_same_type():
    source = """
def foo(a: int):
    b = 5
    return a
"""
    muts = _vvr(source)
    assert _pairs(muts) == {("a", "b")}


def test_vvr_for_loop_variables_in_scope():
    source = """
def foo():
    total = 0
    other = 1
    for i in range(3):
        total += i
    return total
"""
    assert ("total", "other") in _pairs(_vvr(source))


def test_vvr_same_variable_multiple_use_sites():
    source = """
def foo():
    a = 1
    b = 2
    return a + a
"""
    muts = _vvr(source)
    assert len(muts) == 2
    assert all(m.site.original == "a" and m.site.replacement == "b" for m in muts)


def test_vvr_user_defined_function_call_target_not_mutated():
    source = """
def helper():
    return 1

def foo():
    x = 1
    return helper()
"""
    muts = _vvr(source)
    assert all(m.site.original != "helper" for m in muts)


# ---------------------------------------------------------------------------
# Scenario: conflicting inference removes variable from candidate pool
# ---------------------------------------------------------------------------


def test_vvr_conflicting_inference_skips_ambiguous_variable():
    source = '''
def foo(flag):
    if flag:
        x = 1
    else:
        x = "hello"

    y = 5
    return y
'''
    assert _vvr(source) == []


# ---------------------------------------------------------------------------
# Scenario: CVR must not mutate defining assignment literals
# ---------------------------------------------------------------------------


def test_cvr_skips_assignment_rhs_literals():
    source = """
def foo():
    x = 2
    y = 3
    return 1
"""
    muts = _cvr(source)
    assert all(m.site.original == "1" for m in muts)
    assert {m.site.replacement for m in muts} == {"x", "y"}


def test_cvr_assignment_rhs_ignored_even_when_no_expression_literal_matches():
    source = """
def foo():
    x = 1
    return x
"""
    assert _cvr(source) == []


# ---------------------------------------------------------------------------
# VCR constant pools per primitive type
# ---------------------------------------------------------------------------


def test_vcr_int_variable_gets_zero_and_one():
    source = """
def foo(a: int):
    return a
"""
    muts = _vcr(source)
    assert {m.site.replacement for m in muts} <= {"0", "1"}
    _all_parse(muts)


def test_vcr_bool_variable_gets_true_and_false():
    source = """
def foo():
    flag = True
    return flag
"""
    muts = _vcr(source)
    assert {m.site.replacement for m in muts} == {"True", "False"}


def test_vcr_str_variable_gets_empty_string():
    source = """
def foo():
    name = "alice"
    return name
"""
    muts = _vcr(source)
    assert len(muts) == 1
    assert muts[0].site.replacement == "''"


def test_cvr_bool_constant_replaced_with_bool_variables():
    source = """
def foo():
    ok = True
    bad = False
    return True
"""
    muts = _cvr(source)
    assert {m.site.replacement for m in muts} == {"ok", "bad"}


def test_cvr_str_constant_replaced_with_string_variables():
    source = """
def foo():
    a = "x"
    b = "y"
    return "hello"
"""
    muts = _cvr(source)
    assert {m.site.replacement for m in muts} == {"a", "b"}
