import ast

from src.phase1.operators import UOI, UOD


def _parse(src: str):
    return ast.parse(src)


def test_uoi_numeric_variable_gets_negation_only():
    source = """
def foo(x: int):
    return x
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == {"-x"}


def test_uoi_float_variable_gets_negation_only():
    source = """
def foo(x: float):
    return x
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == {"-x"}


def test_uoi_string_variable_gets_no_unary_insertion():
    source = '''
def foo(x: str):
    return x
'''
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == set()


def test_uoi_list_variable_gets_no_unary_insertion():
    source = """
def foo(numbers):
    numbers = [1, 2, 3]
    return numbers
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == set()


def test_uoi_bool_variable_gets_only_not():
    source = """
def foo(flag: bool):
    return flag
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not flag" in replacements
    assert "-flag" not in replacements


def test_uoi_skips_function_call_target():
    source = """
def foo(x: int):
    return bar(x)
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    originals = {m.site.original for m in muts}

    assert "bar" not in originals


def test_uoi_does_not_mutate_builtin_call_target():
    source = """
def foo(x: int):
    return list(x)
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    originals = {m.site.original for m in muts}

    assert "list" not in originals


def test_uoi_mutates_argument_of_call():
    source = """
def foo(x: int):
    return bar(x)
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == {"-x"}


def test_uoi_handles_multiple_numeric_variables():
    source = """
def foo(a: int, b: int):
    return a + b
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == {"-a", "-b"}


def test_uoi_method_call_does_not_mutate_receiver_inside_call():
    source = """
def foo(numbers):
    return numbers.empty()
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    for m in muts:
        mutated_code = ast.unparse(m.tree)
        assert "(not numbers).empty()" not in mutated_code


def test_uoi_variable_assigned_slice_can_be_mutated():
    source = """
def foo(numbers):
    nums_sub = numbers[1:3]
    return nums_sub
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == set()


def test_uoi_if_condition_variable_can_be_mutated():
    source = """
def foo(numbers):
    if numbers:
        return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not numbers" in replacements


def test_uoi_nested_if_condition_variable_can_be_mutated():
    source = """
def foo(a: bool, b: bool):
    if a:
        if b:
            return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not a" in replacements
    assert "not b" in replacements


def test_uoi_one_line_if():
    source = """
def foo(flag: bool):
    if flag: return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not flag" in replacements


def test_uoi_if_compare_clause_treated_as_black_box():
    source = """
def foo(x):
    if x < 0:
        return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)
    replacements = {m.site.replacement for m in muts}

    assert "not x < 0" in replacements


def test_uoi_if_call_clause_treated_as_black_box():
    source = """
def foo(items):
    if is_empty(items):
        return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)
    replacements = {m.site.replacement for m in muts}

    assert "not is_empty(items)" in replacements


def test_uoi_while_clause_treated_as_black_box():
    source = """
def foo(i, n):
    while i < n:
        i += 1
    return i
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)
    replacements = {m.site.replacement for m in muts}

    assert "not i < n" in replacements


def test_uoi_boolop_if_mutates_each_clause():
    source = """
def foo(a, b, c):
    if a or b or c:
        return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)
    replacements = {m.site.replacement for m in muts}

    assert {"not a", "not b", "not c"}.issubset(replacements)


def test_uoi_attribute_call_clause_uses_whole_clause_not_receiver():
    source = """
def foo(obj):
    if obj.empty():
        return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)
    replacements = {m.site.replacement for m in muts}
    mutated_sources = [ast.unparse(m.tree) for m in muts]

    assert "not obj.empty()" in replacements
    assert all("(not obj).empty()" not in code for code in mutated_sources)


def test_uoi_nested_boolop_if_mutates_top_level_clauses():
    source = """
def foo(a, b, c):
    if (a and b) or c:
        return 1
    return 0
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)
    replacements = {m.site.replacement for m in muts}

    assert "not a and b" in replacements
    assert "not c" in replacements


def test_uoi_uninferable_variable_gets_no_unary_insertion():
    source = """
def foo(x):
    return x
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert replacements == set()


def test_uod_deletes_various_unary_operators():
    source = """
def foo(x):
    return -x, not x, +x, ~x
"""
    tree = _parse(source)

    muts = UOD().enumerate_mutations(tree, "foo", source)

    assert len(muts) == 4
    originals = {m.site.original for m in muts}
    replacements = {m.site.replacement for m in muts}

    assert originals == {"-", "not", "+", "~"}
    assert replacements == {"<removed>"}