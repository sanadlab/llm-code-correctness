import ast

from src.phase1.operators import UOI, UOD


def _parse(src: str):
    return ast.parse(src)


def test_uoi_numeric_variable_gets_negation_and_not():
    source = """
def foo(x: int):
    return x
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "-x" in replacements
    assert "not x" in replacements


def test_uoi_float_variable_gets_negation_and_not():
    source = """
def foo(x: float):
    return x
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "-x" in replacements
    assert "not x" in replacements


def test_uoi_string_variable_gets_only_not():
    source = '''
def foo(x: str):
    return x
'''
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not x" in replacements
    assert "-x" not in replacements


def test_uoi_list_variable_gets_only_not():
    source = """
def foo(numbers):
    numbers = [1, 2, 3]
    return numbers
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not numbers" in replacements
    assert "-numbers" not in replacements


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

    assert "-x" in replacements
    assert "not x" in replacements


def test_uoi_handles_multiple_numeric_variables():
    source = """
def foo(a: int, b: int):
    return a + b
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = [m.site.replacement for m in muts]

    assert "-a" in replacements
    assert "not a" in replacements
    assert "-b" in replacements
    assert "not b" in replacements


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

    assert "not nums_sub" in replacements


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


def test_uoi_uninferable_variable_still_gets_not():
    source = """
def foo(x):
    return x
"""
    tree = _parse(source)

    muts = UOI().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}

    assert "not x" in replacements
    assert "-x" not in replacements


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