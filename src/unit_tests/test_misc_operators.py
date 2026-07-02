import ast

from src.phase1.operators import CR, NEG, FCD, CRN, RC


def _parse(source: str):
    return ast.parse(source)


def test_cr_replaces_int_constant_with_alternatives():
    source = """
def foo() -> int:
    x = 1
    return x
"""
    tree = _parse(source)

    muts = CR().enumerate_mutations(tree, "foo", source)

    assert len(muts) >= 1
    originals = {m.site.original for m in muts}
    replacements = {m.site.replacement for m in muts}

    assert "1" in originals
    assert any(rep in {"0", "1", "-1"} for rep in replacements)


def test_cr_replaces_string_constant_when_types_match():
    source = '''
def foo() -> str:
    msg = "hello"
    return msg
'''
    tree = _parse(source)

    muts = CR().enumerate_mutations(tree, "foo", source)

    replacements = {m.site.replacement for m in muts}
    assert "''" in replacements


def test_neg_negates_non_zero_numeric_literals():
    source = """
def foo():
    return 2
"""
    tree = _parse(source)

    muts = NEG().enumerate_mutations(tree, "foo", source)

    assert len(muts) == 1
    site = muts[0].site
    assert site.original == "2"
    assert site.replacement == "-2"

    mutated_code = ast.unparse(muts[0].tree)
    assert "return -2" in mutated_code


def test_fcd_deletes_function_and_method_call_statements():
    source = """
def foo(obj):
    print("hello")
    obj.do_something()
    x = 1
    return x
"""
    tree = _parse(source)

    muts = FCD().enumerate_mutations(tree, "foo", source)

    assert len(muts) == 2
    originals = {m.site.original for m in muts}
    assert originals == {"print", "do_something"}
    for m in muts:
        assert m.site.replacement == "<deleted>"

    code_variants = [ast.unparse(m.tree) for m in muts]
    assert any("print(\"hello\")" not in c for c in code_variants)
    assert any("do_something" not in c for c in code_variants)


def test_crn_replaces_constructor_calls_with_none():
    source = """
class MyClass:
    pass

def foo():
    x = MyClass()
    y = helper()
    return x
"""
    tree = _parse(source)

    muts = CRN().enumerate_mutations(tree, "foo", source)

    assert len(muts) == 1
    site = muts[0].site
    assert site.node_type == "Call"
    assert site.original == "MyClass(...)"
    assert site.replacement == "None"

    code = ast.unparse(muts[0].tree)
    assert "MyClass(" not in code
    assert "None" in code
    assert "helper(" in code


def test_rc_replaces_if_and_while_conditions_with_true():
    source = """
def foo(x):
    if x > 0:
        return 1
    while x > 0:
        x -= 1
    return x
"""
    tree = _parse(source)

    muts = RC().enumerate_mutations(tree, "foo", source)

    assert len(muts) == 2
    node_types = {m.site.node_type for m in muts}
    replacements = {m.site.replacement for m in muts}

    assert node_types == {"If", "While"}
    assert replacements == {"True"}

    codes = [ast.unparse(m.tree) for m in muts]
    assert any("if True" in c for c in codes)
    assert any("while True" in c for c in codes)

