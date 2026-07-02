import ast

# Adjust the import path if your registry is located elsewhere
from src.phase1.operators import RVM, TR, FR


def _rvm(source: str):
    tree = ast.parse(source)
    return RVM().enumerate_mutations(tree, "foo", source)


def _tr(source: str):
    tree = ast.parse(source)
    return TR().enumerate_mutations(tree, "foo", source)


def _fr(source: str):
    tree = ast.parse(source)
    return FR().enumerate_mutations(tree, "foo", source)


# ==========================================================
# RVM (Return Value Mutation)
# ==========================================================

def test_rvm_integer_constant_return():
    source = """
def foo():
    return 5
"""
    muts = _rvm(source)

    assert len(muts) == 1
    assert muts[0].site.replacement == "return -<expr>"


def test_rvm_float_constant_return():
    source = """
def foo():
    return 3.14
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_integer_variable_return():
    source = """
def foo():
    x = 5
    return x
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_float_variable_return():
    source = """
def foo():
    x = 1.5
    return x
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_parameter_return():
    source = """
def foo(x: int):
    return x
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_bool_return_uses_not():
    source = """
def foo():
    return True
"""
    muts = _rvm(source)

    assert len(muts) == 1
    assert muts[0].site.replacement == "return not <expr>"


def test_rvm_bool_variable_return():
    source = """
def foo():
    flag = True
    return flag
"""
    muts = _rvm(source)

    assert len(muts) == 1
    assert muts[0].site.replacement == "return not <expr>"


def test_rvm_multiple_returns_same_numeric_type():
    source = """
def foo(x):
    if x > 0:
        return 1
    return 2
"""
    muts = _rvm(source)

    assert len(muts) == 2


def test_rvm_multiple_returns_same_bool_type():
    source = """
def foo(flag):
    if flag:
        return True
    return False
"""
    muts = _rvm(source)

    assert len(muts) == 2


def test_rvm_mixed_return_types_not_mutated():
    source = """
def foo(flag):
    if flag:
        return 1
    return False
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_string_return_not_mutated():
    source = """
def foo():
    return "hello"
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_none_return_not_mutated():
    source = """
def foo():
    return None
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_list_return_not_mutated():
    source = """
def foo():
    return [1, 2, 3]
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_dict_return_not_mutated():
    source = """
def foo():
    return {"a": 1}
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_tuple_return_not_mutated():
    source = """
def foo():
    return (1, 2)
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_uninferable_parameter_return():
    source = """
def foo(x):
    return x
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_unknown_function_return():
    source = """
def foo():
    return mystery()
"""
    muts = _rvm(source)

    assert muts == []


def test_rvm_builtin_numeric_return():
    source = """
def foo():
    return len([1, 2, 3])
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_builtin_bool_return():
    source = """
def foo():
    return bool(1)
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_nested_numeric_expression_return():
    source = """
def foo():
    x = 1
    y = 2
    return x + y
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_nested_bool_expression_return():
    source = """
def foo():
    a = True
    b = False
    return a and b
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_return_inside_loop():
    source = """
def foo():
    for i in range(10):
        return 1
"""
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_return_inside_nested_if():
    source = """
def foo(x):
    if x > 0:
        if x > 10:
            return 1
    return 2
"""
    muts = _rvm(source)

    assert len(muts) == 2


def test_rvm_return_after_docstring():
    source = '''
def foo():
    """doc"""
    return 1
'''
    muts = _rvm(source)

    assert len(muts) == 1


def test_rvm_void_return_not_mutated():
    source = """
def foo():
    return
"""
    muts = _rvm(source)

    assert muts == []


# ==========================================================
# TR (True Return)
# ==========================================================

def test_tr_true_constant_return():
    source = """
def foo():
    return False
"""
    muts = _tr(source)

    assert len(muts) == 1
    assert muts[0].site.replacement == "return True"


def test_tr_bool_variable_return():
    source = """
def foo():
    flag = False
    return flag
"""
    muts = _tr(source)

    assert len(muts) == 1


def test_tr_multiple_bool_returns():
    source = """
def foo(flag):
    if flag:
        return True
    return False
"""
    muts = _tr(source)

    assert len(muts) == 2


def test_tr_integer_return_not_mutated():
    source = """
def foo():
    return 1
"""
    muts = _tr(source)

    assert muts == []


def test_tr_float_return_not_mutated():
    source = """
def foo():
    return 1.5
"""
    muts = _tr(source)

    assert muts == []


def test_tr_string_return_not_mutated():
    source = """
def foo():
    return "hello"
"""
    muts = _tr(source)

    assert muts == []


def test_tr_uninferable_return_not_mutated():
    source = """
def foo(x):
    return x
"""
    muts = _tr(source)

    assert muts == []


def test_tr_bool_expression_return():
    source = """
def foo():
    a = True
    b = False
    return a and b
"""
    muts = _tr(source)

    assert len(muts) == 1


# ==========================================================
# FR (False Return)
# ==========================================================

def test_fr_false_constant_return():
    source = """
def foo():
    return True
"""
    muts = _fr(source)

    assert len(muts) == 1
    assert muts[0].site.replacement == "return False"


def test_fr_bool_variable_return():
    source = """
def foo():
    flag = True
    return flag
"""
    muts = _fr(source)

    assert len(muts) == 1


def test_fr_multiple_bool_returns():
    source = """
def foo(flag):
    if flag:
        return True
    return False
"""
    muts = _fr(source)

    assert len(muts) == 2


def test_fr_integer_return_not_mutated():
    source = """
def foo():
    return 1
"""
    muts = _fr(source)

    assert muts == []


def test_fr_float_return_not_mutated():
    source = """
def foo():
    return 1.5
"""
    muts = _fr(source)

    assert muts == []


def test_fr_string_return_not_mutated():
    source = """
def foo():
    return "hello"
"""
    muts = _fr(source)

    assert muts == []


def test_fr_uninferable_return_not_mutated():
    source = """
def foo(x):
    return x
"""
    muts = _fr(source)

    assert muts == []


def test_fr_bool_expression_return():
    source = """
def foo():
    a = True
    b = False
    return a or b
"""
    muts = _fr(source)

    assert len(muts) == 1