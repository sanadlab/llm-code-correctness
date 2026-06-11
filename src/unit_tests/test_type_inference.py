import ast
import astroid

from src.phase1.operators import (
    _resolve_inferred_type,
    _infer_node_type,
    _infer_arg_type,
    _infer_return_type,
    _build_type_maps,
    _register_var_type,
    _remove_var_type,
    _binop_operands_type_ok,
    _call_target_ids,
    _RETURN_COORD,
)


def test_resolve_inferred_type_int():
    node = astroid.extract_node("1")
    assert _resolve_inferred_type(node) == "int"


def test_resolve_inferred_type_string():
    node = astroid.extract_node("__('hello')")
    assert _resolve_inferred_type(node) == "str"


def test_resolve_inferred_type_bool():
    node = astroid.extract_node("True")
    assert _resolve_inferred_type(node) == "bool"


def test_resolve_inferred_type_none():
    node = astroid.extract_node("None")
    assert _resolve_inferred_type(node) == "NoneType"


def test_infer_node_type_variable():
    module = astroid.parse(
        """
x = 1
y = x
"""
    )

    assign = module.body[1]
    name_node = assign.value

    assert _infer_node_type(name_node) == "int"


def test_infer_node_type_uninferable():
    module = astroid.parse(
        """
y = unknown_name
"""
    )

    assign = module.body[0]

    assert _infer_node_type(assign.value) is None


def test_infer_arg_type_annotated():
    module = astroid.parse(
        """
def foo(x: int):
    return x
"""
    )

    func = module.body[0]
    arg = func.args.arguments[0]

    assert _infer_arg_type(arg) == "int"


def test_infer_arg_type_unannotated():
    module = astroid.parse(
        """
def foo(x):
    return x
"""
    )

    func = module.body[0]
    arg = func.args.arguments[0]

    assert _infer_arg_type(arg) is None


def test_infer_return_type_single_type():
    module = astroid.parse(
        """
def foo():
    return 1
"""
    )

    func = module.body[0]

    assert _infer_return_type(func) == "int"


def test_infer_return_type_mixed_types():
    module = astroid.parse(
        """
def foo(x):
    if x:
        return 1
    return "hello"
"""
    )

    func = module.body[0]

    assert _infer_return_type(func) is None


def test_register_var_type():
    var_to_type = {}
    type_to_vars = {}

    _register_var_type(
        var_to_type,
        type_to_vars,
        "x",
        "int",
    )

    assert var_to_type["x"] == "int"
    assert "x" in type_to_vars["int"]


def test_register_var_type_conflict_removes_variable():
    var_to_type = {"x": "int"}
    type_to_vars = {"int": ["x"]}

    _register_var_type(
        var_to_type,
        type_to_vars,
        "x",
        "str",
    )

    assert "x" not in var_to_type
    assert "int" not in type_to_vars


def test_remove_var_type():
    var_to_type = {"x": "int"}
    type_to_vars = {"int": ["x"]}

    _remove_var_type(
        var_to_type,
        type_to_vars,
        "x",
    )

    assert var_to_type == {}
    assert type_to_vars == {}


def test_build_type_maps_basic():
    source = """
def task_func(a: int):
    x = 1
    y = x + a
    return x
"""

    coord_to_type, type_to_vars, var_to_type = (
        _build_type_maps(source, "task_func")
    )

    assert var_to_type["a"] == "int"
    assert var_to_type["x"] == "int"

    assert "a" in type_to_vars["int"]
    assert "x" in type_to_vars["int"]

    assert coord_to_type[_RETURN_COORD] == "int"


def test_build_type_maps_invalid_source():
    coord_to_type, type_to_vars, var_to_type = (
        _build_type_maps(
            "def broken(:",
            "task_func",
        )
    )

    assert coord_to_type == {}
    assert type_to_vars == {}
    assert var_to_type == {}


def test_build_type_maps_missing_function():
    source = """
def other():
    return 1
"""

    coord_to_type, type_to_vars, var_to_type = (
        _build_type_maps(source, "task_func")
    )

    assert coord_to_type == {}
    assert type_to_vars == {}
    assert var_to_type == {}


def test_binop_operands_type_ok_numeric():
    source = """
def task_func(a: int):
    return a + 1
"""

    tree = ast.parse(source)

    binop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
    )

    coord_to_type, _, var_to_type = _build_type_maps(
        source,
        "task_func",
    )

    assert _binop_operands_type_ok(
        binop,
        coord_to_type,
        var_to_type,
    )


def test_binop_operands_type_ok_missing_type():
    source = """
def task_func(a):
    return a + 1
"""

    tree = ast.parse(source)

    binop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
    )

    assert not _binop_operands_type_ok(
        binop,
        {},
    )


def test_call_target_ids():
    tree = ast.parse(
        """
def task_func(nums):
    return sum(nums)
"""
    )

    func = tree.body[0]

    ids = _call_target_ids(func.body)

    call = next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
    )

    assert id(call.func) in ids