import ast

import pytest

from src.phase1.operators import (
    _body_nodes_preorder,
    _deep_replace,
    _replace_node,
    _replace_in_lockstep,
    _delete_stmt_in_body,
    _get_function_body,
)


def test_body_nodes_preorder_contains_root_statement_and_children():
    tree = ast.parse(
        """
def foo():
    x = 1 + 2
"""
    )

    _, body = _get_function_body(tree, "foo")

    nodes = _body_nodes_preorder(body)

    assert any(isinstance(n, ast.Assign) for n in nodes)
    assert any(isinstance(n, ast.BinOp) for n in nodes)
    assert any(isinstance(n, ast.Constant) and n.value == 1 for n in nodes)
    assert any(isinstance(n, ast.Constant) and n.value == 2 for n in nodes)


def test_body_nodes_preorder_includes_nested_if_nodes():
    tree = ast.parse(
        """
def foo():
    if True:
        y = 1
"""
    )

    _, body = _get_function_body(tree, "foo")

    nodes = _body_nodes_preorder(body)

    assert any(isinstance(n, ast.If) for n in nodes)
    assert any(isinstance(n, ast.Assign) for n in nodes)


def test_body_nodes_preorder_is_deterministic():
    tree = ast.parse(
        """
def foo():
    x = 1
    y = 2
"""
    )

    _, body = _get_function_body(tree, "foo")

    first = [ast.dump(n) for n in _body_nodes_preorder(body)]
    second = [ast.dump(n) for n in _body_nodes_preorder(body)]

    assert first == second


def test_deep_replace_replaces_name_node():
    tree = ast.parse(
        """
def foo():
    return x
"""
    )

    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "x"
    )

    new_tree = _deep_replace(
        tree,
        target,
        ast.Name(id="y", ctx=ast.Load()),
    )

    assert "return y" in ast.unparse(new_tree)
    assert "return x" in ast.unparse(tree)


def test_deep_replace_replaces_nested_expression():
    tree = ast.parse(
        """
def foo():
    return (x + y) * z
"""
    )

    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.BinOp)
        and isinstance(n.op, ast.Add)
    )

    new_tree = _deep_replace(
        tree,
        target,
        ast.Constant(value=5),
    )

    assert "return 5 * z" in ast.unparse(new_tree)


def test_deep_replace_matches_identity_not_value():
    tree = ast.parse(
        """
def foo():
    return x + x
"""
    )

    names = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name)
    ]

    target = names[0]

    new_tree = _deep_replace(
        tree,
        target,
        ast.Name(id="y", ctx=ast.Load()),
    )

    source = ast.unparse(new_tree)

    assert "y + x" in source
    assert "y + y" not in source


def test_replace_node_delegates_to_deep_replace():
    tree = ast.parse(
        """
def foo():
    return x
"""
    )

    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Name)
    )

    new_tree = _replace_node(
        tree,
        target,
        ast.Name(id="replacement", ctx=ast.Load()),
    )

    assert "replacement" in ast.unparse(new_tree)


def test_replace_in_lockstep_replaces_ast_field():
    original = ast.parse(
        """
def foo():
    return x
"""
    )

    copied = ast.parse(
        """
def foo():
    return x
"""
    )

    target = next(
        n for n in ast.walk(original)
        if isinstance(n, ast.Name)
    )

    _replace_in_lockstep(
        original,
        copied,
        id(target),
        ast.Name(id="y", ctx=ast.Load()),
    )

    assert "return y" in ast.unparse(copied)


def test_replace_in_lockstep_replaces_list_element():
    original = ast.parse(
        """
def foo():
    x = 1
    y = 2
"""
    )

    copied = ast.parse(
        """
def foo():
    x = 1
    y = 2
"""
    )

    target = next(
        n for n in ast.walk(original)
        if isinstance(n, ast.Assign)
        and n.targets[0].id == "y"
    )

    replacement = ast.Pass()

    _replace_in_lockstep(
        original,
        copied,
        id(target),
        replacement,
    )

    source = ast.unparse(copied)

    assert "pass" in source


def test_delete_stmt_in_body_removes_middle_statement():
    tree = ast.parse(
        """
def foo():
    a = 1
    b = 2
    c = 3
"""
    )

    func, body = _get_function_body(tree, "foo")

    new_tree = _delete_stmt_in_body(
        tree,
        func,
        body,
        1,
        func.body,
    )

    source = ast.unparse(new_tree)

    assert "a = 1" in source
    assert "c = 3" in source
    assert "b = 2" not in source


def test_delete_stmt_in_body_deletes_only_statement_and_inserts_pass():
    tree = ast.parse(
        """
def foo():
    x = 1
"""
    )

    func, body = _get_function_body(tree, "foo")

    new_tree = _delete_stmt_in_body(
        tree,
        func,
        body,
        0,
        func.body,
    )

    source = ast.unparse(new_tree)

    assert "pass" in source


def test_delete_stmt_in_body_docstring_only_remaining_inserts_pass():
    tree = ast.parse(
        '''
def foo():
    """doc"""
    x = 1
'''
    )

    func, body = _get_function_body(tree, "foo")

    new_tree = _delete_stmt_in_body(
        tree,
        func,
        body,
        0,
        func.body,
    )

    source = ast.unparse(new_tree)

    assert "pass" in source


def test_delete_stmt_in_body_raises_when_function_missing():
    tree = ast.parse(
        """
def foo():
    x = 1
"""
    )

    fake_func = ast.FunctionDef(
        name="does_not_exist",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[],
        decorator_list=[],
    )

    with pytest.raises(ValueError):
        _delete_stmt_in_body(
            tree,
            fake_func,
            [tree.body[0].body[0]],
            0,
            [tree.body[0].body[0]],
        )