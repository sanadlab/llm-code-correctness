import ast

import pytest

from src.phase1.operators import (
    _constant_primitive_type,
    _constant_in_assignment_rhs,
    _get_function_body,
    _in_scope_variables,
    _node_in_body,
    _stmt_index_of,
)


def test_get_function_body_finds_nested_function():
    tree = ast.parse(
        """
def outer():
    def inner():
        x = 1
        return x
    return inner()
"""
    )

    func, body = _get_function_body(tree, "inner")

    assert func.name == "inner"
    assert len(body) == 2


def test_get_function_body_docstring_only():
    tree = ast.parse(
        '''
def foo():
    """doc"""
'''
    )

    _, body = _get_function_body(tree, "foo")

    assert body == []


def test_in_scope_variables_collects_varargs_and_kwargs():
    tree = ast.parse(
        """
def foo(a, *args, b, **kwargs):
    pass
"""
    )

    func = tree.body[0]

    names = _in_scope_variables(func)

    assert set(names) == {"a", "args", "b", "kwargs"}


def test_in_scope_variables_returns_sorted_names():
    tree = ast.parse(
        """
def foo(z):
    b = 1
    a = 2
"""
    )

    func = tree.body[0]

    names = _in_scope_variables(func)

    assert names == sorted(names)


def test_node_in_body_returns_true_for_nested_node():
    tree = ast.parse(
        """
def foo():
    x = 1 + 2
"""
    )

    _, body = _get_function_body(tree, "foo")

    constant = next(
        node for node in ast.walk(body[0])
        if isinstance(node, ast.Constant) and node.value == 1
    )

    assert _node_in_body(constant, body)


def test_stmt_index_of_nested_node_returns_parent_statement_index():
    tree = ast.parse(
        """
def foo():
    x = 1
    y = 2 + 3
"""
    )

    _, body = _get_function_body(tree, "foo")

    constant = next(
        node for node in ast.walk(body[1])
        if isinstance(node, ast.Constant) and node.value == 2
    )

    assert _stmt_index_of(constant, body) == 1


def test_constant_in_assignment_rhs_detects_initializer_literal():
    tree = ast.parse(
        """
def foo():
    x = 1
    return 1
"""
    )

    _, body = _get_function_body(tree, "foo")

    assign_const = next(
        node
        for node in ast.walk(body[0])
        if isinstance(node, ast.Constant) and node.value == 1
    )
    return_const = next(
        node
        for node in ast.walk(body[1])
        if isinstance(node, ast.Constant) and node.value == 1
    )

    assert _constant_in_assignment_rhs(assign_const, body)
    assert not _constant_in_assignment_rhs(return_const, body)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        (),
        object(),
        complex(1, 2),
        b"bytes",
    ],
)
def test_constant_primitive_type_returns_none_for_unsupported_types(value):
    assert _constant_primitive_type(value) is None