"""Mutation operator registry.

Each operator is a class that:
1. Enumerates applicable sites in deterministic preorder AST traversal order.
2. For each site, yields one or more (replacement_index, mutated_ast) pairs.

All operators mutate only nodes inside the target function body (after the
docstring), never the function signature, decorators, annotations, or docstring.
"""

import ast
import copy
from dataclasses import dataclass
from typing import Any

# ── Site metadata ─────────────────────────────────────────

@dataclass
class MutationSite:
    line: int
    col: int
    node_type: str
    original: str
    replacement: str


@dataclass
class Mutation:
    operator: str
    operator_code: str
    site_index: int
    replacement_index: int
    site: MutationSite
    # The mutated full-module AST (caller serializes)
    tree: ast.Module


# ── Helpers ───────────────────────────────────────────────

def _body_nodes_preorder(func_body: list[ast.stmt]) -> list[ast.AST]:
    """Collect all AST nodes inside a function body in deterministic preorder."""
    nodes = []
    for stmt in func_body:
        for node in ast.walk(stmt):
            nodes.append(node)
    return nodes


def _get_function_body(tree: ast.Module, entry_point: str) -> tuple[ast.FunctionDef, list[ast.stmt]]:
    """Find the FunctionDef and return (func_node, mutable_body).
    The mutable body skips a leading docstring Expr if present."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            body = node.body
            # Skip docstring
            if (body and isinstance(body[0], ast.Expr) and
                isinstance(body[0].value, (ast.Constant, ast.Str))):
                return node, body[1:]
            return node, body
    raise ValueError(f"Function {entry_point} not found in AST")


def _in_scope_variables(func_node: ast.FunctionDef) -> list[str]:
    """Collect variable names visible inside the function (args + assignments)."""
    names = set()
    for arg in func_node.args.args:
        names.add(arg.arg)
    if func_node.args.vararg:
        names.add(func_node.args.vararg.arg)
    if func_node.args.kwarg:
        names.add(func_node.args.kwarg.arg)
    for arg in func_node.args.kwonlyargs:
        names.add(arg.arg)
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.For):
            # Walk the loop's target to catch tuple/list unpacking
            for target_node in ast.walk(node.target):
                if isinstance(target_node, ast.Name) and isinstance(target_node.ctx, ast.Store):
                    names.add(target_node.id)
    return sorted(names)


def _node_in_body(node: ast.AST, body_stmts: list[ast.stmt]) -> bool:
    """Check if a node is inside one of the body statements."""
    for stmt in body_stmts:
        for n in ast.walk(stmt):
            if n is node:
                return True
    return False


def _stmt_index_of(node: ast.AST, body_stmts: list[ast.stmt]) -> int | None:
    """Find which body statement contains this node."""
    for i, stmt in enumerate(body_stmts):
        for n in ast.walk(stmt):
            if n is node:
                return i
    return None


def _replace_node(tree: ast.Module, old_node: ast.AST, new_node: ast.AST) -> ast.Module:
    """Replace old_node with new_node in-place in a deep-copied tree."""

    class Replacer(ast.NodeTransformer):
        def __init__(self, old_id):
            self.old_id = old_id

        def visit(self, node):
            if id(node) == self.old_id:
                return ast.copy_location(new_node, node)
            return self.generic_visit(node)

    # We need to track by id, so we operate on the original tree's ids
    old_id = id(old_node)
    new_tree = copy.deepcopy(tree)
    # But deepcopy changes ids, so we need a different approach:
    # Walk both trees in parallel to find the matching node.
    return _deep_replace(tree, old_node, new_node)


def _deep_replace(tree: ast.Module, old_node: ast.AST, new_node: ast.AST) -> ast.Module:
    """Deep-copy tree with old_node replaced by new_node (matched by identity)."""
    old_id = id(old_node)
    id_map = {}  # maps id -> True for nodes to replace

    class _Marker(ast.NodeVisitor):
        def visit(self, node):
            if id(node) == old_id:
                id_map[old_id] = True
            self.generic_visit(node)

    _Marker().visit(tree)

    class _Replacer(ast.NodeTransformer):
        def visit(self, node):
            if id(node) == old_id:
                return ast.fix_missing_locations(ast.copy_location(
                    copy.deepcopy(new_node), node
                ))
            return self.generic_visit(node)

    new_tree = copy.deepcopy(tree)
    # Now re-walk to find the equivalent node in the deep copy.
    # Since deepcopy preserves structure, walk both in lockstep.
    _replace_in_lockstep(tree, new_tree, old_id, new_node)
    return new_tree


def _replace_in_lockstep(orig: ast.AST, copied: ast.AST, target_id: int, replacement: ast.AST):
    """Walk orig and copied trees in lockstep; when we find the node with
    target_id in orig, replace the corresponding node in copied's parent."""
    for field_name, orig_value in ast.iter_fields(orig):
        copied_value = getattr(copied, field_name, None)
        if isinstance(orig_value, list) and isinstance(copied_value, list):
            for i, (ov, cv) in enumerate(zip(orig_value, copied_value)):
                if isinstance(ov, ast.AST):
                    if id(ov) == target_id:
                        new = ast.fix_missing_locations(
                            ast.copy_location(copy.deepcopy(replacement), cv)
                        )
                        copied_value[i] = new
                    else:
                        _replace_in_lockstep(ov, cv, target_id, replacement)
        elif isinstance(orig_value, ast.AST) and isinstance(copied_value, ast.AST):
            if id(orig_value) == target_id:
                setattr(copied, field_name, ast.fix_missing_locations(
                    ast.copy_location(copy.deepcopy(replacement), copied_value)
                ))
            else:
                _replace_in_lockstep(orig_value, copied_value, target_id, replacement)


def _delete_stmt_in_body(tree: ast.Module, func_node: ast.FunctionDef,
                          body_stmts: list[ast.stmt], stmt_idx: int,
                          full_body: list[ast.stmt]) -> ast.Module:
    """Delete a statement from the function body. If it's the last, replace with pass.
    Returns a deep-copied tree."""
    new_tree = copy.deepcopy(tree)
    # Find the function in the new tree
    for node in ast.walk(new_tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_node.name:
            # Determine the index in the full body
            target_stmt = body_stmts[stmt_idx]
            # Find matching index in full body
            full_idx = None
            for fi, fs in enumerate(full_body):
                # Match by line/col
                if (getattr(fs, 'lineno', None) == getattr(target_stmt, 'lineno', None) and
                    getattr(fs, 'col_offset', None) == getattr(target_stmt, 'col_offset', None)):
                    full_idx = fi
                    break
            if full_idx is None:
                # Fallback: try offset from end
                docstring_offset = len(full_body) - len(body_stmts)
                full_idx = docstring_offset + stmt_idx

            # Find the same position in the new tree's function body
            new_body = node.body
            if full_idx < len(new_body):
                del new_body[full_idx]
            if not new_body or (len(new_body) == 1 and isinstance(new_body[0], ast.Expr) and
                                isinstance(new_body[0].value, (ast.Constant, ast.Str))):
                # Only docstring left or empty — insert pass
                pass_node = ast.Pass()
                ast.copy_location(pass_node, target_stmt)
                new_body.append(pass_node)
            ast.fix_missing_locations(new_tree)
            return new_tree
    raise ValueError("Could not find function in copied tree")


# ── Operator base ─────────────────────────────────────────

class MutationOperator:
    name: str = ""
    code: str = ""

    def enumerate_mutations(self, tree: ast.Module, entry_point: str) -> list[Mutation]:
        raise NotImplementedError


# ── Arithmetic ────────────────────────────────────────────

_ARITH_OPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
}
_ARITH_OP_TYPES = list(_ARITH_OPS.keys())


class AOR(MutationOperator):
    name = "ArithmeticOperatorReplacement"
    code = "AOR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
                # Skip operator replacement for sequence/string repetition
                # patterns like [1,2] * 2 or "a" * 3 as these are not
                # numeric multiplications and changing the operator would be
                # semantically incorrect.
                if isinstance(node.op, ast.Mult):
                    left_seq = isinstance(node.left, (ast.List, ast.Tuple, ast.Set))
                    right_seq = isinstance(node.right, (ast.List, ast.Tuple, ast.Set))
                    left_str = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                    right_str = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                    left_is_int = isinstance(node.left, ast.Constant) and isinstance(node.left.value, int)
                    right_is_int = isinstance(node.right, ast.Constant) and isinstance(node.right.value, int)
                    if ((left_seq or left_str) and right_is_int) or ((right_seq or right_str) and left_is_int):
                        continue
                orig_type = type(node.op)
                repl_idx = 0
                any_produced = False
                for alt_type in _ARITH_OP_TYPES:
                    if alt_type == orig_type:
                        continue
                    new_op = alt_type()
                    new_node = ast.BinOp(left=node.left, op=new_op, right=node.right)
                    new_tree = _deep_replace(tree, node, new_node)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type="BinOp",
                            original=_ARITH_OPS[orig_type],
                            replacement=_ARITH_OPS[alt_type],
                        ),
                        tree=new_tree,
                    ))
                    repl_idx += 1
                    any_produced = True
                if any_produced:
                    site_idx += 1
        return mutations


class AOD(MutationOperator):
    """Remove one operand from a binary arithmetic expression."""
    name = "ArithmeticOperatorDeletion"
    code = "AOD"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
                # Special-case multiplication used for sequence/string repetition
                # (e.g. [1,2] * 2, "a" * 3): if detected, only keep the
                # non-int operand (the sequence) as the deletion replacement.
                if isinstance(node.op, ast.Mult):
                    left_seq = isinstance(node.left, (ast.List, ast.Tuple, ast.Set))
                    right_seq = isinstance(node.right, (ast.List, ast.Tuple, ast.Set))
                    left_str = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                    right_str = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                    left_is_int = isinstance(node.left, ast.Constant) and isinstance(node.left.value, int)
                    right_is_int = isinstance(node.right, ast.Constant) and isinstance(node.right.value, int)

                    if (left_seq or left_str) and right_is_int:
                        # keep left (sequence)
                        new_tree_l = _deep_replace(tree, node, node.left)
                        mutations.append(Mutation(
                            operator=self.name, operator_code=self.code,
                            site_index=site_idx, replacement_index=0,
                            site=MutationSite(
                                line=node.lineno, col=node.col_offset,
                                node_type="BinOp",
                                original=f"a {_ARITH_OPS[type(node.op)]} b",
                                replacement="a",
                            ),
                            tree=new_tree_l,
                        ))
                        site_idx += 1
                        continue
                    if (right_seq or right_str) and left_is_int:
                        # keep right (sequence)
                        new_tree_r = _deep_replace(tree, node, node.right)
                        mutations.append(Mutation(
                            operator=self.name, operator_code=self.code,
                            site_index=site_idx, replacement_index=0,
                            site=MutationSite(
                                line=node.lineno, col=node.col_offset,
                                node_type="BinOp",
                                original=f"a {_ARITH_OPS[type(node.op)]} b",
                                replacement="b",
                            ),
                            tree=new_tree_r,
                        ))
                        site_idx += 1
                        continue

                # Default behavior: produce both left- and right- deletion variants
                new_tree_l = _deep_replace(tree, node, node.left)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="BinOp",
                        original=f"a {_ARITH_OPS[type(node.op)]} b",
                        replacement="a",
                    ),
                    tree=new_tree_l,
                ))
                new_tree_r = _deep_replace(tree, node, node.right)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=1,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="BinOp",
                        original=f"a {_ARITH_OPS[type(node.op)]} b",
                        replacement="b",
                    ),
                    tree=new_tree_r,
                ))
                site_idx += 1
        return mutations


class AOI(MutationOperator):
    """Insert arithmetic op on a variable/value: a -> a + 1, a -> a - 1."""
    name = "ArithmeticOperatorInsertion"
    code = "AOI"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Keep this operator conservative: do not mutate names that are
                # used as call targets, since "sum(coins)" -> "(sum + 1)(coins)"
                # is not a meaningful arithmetic change.
                if any(
                    isinstance(parent, ast.Call) and parent.func is node
                    for stmt in body
                    for parent in ast.walk(stmt)
                ):
                    continue
                # a -> a + 1
                new_add = ast.BinOp(
                    left=copy.deepcopy(node),
                    op=ast.Add(),
                    right=ast.Constant(value=1),
                )
                new_tree_add = _deep_replace(tree, node, new_add)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Name",
                        original=node.id,
                        replacement=f"{node.id} + 1",
                    ),
                    tree=new_tree_add,
                ))
                # a -> a - 1
                new_sub = ast.BinOp(
                    left=copy.deepcopy(node),
                    op=ast.Sub(),
                    right=ast.Constant(value=1),
                )
                new_tree_sub = _deep_replace(tree, node, new_sub)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=1,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Name",
                        original=node.id,
                        replacement=f"{node.id} - 1",
                    ),
                    tree=new_tree_sub,
                ))
                site_idx += 1
        return mutations


# ── Relational ────────────────────────────────────────────

_CMP_OPS = {
    ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=",
}
_CMP_OP_TYPES = list(_CMP_OPS.keys())


class ROR(MutationOperator):
    name = "RelationalOperatorReplacement"
    code = "ROR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Compare):
                for cmp_idx, op in enumerate(node.ops):
                    if type(op) not in _CMP_OPS:
                        continue
                    orig_type = type(op)
                    repl_idx = 0
                    for alt_type in _CMP_OP_TYPES:
                        if alt_type == orig_type:
                            continue
                        new_ops = list(node.ops)
                        new_ops[cmp_idx] = alt_type()
                        new_node = ast.Compare(
                            left=node.left,
                            ops=new_ops,
                            comparators=list(node.comparators),
                        )
                        new_tree = _deep_replace(tree, node, new_node)
                        mutations.append(Mutation(
                            operator=self.name, operator_code=self.code,
                            site_index=site_idx, replacement_index=repl_idx,
                            site=MutationSite(
                                line=node.lineno, col=node.col_offset,
                                node_type="Compare",
                                original=_CMP_OPS[orig_type],
                                replacement=_CMP_OPS[alt_type],
                            ),
                            tree=new_tree,
                        ))
                        repl_idx += 1
                    site_idx += 1
        return mutations


# ── Conditional ───────────────────────────────────────────

class COR(MutationOperator):
    """Replace and <-> or."""
    name = "ConditionalOperatorReplacement"
    code = "COR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BoolOp):
                orig_type = type(node.op)
                if orig_type == ast.And:
                    new_op = ast.Or()
                    orig_str, repl_str = "and", "or"
                elif orig_type == ast.Or:
                    new_op = ast.And()
                    orig_str, repl_str = "or", "and"
                else:
                    continue
                new_node = ast.BoolOp(op=new_op, values=list(node.values))
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="BoolOp",
                        original=orig_str,
                        replacement=repl_str,
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


class COD(MutationOperator):
    """Remove one clause from a compound boolean (and/or with 2+ values)."""
    name = "ConditionalOperatorDeletion"
    code = "COD"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BoolOp) and len(node.values) >= 2:
                for val_idx in range(len(node.values)):
                    remaining = [v for i, v in enumerate(node.values) if i != val_idx]
                    if len(remaining) == 1:
                        replacement = remaining[0]
                    else:
                        replacement = ast.BoolOp(op=copy.deepcopy(node.op), values=remaining)
                    new_tree = _deep_replace(tree, node, replacement)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=val_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type="BoolOp",
                            original=ast.dump(node.values[val_idx]),
                            replacement="<removed>",
                        ),
                        tree=new_tree,
                    ))
                site_idx += 1
        return mutations


# ── Unary ─────────────────────────────────────────────────

class UOI(MutationOperator):
    """Insert a unary operator: -x or not x."""
    name = "UnaryOperatorInsertion"
    code = "UOI"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Keep unary insertion conservative: do not rewrite names that
                # are being used as call targets, since "sum(x)" -> "(not sum)(x)"
                # or "(-sum)(x)" is not a meaningful mutation.
                if any(
                    isinstance(parent, ast.Call) and parent.func is node
                    for stmt in body
                    for parent in ast.walk(stmt)
                ):
                    continue
                # -x
                neg = ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(node))
                new_tree_neg = _deep_replace(tree, node, neg)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Name",
                        original=node.id,
                        replacement=f"-{node.id}",
                    ),
                    tree=new_tree_neg,
                ))
                # not x
                not_node = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node))
                new_tree_not = _deep_replace(tree, node, not_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=1,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Name",
                        original=node.id,
                        replacement=f"not {node.id}",
                    ),
                    tree=new_tree_not,
                ))
                site_idx += 1
        return mutations


class UOD(MutationOperator):
    """Remove a unary operator: -x -> x, not x -> x."""
    name = "UnaryOperatorDeletion"
    code = "UOD"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.Not, ast.UAdd, ast.Invert)):
                op_str = {ast.USub: "-", ast.Not: "not", ast.UAdd: "+", ast.Invert: "~"}[type(node.op)]
                new_tree = _deep_replace(tree, node, node.operand)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="UnaryOp",
                        original=op_str,
                        replacement="<removed>",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Constants ─────────────────────────────────────────────

class CR(MutationOperator):
    """Replace a numeric/string/bool literal with an alternative."""
    name = "ConstantReplacement"
    code = "CR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if not isinstance(node, ast.Constant):
                continue
            replacements = self._alternatives(node.value)
            if not replacements:
                continue
            for repl_idx, new_val in enumerate(replacements):
                new_node = ast.Constant(value=new_val)
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=repl_idx,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Constant",
                        original=repr(node.value),
                        replacement=repr(new_val),
                    ),
                    tree=new_tree,
                ))
            site_idx += 1
        return mutations

    @staticmethod
    def _alternatives(val):
        if isinstance(val, bool):
            return [not val]
        if isinstance(val, int):
            alts = []
            if val != 0:
                alts.append(0)
            if val != 1:
                alts.append(1)
            if val != -1:
                alts.append(-1)
            return alts
        if isinstance(val, float):
            alts = []
            if val != 0.0:
                alts.append(0.0)
            if val != 1.0:
                alts.append(1.0)
            return alts
        if isinstance(val, str):
            if val != "":
                return [""]
            return ["mutated_string"]
        return []


# ── Variable replacement ──────────────────────────────────

class VVR(MutationOperator):
    """Replace a variable with another in-scope variable."""
    name = "VarToVarReplacement"
    code = "VVR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        in_scope = _in_scope_variables(func)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in in_scope:
                others = [v for v in in_scope if v != node.id]
                if not others:
                    continue
                for repl_idx, other in enumerate(others):
                    new_node = ast.Name(id=other, ctx=ast.Load())
                    new_tree = _deep_replace(tree, node, new_node)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type="Name",
                            original=node.id,
                            replacement=other,
                        ),
                        tree=new_tree,
                    ))
                site_idx += 1
        return mutations


class VCR(MutationOperator):
    """Replace a variable reference with a constant."""
    name = "VarToConsReplacement"
    code = "VCR"

    _CONSTANTS = [0, 1, None, True, False, ""]

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                for repl_idx, val in enumerate(self._CONSTANTS):
                    new_node = ast.Constant(value=val)
                    new_tree = _deep_replace(tree, node, new_node)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type="Name",
                            original=node.id,
                            replacement=repr(val),
                        ),
                        tree=new_tree,
                    ))
                site_idx += 1
        return mutations


class CVR(MutationOperator):
    """Replace a constant with an in-scope variable."""
    name = "ConsToVarReplacement"
    code = "CVR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        in_scope = _in_scope_variables(func)
        if not in_scope:
            return []
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
                for repl_idx, var in enumerate(in_scope):
                    new_node = ast.Name(id=var, ctx=ast.Load())
                    new_tree = _deep_replace(tree, node, new_node)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type="Constant",
                            original=repr(node.value),
                            replacement=var,
                        ),
                        tree=new_tree,
                    ))
                site_idx += 1
        return mutations


# ── Statement deletion ────────────────────────────────────

class SD(MutationOperator):
    """Delete a non-signature statement from the function body."""
    name = "StatementDeletion"
    code = "SD"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        full_body = func.body
        mutations = []
        for site_idx, stmt in enumerate(body):
            new_tree = _delete_stmt_in_body(tree, func, body, site_idx, full_body)
            mutations.append(Mutation(
                operator=self.name, operator_code=self.code,
                site_index=site_idx, replacement_index=0,
                site=MutationSite(
                    line=getattr(stmt, 'lineno', 0),
                    col=getattr(stmt, 'col_offset', 0),
                    node_type=type(stmt).__name__,
                    original=ast.dump(stmt),
                    replacement="<deleted>",
                ),
                tree=new_tree,
            ))
        return mutations


# ── Return value mutations ────────────────────────────────

class RVM(MutationOperator):
    """Alter a return value (constant change, negate)."""
    name = "ReturnValueMutation"
    code = "RVM"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Return) and node.value is not None:
                # Negate: return x -> return not x
                negated = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node.value))
                new_ret = ast.Return(value=negated)
                new_tree = _deep_replace(tree, node, new_ret)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Return",
                        original="return <expr>",
                        replacement="return not <expr>",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


class TR(MutationOperator):
    """Replace return expression with True."""
    name = "TrueReturn"
    code = "TR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Return) and node.value is not None:
                new_ret = ast.Return(value=ast.Constant(value=True))
                new_tree = _deep_replace(tree, node, new_ret)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Return",
                        original="return <expr>",
                        replacement="return True",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


class FR(MutationOperator):
    """Replace return expression with False."""
    name = "FalseReturn"
    code = "FR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Return) and node.value is not None:
                new_ret = ast.Return(value=ast.Constant(value=False))
                new_tree = _deep_replace(tree, node, new_ret)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Return",
                        original="return <expr>",
                        replacement="return False",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Condition mutations ───────────────────────────────────

class RC(MutationOperator):
    """Replace a condition with True."""
    name = "RemoveConditional"
    code = "RC"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.If):
                new_node = ast.If(
                    test=ast.Constant(value=True),
                    body=copy.deepcopy(node.body),
                    orelse=copy.deepcopy(node.orelse),
                )
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="If",
                        original=ast.dump(node.test),
                        replacement="True",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
            elif isinstance(node, ast.While):
                new_node = ast.While(
                    test=ast.Constant(value=True),
                    body=copy.deepcopy(node.body),
                    orelse=copy.deepcopy(node.orelse),
                )
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="While",
                        original=ast.dump(node.test),
                        replacement="True",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Negation ──────────────────────────────────────────────

class NEG(MutationOperator):
    """Negate a numeric value: x -> -x."""
    name = "Negation"
    code = "NEG"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value != 0:
                new_node = ast.Constant(value=-node.value)
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Constant",
                        original=repr(node.value),
                        replacement=repr(-node.value),
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Function call deletion ────────────────────────────────

class FCD(MutationOperator):
    """Delete a function/method call statement (Expr(Call(...)))."""
    name = "FunctionCallDeletion"
    code = "FCD"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        full_body = func.body
        mutations = []
        site_idx = 0
        for stmt_idx, stmt in enumerate(body):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                new_tree = _delete_stmt_in_body(tree, func, body, stmt_idx, full_body)
                call_name = ""
                if isinstance(stmt.value.func, ast.Name):
                    call_name = stmt.value.func.id
                elif isinstance(stmt.value.func, ast.Attribute):
                    call_name = stmt.value.func.attr
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=stmt.lineno, col=stmt.col_offset,
                        node_type="Call",
                        original=call_name,
                        replacement="<deleted>",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Constructor replacement ───────────────────────────────

class CRN(MutationOperator):
    """Replace a constructor call with None."""
    name = "ConstructorReplacementNone"
    code = "CRN"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Call):
                # Heuristic: Name starting with uppercase = constructor
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if not name or not name[0].isupper():
                    continue
                new_node = ast.Constant(value=None)
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Call",
                        original=f"{name}(...)",
                        replacement="None",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Bitwise ───────────────────────────────────────────────

_BIT_OPS = {
    ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
}
_BIT_OP_TYPES = list(_BIT_OPS.keys())


class BOR(MutationOperator):
    """Replace |, &, ^ with each alternative."""
    name = "BitwiseOperatorReplacement"
    code = "BOR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _BIT_OPS:
                orig_type = type(node.op)
                repl_idx = 0
                any_produced = False
                for alt_type in _BIT_OP_TYPES:
                    if alt_type == orig_type:
                        continue
                    new_op = alt_type()
                    new_node = ast.BinOp(left=node.left, op=new_op, right=node.right)
                    new_tree = _deep_replace(tree, node, new_node)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type="BinOp",
                            original=_BIT_OPS[orig_type],
                            replacement=_BIT_OPS[alt_type],
                        ),
                        tree=new_tree,
                    ))
                    repl_idx += 1
                    any_produced = True
                if any_produced:
                    site_idx += 1
        return mutations


_SHIFT_OPS = {
    ast.LShift: "<<", ast.RShift: ">>",
}


class BSOR(MutationOperator):
    """Replace << <-> >>."""
    name = "BitshiftOperatorReplacement"
    code = "BSOR"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _SHIFT_OPS:
                orig_type = type(node.op)
                alt_type = ast.RShift if orig_type == ast.LShift else ast.LShift
                new_node = ast.BinOp(left=node.left, op=alt_type(), right=node.right)
                new_tree = _deep_replace(tree, node, new_node)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="BinOp",
                        original=_SHIFT_OPS[orig_type],
                        replacement=_SHIFT_OPS[type(alt_type())],
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


class BSOD(MutationOperator):
    """Remove the shift, keeping the left operand."""
    name = "BitshiftOperatorDeletion"
    code = "BSOD"

    def enumerate_mutations(self, tree, entry_point):
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _SHIFT_OPS:
                new_tree = _deep_replace(tree, node, node.left)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="BinOp",
                        original=f"a {_SHIFT_OPS[type(node.op)]} b",
                        replacement="a",
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


# ── Registry ──────────────────────────────────────────────

ALL_OPERATORS: dict[str, type[MutationOperator]] = {
    "AOR": AOR, "AOI": AOI, "AOD": AOD,
    "ROR": ROR, "COR": COR, "COD": COD,
    "UOI": UOI, "UOD": UOD,
    "CR": CR, "VVR": VVR, "VCR": VCR, "CVR": CVR,
    "SD": SD, "RVM": RVM, "TR": TR, "FR": FR,
    "RC": RC, "NEG": NEG, "FCD": FCD, "CRN": CRN,
    "BOR": BOR, "BSOR": BSOR, "BSOD": BSOD,
}


def get_operators(include: list[str]) -> list[MutationOperator]:
    ops = []
    for code in include:
        if code not in ALL_OPERATORS:
            raise ValueError(f"Unknown operator code: {code}")
        ops.append(ALL_OPERATORS[code]())
    return ops
