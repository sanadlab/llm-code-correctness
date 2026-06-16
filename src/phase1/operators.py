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

import astroid
from astroid import Uninferable, nodes as astroid_nodes
from astroid.exceptions import InferenceError

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


def _constant_in_assignment_rhs(node: ast.Constant, func_body: list[ast.stmt]) -> bool:
    """True when a literal appears in the RHS of a plain assignment (defining initializer)."""
    for stmt in func_body:
        if isinstance(stmt, ast.Assign):
            for child in ast.walk(stmt.value):
                if child is node:
                    return True
    return False


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


# ── Astroid type inference ─────────────────────────────────

Coord = tuple[int, int]
ReturnCoord = tuple[str, int]
TypeCoord = Coord | ReturnCoord

_PRIMITIVE_PYTYPES: dict[str, str] = {
    "builtins.int": "int",
    "builtins.float": "float",
    "builtins.str": "str",
    "builtins.bool": "bool",
    "builtins.list": "list",
    "builtins.dict": "dict",
    "builtins.tuple": "tuple",
    "builtins.NoneType": "NoneType",
}

_SEQUENCE_TYPES = frozenset({"str", "list", "tuple"})

_RETURN_COORD: ReturnCoord = ("__return__", 0)


def _resolve_inferred_type(inferred: Any) -> str | None:
    """Map an astroid inference result to a primitive type string, or None."""
    if inferred is Uninferable or inferred is None:
        return None

    if isinstance(inferred, astroid_nodes.Const):
        val = inferred.value
        if val is None:
            return "NoneType"
        if type(val) is bool:
            return "bool"
        if type(val) is int:
            return "int"
        if type(val) is float:
            return "float"
        if type(val) is str:
            return "str"
        return None

    pytype_fn = getattr(inferred, "pytype", None)
    if pytype_fn is None:
        return None
    pytype = pytype_fn()
    return _PRIMITIVE_PYTYPES.get(pytype)


def _infer_node_type(node: astroid_nodes.NodeNG) -> str | None:
    """Infer a node's type via astroid; return None if uninferable."""
    try:
        inferred = next(node.infer())
    except (StopIteration, InferenceError, AttributeError, TypeError):
        return None
    except Exception:
        # Astroid can raise on some parameter / scope edge cases; skip them.
        return None
    return _resolve_inferred_type(inferred)


def _annotation_for_arg(arg: astroid_nodes.AssignName) -> astroid_nodes.NodeNG | None:
    """Return the annotation node for a function parameter, if any."""
    annotation = getattr(arg, "annotation", None)
    if annotation is not None:
        return annotation

    parent = arg.parent
    if not isinstance(parent, astroid_nodes.Arguments):
        return None

    args_node = parent
    if arg in args_node.arguments:
        idx = args_node.arguments.index(arg)
        if idx < len(args_node.annotations):
            return args_node.annotations[idx]
    if arg in args_node.kwonlyargs:
        idx = args_node.kwonlyargs.index(arg)
        if idx < len(args_node.kwonlyargs_annotations):
            return args_node.kwonlyargs_annotations[idx]
    if args_node.vararg == arg.name and args_node.varargannotation is not None:
        return args_node.varargannotation
    if args_node.kwarg == arg.name and args_node.kwargannotation is not None:
        return args_node.kwargannotation
    return None


def _infer_arg_type(arg: astroid_nodes.AssignName) -> str | None:
    """Infer a parameter type from its annotation, avoiding fragile arg.infer()."""
    annotation = _annotation_for_arg(arg)
    if annotation is None:
        return None
    if isinstance(annotation, astroid_nodes.Name):
        return _PRIMITIVE_PYTYPES.get(f"builtins.{annotation.name}")
    if isinstance(annotation, astroid_nodes.Const):
        return _resolve_inferred_type(annotation)
    return _infer_node_type(annotation)


def _node_coord(node: astroid_nodes.NodeNG) -> Coord | None:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    if lineno is None or col_offset is None:
        return None
    return (lineno, col_offset)


def _register_var_type(
    var_to_type: dict[str, str],
    type_to_vars: dict[str, list[str]],
    name: str,
    typ: str | None,
) -> None:
    """Record a variable's inferred type in both var_to_type and type_to_vars."""
    if typ is None or not name:
        return
    existing = var_to_type.get(name)
    if existing is not None and existing != typ:
        # Conflicting inferences — drop the variable from both maps.
        _remove_var_type(var_to_type, type_to_vars, name)
        return
    var_to_type[name] = typ
    bucket = type_to_vars.setdefault(typ, [])
    if name not in bucket:
        bucket.append(name)


def _remove_var_type(
    var_to_type: dict[str, str],
    type_to_vars: dict[str, list[str]],
    name: str,
) -> None:
    old_typ = var_to_type.pop(name, None)
    if old_typ is None:
        return
    bucket = type_to_vars.get(old_typ)
    if bucket is not None:
        type_to_vars[old_typ] = [v for v in bucket if v != name]
        if not type_to_vars[old_typ]:
            del type_to_vars[old_typ]


def _infer_assign_name_type(node: astroid_nodes.AssignName) -> str | None:
    """Infer type of an assignment target from itself or its RHS."""
    typ = _infer_node_type(node)
    if typ is not None:
        return typ
    parent = node.parent
    if isinstance(parent, astroid_nodes.Assign):
        return _infer_node_type(parent.value)
    return None


def _infer_return_type(
    func: astroid_nodes.FunctionDef,
    var_to_type: dict[str, str] | None = None,
) -> str | None:
    return_types: list[str] = []
    for ret in func.nodes_of_class(astroid_nodes.Return):
        if ret.value is None:
            continue
        typ = _infer_node_type(ret.value)
        if typ is None and var_to_type and isinstance(ret.value, astroid_nodes.Name):
            typ = var_to_type.get(ret.value.name)
        if typ is not None:
            return_types.append(typ)
    if not return_types:
        return None
    unique = set(return_types)
    if len(unique) == 1:
        return return_types[0]
    return None


def _build_type_maps(
    source_code: str, entry_point: str,
) -> tuple[dict[TypeCoord, str], dict[str, list[str]], dict[str, str]]:
    """Build coordinate and variable-type lookup tables via astroid inference.

    Returns:
        coord_to_type: maps (lineno, col_offset) -> primitive type string
        type_to_vars: maps type string -> sorted list of in-scope variable names
        var_to_type: maps variable name -> primitive type string
    """
    coord_to_type: dict[TypeCoord, str] = {}
    type_to_vars: dict[str, list[str]] = {}
    var_to_type: dict[str, str] = {}

    try:
        module = astroid.parse(source_code)
    except Exception:
        return coord_to_type, type_to_vars, var_to_type

    func: astroid_nodes.FunctionDef | None = None
    for node in module.body:
        if isinstance(node, astroid_nodes.FunctionDef) and node.name == entry_point:
            func = node
            break
    if func is None:
        return coord_to_type, type_to_vars, var_to_type

    for arg in func.args.arguments:
        typ = _infer_arg_type(arg)
        if typ is None:
            continue
        coord = _node_coord(arg)
        if coord is not None:
            coord_to_type[coord] = typ
        _register_var_type(var_to_type, type_to_vars, arg.name, typ)

    target_classes = (astroid_nodes.Name, astroid_nodes.AssignName, astroid_nodes.Call, astroid_nodes.Const)
    for node in func.nodes_of_class(target_classes):
        if isinstance(node, astroid_nodes.AssignName):
            typ = _infer_assign_name_type(node)
        else:
            typ = _infer_node_type(node)
        if typ is None:
            continue
        coord = _node_coord(node)
        if coord is not None:
            coord_to_type[coord] = typ
        if isinstance(node, astroid_nodes.AssignName):
            _register_var_type(var_to_type, type_to_vars, node.name, typ)

    for node in func.nodes_of_class(astroid_nodes.Name):
        coord = _node_coord(node)
        if coord is None or coord in coord_to_type:
            continue
        typ = var_to_type.get(node.name)
        if typ is not None:
            coord_to_type[coord] = typ

    return_type = _infer_return_type(func, var_to_type)
    if return_type is not None:
        coord_to_type[_RETURN_COORD] = return_type

    for typ in type_to_vars:
        type_to_vars[typ].sort()

    return coord_to_type, type_to_vars, var_to_type


def _ast_coord(node: ast.AST) -> Coord:
    return (node.lineno, node.col_offset)


def _call_target_ids(body: list[ast.stmt]) -> set[int]:
    """AST node ids for Name nodes used as the callable in a Call."""
    return {
        id(node.func)
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _is_user_variable(name: str, in_scope: set[str]) -> bool:
    """True when name is a local/parameter variable, not a builtin or global."""
    return name in in_scope


def _operand_primitive_type(
    operand: ast.AST,
    coord_to_type: dict[TypeCoord, str],
    var_to_type: dict[str, str] | None = None,
) -> str | None:
    typ = coord_to_type.get(_ast_coord(operand))
    if typ is None and var_to_type and isinstance(operand, ast.Name):
        typ = var_to_type.get(operand.id)
    return typ


def _binop_operands_type_ok(
    node: ast.BinOp,
    coord_to_type: dict[TypeCoord, str],
    var_to_type: dict[str, str] | None = None,
) -> bool:
    """True when both BinOp operands are inferable and not sequence types."""
    left_type = _operand_primitive_type(node.left, coord_to_type, var_to_type)
    right_type = _operand_primitive_type(node.right, coord_to_type, var_to_type)
    if left_type is None or right_type is None:
        return False
    if left_type in _SEQUENCE_TYPES or right_type in _SEQUENCE_TYPES:
        return False
    return True


def _constant_primitive_type(val: Any) -> str | None:
    if val is None:
        return "NoneType"
    if type(val) is bool:
        return "bool"
    if type(val) is int:
        return "int"
    if type(val) is float:
        return "float"
    if type(val) is str:
        return "str"
    return None


# ── Operator base ─────────────────────────────────────────

class MutationOperator:
    name: str = ""
    code: str = ""

    def enumerate_mutations(
        self, tree: ast.Module, entry_point: str, source_code: str,
    ) -> list[Mutation]:
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, var_to_type = _build_type_maps(source_code, entry_point)
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
                if not _binop_operands_type_ok(node, coord_to_type, var_to_type):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, var_to_type = _build_type_maps(source_code, entry_point)
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
                if not _binop_operands_type_ok(node, coord_to_type, var_to_type):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        # 1. Fetch the type maps
        coord_to_type, _, var_to_type = _build_type_maps(source_code, entry_point)
        
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # 2. Check if the type is strictly numeric
                node_type = coord_to_type.get(_ast_coord(node)) or var_to_type.get(node.id)
                if node_type not in {"int", "float"}:
                    continue
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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
    """Insert a unary operator: -x or not x.
    
    The numeric negation (-x) is restricted strictly to variables inferred
    as an 'int' or 'float' type. The boolean negation (not x) is restricted
    to 'bool' types or attributes matching common boolean naming conventions.
    """
    name = "UnaryOperatorInsertion"
    code = "UOI"

    def enumerate_mutations(self, tree, entry_point, source_code):
        # Build type maps using the existing astroid inference helper
        coord_to_type, _, var_to_type = _build_type_maps(source_code, entry_point)
        
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0

        # PRE-COMPUTE: Find all nodes that are being called as functions
        # This prevents O(N^2) tree walking inside the main loop.
        call_target_ids = {
            id(node.func) 
            for stmt in body 
            for node in ast.walk(stmt) 
            if isinstance(node, ast.Call)
        }
        
        # PRE-COMPUTE: Find all nodes that are the 'value' of an attribute lookup
        # This prevents mutating the base object (e.g., 'numbers' in 'numbers.empty')
        attribute_value_ids = {
            id(node.value)
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Attribute)
        }
        
        # PRE-COMPUTE: clauses used as condition tests in if/while statements.
        # For `if a and b`, treat `a` and `b` as the clause units.
        condition_clause_ids = set()
        for stmt in body:
            for control in ast.walk(stmt):
                if not isinstance(control, (ast.If, ast.While)):
                    continue
                test = control.test
                if isinstance(test, ast.BoolOp):
                    for clause in test.values:
                        condition_clause_ids.add(id(clause))
                else:
                    condition_clause_ids.add(id(test))

        for node in _body_nodes_preorder(body):
            # Target standalone variable/property loads and condition clauses.
            is_load_name_or_attr = (
                isinstance(node, (ast.Name, ast.Attribute))
                and isinstance(getattr(node, "ctx", None), ast.Load)
            )
            is_condition_clause = id(node) in condition_clause_ids
            if is_load_name_or_attr or is_condition_clause:
                
                # O(1) lookup: skip if this exact node is a call target or an attribute base
                if id(node) in call_target_ids or id(node) in attribute_value_ids:
                    continue
                
                repl_idx = 0
                any_produced = False
                
                # Safely unparse the node for the mutation report string
                node_str = ast.unparse(node) if hasattr(ast, "unparse") else (getattr(node, "id", None) or getattr(node, "attr", "<attr>"))
                
                # Fetch inferred type from astroid maps
                node_type = coord_to_type.get(_ast_coord(node))
                if not node_type and hasattr(node, "id"):
                    node_type = var_to_type.get(node.id)
                
                # ── Case 1: -x (Numeric types only) ─────────────────
                if node_type in {"int", "float"}:
                    neg = ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(node))
                    new_tree_neg = _deep_replace(tree, node, neg)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type=type(node).__name__,
                            original=node_str,
                            replacement=f"-{node_str}",
                        ),
                        tree=new_tree_neg,
                    ))
                    repl_idx += 1
                    any_produced = True
                
                # ── Case 2: not x (booleans + condition clauses) ──────────
                is_boolean = (node_type == "bool")
                
                # Heuristic fallback for third-party libraries (e.g., pandas df.empty)
                # where astroid often yields Uninferable.
                if not node_type and isinstance(node, ast.Attribute):
                    if node.attr == "empty" or node.attr.startswith(("is_", "has_", "can_", "should_")):
                        is_boolean = True

                if is_boolean or is_condition_clause:
                    not_node = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node))
                    new_tree_not = _deep_replace(tree, node, not_node)
                    mutations.append(Mutation(
                        operator=self.name, operator_code=self.code,
                        site_index=site_idx, replacement_index=repl_idx,
                        site=MutationSite(
                            line=node.lineno, col=node.col_offset,
                            node_type=type(node).__name__,
                            original=node_str,
                            replacement=f"not {node_str}",
                        ),
                        tree=new_tree_not,
                    ))
                    repl_idx += 1
                    any_produced = True
                
                if any_produced:
                    site_idx += 1
                    
        return mutations
        
class UOD(MutationOperator):
    """Remove a unary operator: -x -> x, not x -> x."""
    name = "UnaryOperatorDeletion"
    code = "UOD"

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, _ = _build_type_maps(source_code, entry_point)
        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if not isinstance(node, ast.Constant):
                continue
            coord = _ast_coord(node)
            inferred_type = coord_to_type.get(coord)
            if inferred_type is None:
                continue
            literal_type = _constant_primitive_type(node.value)
            if literal_type is None or literal_type != inferred_type:
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
        if type(val) is bool:
            return [not val]
        if type(val) is int:
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, type_to_vars, var_to_type = _build_type_maps(source_code, entry_point)
        func, body = _get_function_body(tree, entry_point)
        in_scope = set(_in_scope_variables(func))
        call_targets = _call_target_ids(body)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if not _is_user_variable(node.id, in_scope):
                    continue
                if id(node) in call_targets:
                    continue
                var_type = coord_to_type.get(_ast_coord(node)) or var_to_type.get(node.id)
                if var_type is None:
                    continue
                others = [
                    v for v in type_to_vars.get(var_type, [])
                    if v != node.id and _is_user_variable(v, in_scope)
                ]
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, var_to_type = _build_type_maps(source_code, entry_point)
        func, body = _get_function_body(tree, entry_point)
        in_scope = set(_in_scope_variables(func))
        call_targets = _call_target_ids(body)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if not _is_user_variable(node.id, in_scope):
                    continue
                if id(node) in call_targets:
                    continue
                var_type = coord_to_type.get(_ast_coord(node)) or var_to_type.get(node.id)
                if var_type is None:
                    continue
                matching_constants = [
                    val for val in self._CONSTANTS
                    if _constant_primitive_type(val) == var_type
                ]
                if not matching_constants:
                    continue
                for repl_idx, val in enumerate(matching_constants):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, type_to_vars, _ = _build_type_maps(source_code, entry_point)
        func, body = _get_function_body(tree, entry_point)
        in_scope = set(_in_scope_variables(func))
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Constant):
                if _constant_in_assignment_rhs(node, body):
                    continue
                const_type = coord_to_type.get(_ast_coord(node))
                if const_type is None:
                    continue
                candidates = [
                    v for v in type_to_vars.get(const_type, [])
                    if _is_user_variable(v, in_scope)
                ]
                if not candidates:
                    continue
                for repl_idx, var in enumerate(candidates):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, _ = _build_type_maps(source_code, entry_point)
        return_type = coord_to_type.get(_RETURN_COORD)
        if return_type not in {"int", "float", "bool"}:
            return []

        func, body = _get_function_body(tree, entry_point)
        mutations = []
        site_idx = 0
        for node in _body_nodes_preorder(body):
            if isinstance(node, ast.Return) and node.value is not None:
                if return_type in {"int", "float"}:
                    replacement = ast.UnaryOp(
                        op=ast.USub(), operand=copy.deepcopy(node.value),
                    )
                    replacement_label = "return -<expr>"
                else:
                    replacement = ast.UnaryOp(
                        op=ast.Not(), operand=copy.deepcopy(node.value),
                    )
                    replacement_label = "return not <expr>"
                new_ret = ast.Return(value=replacement)
                new_tree = _deep_replace(tree, node, new_ret)
                mutations.append(Mutation(
                    operator=self.name, operator_code=self.code,
                    site_index=site_idx, replacement_index=0,
                    site=MutationSite(
                        line=node.lineno, col=node.col_offset,
                        node_type="Return",
                        original="return <expr>",
                        replacement=replacement_label,
                    ),
                    tree=new_tree,
                ))
                site_idx += 1
        return mutations


class TR(MutationOperator):
    """Replace return expression with True."""
    name = "TrueReturn"
    code = "TR"

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, _ = _build_type_maps(source_code, entry_point)
        if coord_to_type.get(_RETURN_COORD) != "bool":
            return []

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

    def enumerate_mutations(self, tree, entry_point, source_code):
        coord_to_type, _, _ = _build_type_maps(source_code, entry_point)
        if coord_to_type.get(_RETURN_COORD) != "bool":
            return []

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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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

    def enumerate_mutations(self, tree, entry_point, source_code):
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
