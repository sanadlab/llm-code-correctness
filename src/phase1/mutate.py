"""Step 3: Apply mutation operators and write mutants.jsonl.

Reads validated_tasks.jsonl, applies all configured operators to each task's
function body, deduplicates identical mutations, and writes mutants.jsonl.
"""

import ast
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from src.phase1.config import Phase1Config
from src.phase1.io_utils import read_jsonl, write_jsonl
from src.phase1.operators import get_operators, _get_function_body


def mutate(cfg: Phase1Config) -> None:
    tasks = read_jsonl(cfg.validated_tasks_path)
    operator_codes = cfg.operators.include
    operators = get_operators(operator_codes)
    mutate_parallel = max(1, int(os.environ.get("MUTATE_PARALLEL", "1")))
    print(
        f"Mutating {len(tasks)} tasks with {len(operators)} operators "
        f"(parallel={mutate_parallel})..."
    )

    all_mutants = []
    if mutate_parallel == 1:
        for task in tasks:
            task_mutants = _mutate_task(task, operators)
            all_mutants.extend(task_mutants)
    else:
        chunksize = max(1, len(tasks) // (mutate_parallel * 4))
        with ProcessPoolExecutor(max_workers=mutate_parallel) as executor:
            for task_id, task_mutants in executor.map(
                _mutate_task_worker,
                tasks,
                repeat(operator_codes),
                chunksize=chunksize,
            ):
                if task_mutants:
                    print(f"  {task_id}: {len(task_mutants)} unique mutants")
                all_mutants.extend(task_mutants)

    write_jsonl(cfg.mutants_path, all_mutants)
    print(f"Wrote {len(all_mutants)} mutants to {cfg.mutants_path}")


def _mutate_task_worker(task: dict, operator_codes: list[str]) -> tuple[str, list[dict]]:
    operators = get_operators(operator_codes)
    task_id = task["task_id"]
    try:
        return task_id, _mutate_task(task, operators, log=False)
    except Exception as exc:
        print(f"  WARN: {task_id} mutation failed: {exc}")
        return task_id, []


def _mutate_task(task: dict, operators, log: bool = True) -> list[dict]:
    task_id = task["task_id"]
    entry_point = task["entry_point"]
    complete_prompt = task["complete_prompt"]
    canonical_solution = task["canonical_solution"]

    full_source = complete_prompt + canonical_solution
    try:
        tree = ast.parse(full_source)
    except SyntaxError:
        print(f"  WARN: {task_id} does not parse, skipping")
        return []

    # Figure out where the canonical_solution starts in the full source
    prompt_lines = complete_prompt.split("\n")
    prompt_line_count = len(prompt_lines)
    # The canonical_solution starts at the line after the last prompt line
    # (accounting for no trailing newline in prompt)
    if complete_prompt.endswith("\n"):
        body_start_line = prompt_line_count + 1
    else:
        body_start_line = prompt_line_count

    mutants = []
    seen_codes = set()  # for dedup

    for op in operators:
        try:
            mutations = op.enumerate_mutations(tree, entry_point, full_source)
        except (ValueError, KeyError) as e:
            # Function not found or other issue — skip operator for this task
            continue

        for mut in mutations:
            # Serialize the mutated tree back to source
            try:
                full_mutated = ast.unparse(mut.tree)
            except Exception:
                continue

            # Verify it parses
            try:
                ast.parse(full_mutated)
            except SyntaxError:
                continue

            # Extract the mutated body, preserving original formatting for
            # unchanged lines (comments, blanks, quote style).
            mutated_body = _extract_body_fragment(
                full_source, tree, mut.tree, entry_point
            )
            if mutated_body is None:
                continue

            # Dedup: skip if identical body for this task
            if mutated_body in seen_codes:
                continue
            # Also skip if same as canonical
            if mutated_body.strip() == canonical_solution.strip():
                continue

            seen_codes.add(mutated_body)

            # Build mutant_id
            if mut.replacement_index == 0 and _is_single_replacement(mutations, mut):
                mutant_id = f"{task_id}__{mut.operator_code}__{mut.site_index}"
            else:
                mutant_id = f"{task_id}__{mut.operator_code}__{mut.site_index}__{mut.replacement_index}"

            mutants.append({
                "task_id": task_id,
                "mutant_id": mutant_id,
                "operator": mut.operator,
                "operator_code": mut.operator_code,
                "site_index": mut.site_index,
                "replacement_index": mut.replacement_index,
                "code": mutated_body,
                "site": {
                    "line": mut.site.line,
                    "col": mut.site.col,
                    "node_type": mut.site.node_type,
                    "original": mut.site.original,
                    "replacement": mut.site.replacement,
                },
            })

    # Sort by mutant_id for determinism
    mutants.sort(key=lambda m: m["mutant_id"])

    # Second dedup pass: keep lexicographically smallest mutant_id per code
    final = {}
    for m in mutants:
        code = m["code"]
        if code not in final or m["mutant_id"] < final[code]["mutant_id"]:
            final[code] = m

    result = sorted(final.values(), key=lambda m: m["mutant_id"])
    if result and log:
        print(f"  {task_id}: {len(result)} unique mutants")
    return result


def _is_single_replacement(mutations, mut) -> bool:
    """Check if this (operator, site_index) has only one replacement."""
    count = sum(1 for m in mutations
                if m.operator_code == mut.operator_code and m.site_index == mut.site_index)
    return count == 1


def _find_function(tree: ast.Module, entry_point: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _body_stmts_after_docstring(func: ast.FunctionDef) -> list[ast.stmt]:
    body = func.body
    start_idx = 0
    if (body and isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
        start_idx = 1
    return body[start_idx:]


def _stmt_dump(stmt: ast.stmt) -> str:
    return ast.dump(stmt, annotate_fields=False, include_attributes=False)


def _classify_body_statements(
    orig_body: list[ast.stmt],
    mut_body: list[ast.stmt],
) -> list[tuple[str, ast.stmt | None, ast.stmt | None]]:
    """Align top-level body statements: keep, delete, modify, or insert."""
    classified: list[tuple[str, ast.stmt | None, ast.stmt | None]] = []
    i, j = 0, 0
    while i < len(orig_body) or j < len(mut_body):
        if i >= len(orig_body):
            classified.append(("insert", None, mut_body[j]))
            j += 1
            continue
        if j >= len(mut_body):
            classified.append(("delete", orig_body[i], None))
            i += 1
            continue

        orig_stmt, mut_stmt = orig_body[i], mut_body[j]
        if _stmt_dump(orig_stmt) == _stmt_dump(mut_stmt):
            classified.append(("keep", orig_stmt, mut_stmt))
            i += 1
            j += 1
        elif (i + 1 < len(orig_body) and
              _stmt_dump(orig_body[i + 1]) == _stmt_dump(mut_stmt)):
            classified.append(("delete", orig_stmt, None))
            i += 1
        else:
            classified.append(("modify", orig_stmt, mut_stmt))
            i += 1
            j += 1
    return classified


def _unparse_stmt_indented(stmt: ast.stmt) -> str:
    stmt_source = ast.unparse(stmt)
    return "\n".join("    " + line for line in stmt_source.split("\n"))


def _extract_body_fragment(
    full_source: str,
    orig_tree: ast.Module,
    mut_tree: ast.Module,
    entry_point: str,
) -> str | None:
    """Extract the function body matching canonical_solution format.

    Unchanged statements are copied verbatim from the original source so
    comments, blank lines, and quote style are preserved. Only deleted or
    modified statements are omitted or re-serialized via ast.unparse.
    """
    orig_func = _find_function(orig_tree, entry_point)
    mut_func = _find_function(mut_tree, entry_point)
    if orig_func is None or mut_func is None:
        return None

    orig_body = _body_stmts_after_docstring(orig_func)
    mut_body = _body_stmts_after_docstring(mut_func)
    if not orig_body and not mut_body:
        return "\n"

    classified = _classify_body_statements(orig_body, mut_body)
    full_lines = full_source.splitlines(keepends=True)

    deleted_lines: set[int] = set()
    modified_by_start: dict[int, tuple[ast.stmt, str]] = {}
    for kind, orig_stmt, mut_stmt in classified:
        if kind == "delete" and orig_stmt is not None:
            end = getattr(orig_stmt, "end_lineno", orig_stmt.lineno)
            deleted_lines.update(range(orig_stmt.lineno, end + 1))
        elif kind == "modify" and orig_stmt is not None and mut_stmt is not None:
            modified_by_start[orig_stmt.lineno] = (
                orig_stmt,
                _unparse_stmt_indented(mut_stmt),
            )
        elif kind == "insert" and mut_stmt is not None:
            modified_by_start[mut_stmt.lineno] = (
                mut_stmt,
                _unparse_stmt_indented(mut_stmt),
            )

    body_start = orig_body[0].lineno if orig_body else mut_body[0].lineno
    body_end = max(
        getattr(orig_func, "end_lineno", body_start),
        getattr(mut_func, "end_lineno", body_start),
    )

    parts: list[str] = []
    lineno = body_start
    while lineno <= body_end:
        if lineno in deleted_lines:
            lineno += 1
            continue
        if lineno in modified_by_start:
            skip_stmt, text = modified_by_start[lineno]
            parts.append(text)
            if not text.endswith("\n"):
                parts.append("\n")
            end = getattr(skip_stmt, "end_lineno", skip_stmt.lineno)
            lineno = end + 1
            continue
        parts.append(full_lines[lineno - 1])
        lineno += 1

    result = "".join(parts)
    if not result.endswith("\n"):
        result += "\n"
    return result
