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

            # Extract the mutated body: re-parse and get just the function body part
            mutated_body = _extract_body_fragment(
                full_mutated, entry_point, complete_prompt
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


def _extract_body_fragment(full_mutated_source: str, entry_point: str,
                            complete_prompt: str) -> str | None:
    """Extract the function body (the part corresponding to canonical_solution)
    from the full mutated source.

    Strategy: re-parse the mutated source, find the target function, unparse
    just its body, and re-indent to match canonical_solution format.
    """
    try:
        mutated_tree = ast.parse(full_mutated_source)
    except SyntaxError:
        return None

    # Find the target function
    target_func = None
    for node in ast.walk(mutated_tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            target_func = node
            break
    if target_func is None:
        return None

    # Unparse individually each statement in the function body
    # Skip leading docstring if present
    body = target_func.body
    start_idx = 0
    if (body and isinstance(body[0], ast.Expr) and
        isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
        start_idx = 1

    lines = []

    for stmt in body[start_idx:]:
        stmt_source = ast.unparse(stmt)
        # Re-indent: each line gets 4-space indent
        for line in stmt_source.split("\n"):
            lines.append("    " + line)

    result = "\n".join(lines) + "\n"
    return result
