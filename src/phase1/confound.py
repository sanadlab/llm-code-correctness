"""Step 5: Optional confounder generation via a local vLLM model.

When enabled, reads killed mutants and uses a local LLM (Qwen2.5-Coder or
similar) served by vLLM to produce cosmetic rewrites (variable renaming,
formatting, comments) that preserve the mutant's failure behavior.
Default: disabled.
"""

import json

from src.phase1.config import Phase1Config
from src.phase1.io_utils import append_jsonl, read_jsonl, write_jsonl

REFACTOR_SYSTEM_PROMPT = (
    "You are a Python code refactoring expert. "
    "You rewrite the BODY of a Python function using behavior-preserving structural and cosmetic edits. "
    "You MUST preserve exact runtime behavior, exact bugs, exact exceptions, and exact test outcomes for all inputs. "
    "You MUST NOT fix the code, improve the code, or introduce mutations. "
    "Your output MUST visibly differ from the input syntactically. "
    "You always output ONLY the indented function body, never a def line, never imports, and never any text outside the body."
)

REFACTOR_PROMPT_V1 = (
    "Below is the indented body of a Python function named `{{ entry_point }}`.\n"
    "Rewrite it with ONLY minimal cosmetic changes.\n\n"
    "Primary goal:\n"
    "- Preserve the exact current behavior of the code, including any existing bug.\n"
    "- The rewritten body should fail or pass in exactly the same way as the input body.\n"
    "- Do NOT repair the bug. Do NOT make the code more correct.\n\n"
    "OUTPUT FORMAT — critical:\n"
    "- Return ONLY the rewritten function body, exactly as it would appear "
    "indented inside `def {{ entry_point }}(...):`.\n"
    "- Do NOT include a `def` line, function signature, or any wrapper.\n"
    "- Do NOT include import statements.\n"
    "- The first line of your output must be indented (4 spaces), "
    "just like the input.\n"
    "- Preserve the same number of lines and the same statement order whenever possible.\n"
    "- Do NOT wrap the output in Markdown fences.\n\n"
    "ALLOWED changes:\n"
    "- Rename local variables or loop indices, but only if the rename is fully local and consistent.\n"
    "- Change existing comments in place without adding or removing comment lines.\n"
    "- Make tiny formatting changes that do not add or remove lines.\n\n"
    "FORBIDDEN changes:\n"
    "- Do NOT add, remove, or change any logic, control flow, or function calls.\n"
    "- Do NOT change any external API usage (e.g. do not rename `shuffle` to `random.shuffle`).\n"
    "- Do NOT alter the function signature or default arguments.\n"
    "- Do NOT change operators, literals, constants, attribute names, subscripts, return values, exception behavior, or container structure.\n"
    "- Do NOT reorder statements.\n"
    "- Do NOT add or remove comments, blank lines, or docstrings.\n"
    "- Do NOT rewrite expressions into different forms such as comprehensions, helper variables, or different library calls.\n"
    "- Do NOT introduce any new names except a pure rename of an existing local variable.\n\n"
    "Fallback rule:\n"
    "- If you are not completely sure a rewrite preserves the exact same behavior and exact same bug, copy the input body unchanged.\n\n"
    "Function body:\n"
    "```\n"
    "{{ code }}\n"
    "```\n\n"
    "Rewritten body:"
)

REFACTOR_PROMPT_V2 = (
    "Below is the indented body of a Python function named `{{ entry_point }}`.\n"
    "Rewrite it with ONLY minimal cosmetic changes.\n\n"
    "Primary goal:\n"
    "- Preserve the exact current behavior of the code, including any existing bug.\n"
    "- The rewritten body should fail or pass in exactly the same way as the input body.\n"
    "- Do NOT repair the bug. Do NOT make the code more correct.\n\n"
    "MANDATORY — your output must differ from the input:\n"
    "- You MUST make at least one visible cosmetic change (e.g. rename one or more local variables consistently, "
    "reword an existing comment, or switch quote style on a string literal).\n"
    "- NEVER return the input body unchanged or with only whitespace differences.\n"
    "- Identical output will be rejected and discarded.\n"
    "- If the body has no comments and few local names, rename the simplest local variable throughout.\n\n"
    "OUTPUT FORMAT — critical:\n"
    "- Return ONLY the rewritten function body, exactly as it would appear "
    "indented inside `def {{ entry_point }}(...):`.\n"
    "- Do NOT include a `def` line, function signature, or any wrapper.\n"
    "- Do NOT include import statements.\n"
    "- The first line of your output must be indented (4 spaces), "
    "just like the input.\n"
    "- Preserve the same number of lines and the same statement order whenever possible.\n"
    "- Do NOT wrap the output in Markdown fences.\n\n"
    "ALLOWED changes:\n"
    "- Rename local variables or loop indices, but only if the rename is fully local and consistent.\n"
    "- Change existing comments in place without adding or removing comment lines.\n"
    "- Switch double-quoted string literals to single quotes (or vice versa) when safe.\n"
    "- Make tiny formatting changes that do not add or remove lines.\n\n"
    "FORBIDDEN changes:\n"
    "- Do NOT add, remove, or change any logic, control flow, or function calls.\n"
    "- Do NOT change any external API usage (e.g. do not rename `shuffle` to `random.shuffle`).\n"
    "- Do NOT alter the function signature or default arguments.\n"
    "- Do NOT change operators, literals, constants, attribute names, subscripts, return values, exception behavior, or container structure.\n"
    "- Do NOT reorder statements.\n"
    "- Do NOT add or remove comments, blank lines, or docstrings.\n"
    "- Do NOT rewrite expressions into different forms such as comprehensions, helper variables, or different library calls.\n"
    "- Do NOT introduce any new names except a pure rename of an existing local variable.\n\n"
    "When uncertain:\n"
    "- Prefer the smallest safe cosmetic edit (usually a local rename) over returning identical code.\n"
    "- Do NOT return the input unchanged under any circumstances.\n\n"
    "Function body:\n"
    "```\n"
    "{{ code }}\n"
    "```\n\n"
    "Rewritten body (must differ from the input above):"
)

REFACTOR_PROMPT_V3 = (
    "Below is the indented body of a Python function named `{{ entry_point }}`.\n"
    "Rewrite it to be syntactically distinct while maintaining exact semantic equivalence.\n\n"
    "PRIMARY GOAL:\n"
    "- Preserve the exact current behavior, including any existing bugs, logic errors, or efficiency flaws.\n"
    "- The rewritten body must fail or pass in exactly the same way as the input for every possible input.\n"
    "- Do NOT repair bugs. Do NOT make the code more correct.\n\n"
    "OUTPUT FORMAT:\n"
    "- Return ONLY the rewritten function body, exactly as it would appear indented inside `def {{ entry_point }}(...):`.\n"
    "- Do NOT include a `def` line, imports, or Markdown fences.\n"
    "- The first line of your output must be indented (4 spaces).\n\n"
    "ALLOWED TRANSFORMATIONS — Each rule must be executed as a complete, multi-step operation:\n"
    "1. Complete Conditional Inversion: You may change `if X:` to `if not X:`, but you MUST simultaneously swap the entire text body of the `if` block with the entire text body of the `else` block. Changing the condition without swapping the blocks is a forbidden mutation.\n"
    "2. Exhaustive Local Renaming: You may rename a local variable or loop index, but you MUST rename every single instance of that variable across the entire function. Leaving any instance unrenamed is a forbidden mutation.\n"
    "3. De Morgan's Equivalence: You may rewrite boolean logic using De Morgan's laws (e.g., rewriting `not (A or B)` into `not A and not B`). You MUST transform both the operators and the operands simultaneously so the logic truth table does not change.\n"
    "4. Literal String Formatting Conversion: You may convert an f-string (e.g., `f\"{x}\"`) to equivalent `.format()` syntax (e.g., `\"{}\".format(x)`), or vice versa. The generated string value must remain identical.\n"
    "5. Local Variable Extraction: You may break up a complex calculation by extracting a safe, side-effect-free sub-expression into a new local intermediate variable immediately before its execution point.\n\n"
    "CRITICAL MUTATION SAFEGUARDS — FORBIDDEN CHANGES:\n"
    "- Do NOT apply a logical operator (`not`, `~`, `-`) to any condition or variable unless you apply a counter-balancing transformation that nullifies its behavioral effect.\n"
    "- Do NOT add, remove, reorder, or alter any function calls, method calls, or expressions that produce side effects (such as mutations, object writes, or external API calls).\n"
    "- Do NOT change loop constraints, loop ranges, or loop iteration boundaries.\n"
    "- Do NOT change return values, literal constants, exception types, or container types.\n\n"
    "FALLBACK RULE:\n"
    "- If you cannot guarantee that a transformation preserves the exact same logic and bugs, copy the input body completely unchanged.\n\n"
    "Function body:\n"
    "{{ code }}\n\n"
    "Rewritten body:"
)

RETRY_UNCHANGED_SUFFIX = (
    "\n\nIMPORTANT: Your output was identical to the input and was rejected. "
    "You MUST apply at least one cosmetic change — rename a local variable, "
    "reword a comment, or change string quote style — while preserving the exact bug."
)

PROMPT_TEMPLATES = {
    "v1": REFACTOR_PROMPT_V1,
    "v2": REFACTOR_PROMPT_V2,
    "v3": REFACTOR_PROMPT_V3,
}


def same_function_body(a: str, b: str) -> bool:
    """True when two function-body strings are identical after strip."""
    return a.strip() == b.strip()


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs wrap around code output."""
    import re
    stripped = text.strip()
    if not stripped.startswith('```'):
        return stripped
    stripped = re.sub(r'^```[\w]*\n?', '', stripped)
    stripped = re.sub(r'\n?```\s*$', '', stripped)
    return stripped


def confound(cfg: Phase1Config) -> None:
    if not cfg.confound.enabled:
        print("Confounding disabled, writing empty confound_candidates.jsonl")
        write_jsonl(cfg.confound_candidates_path, [])
        return

    validated = read_jsonl(cfg.validated_mutants_path)
    killed = [m for m in validated if m["status"] != "pass"]
    if not killed:
        print("No killed mutants, writing empty confound_candidates.jsonl")
        write_jsonl(cfg.confound_candidates_path, [])
        return

    print(f"Generating confounded variants for {len(killed)} killed mutants...")
    _local_llm_confound_batch(killed, cfg)



def _local_llm_confound_batch(killed_mutants: list[dict], cfg: Phase1Config) -> None:
    """Run confounding via a local LLM (vLLM offline engine) with resumable chunking.

    Uses vLLM's LLM class for fast batched offline inference on GPU.
    Recommended model: Qwen/Qwen2.5-Coder-32B-Instruct (fits in H100 80GB at bfloat16).
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for confounding with backend='local'. "
            "Install with: pip install vllm"
        )

    local_cfg = cfg.confound.local
    tasks = {t["task_id"]: t for t in read_jsonl(cfg.validated_tasks_path)}
    total_mutants = len(killed_mutants)
    partial_path = cfg.confound_partial_candidates_path
    progress_path = cfg.confound_progress_path
    chunk_size = max(1, cfg.confound.chunk_size)

    start_index = 0
    written_candidates = 0
    if partial_path.exists() and progress_path.exists():
        state = _load_progress(progress_path)
        if (
            state.get("total_mutants") == total_mutants
            and state.get("max_variants_per_mutant") == cfg.confound.max_variants_per_mutant
        ):
            start_index = min(int(state.get("processed_mutants", 0)), total_mutants)
            written_candidates = int(state.get("written_candidates", 0))
            print(
                f"  Resuming local LLM confounding at mutant {start_index} / {total_mutants} "
                f"with {written_candidates} candidates already written"
            )
        else:
            print("  Existing partial outputs do not match; restarting from scratch")
            _reset_partial_outputs(cfg, total_mutants)
    else:
        _reset_partial_outputs(cfg, total_mutants)

    print(f"  Loading model {local_cfg.model} (dtype={local_cfg.dtype}, tp={local_cfg.tensor_parallel_size}, gpu_util={local_cfg.gpu_memory_utilization})")
    llm = LLM(
        model=local_cfg.model,
        dtype=local_cfg.dtype,
        tensor_parallel_size=local_cfg.tensor_parallel_size,
        gpu_memory_utilization=local_cfg.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=local_cfg.temperature,
        top_p=local_cfg.top_p,
        max_tokens=local_cfg.max_tokens,
    )

    prompt_version = local_cfg.prompt_template_version
    prompt_template = PROMPT_TEMPLATES.get(prompt_version, REFACTOR_PROMPT_V1)
    max_variants = cfg.confound.max_variants_per_mutant

    total_chunks = (total_mutants + chunk_size - 1) // chunk_size
    for chunk_start in range(start_index, total_mutants, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_mutants)
        chunk_number = (chunk_start // chunk_size) + 1
        chunk = killed_mutants[chunk_start:chunk_end]
        print(
            f"  Processing confound chunk {chunk_number} / {total_chunks} "
            f"(mutants {chunk_start + 1}–{chunk_end})"
        )

        # Build (prompt, metadata) pairs — repeat max_variants times per mutant
        prompts = []
        meta = []
        filled_texts = []
        for m in chunk:
            task = tasks.get(m["task_id"], {})
            entry_point = task.get("entry_point", "task_func")
            filled = (
                prompt_template
                .replace("{{ entry_point }}", entry_point)
                .replace("{{ code }}", m["code"])
            )
            chat_prompt = _format_chat_prompt(REFACTOR_SYSTEM_PROMPT, filled, local_cfg.model)
            for _ in range(max_variants):
                prompts.append(chat_prompt)
                filled_texts.append(filled)
                meta.append({
                    "task_id": m["task_id"],
                    "source_mutant_id": m["mutant_id"],
                    "source_failure_fingerprint": m.get("failure_fingerprint", ""),
                })

        outputs = llm.generate(prompts, sampling_params)

        source_code_by_id = {m["mutant_id"]: m["code"] for m in chunk}
        variant_counter: dict[str, int] = {}
        candidates = []
        skipped_unchanged = 0
        retry_prompts: list[str] = []
        retry_meta: list[dict] = []

        def _try_add_candidate(output_text: str, m_meta: dict) -> bool:
            source_mid = m_meta["source_mutant_id"]
            source_code = source_code_by_id.get(source_mid, "")
            refactored = _strip_code_fences(output_text)
            
            import tokenize, io
            def parses(c):
                try:
                    for _ in tokenize.generate_tokens(io.StringIO(c).readline): pass
                    return True
                except:
                    return False
            
            if not parses(refactored):
                if parses("    " + refactored):
                    refactored = "    " + refactored
                elif parses("        " + refactored):
                    refactored = "        " + refactored

            if not refactored:
                return False
            if same_function_body(refactored, source_code):
                return False
            vi = variant_counter.get(source_mid, 0)
            variant_counter[source_mid] = vi + 1
            candidates.append({
                "task_id": m_meta["task_id"],
                "source_mutant_id": source_mid,
                "variant_id": f"{source_mid}__confound__{vi}",
                "code": refactored,
                "source_failure_fingerprint": m_meta["source_failure_fingerprint"],
                "local": {
                    "model": local_cfg.model,
                    "prompt_template_version": prompt_version,
                },
            })
            return True

        for output, m_meta, filled in zip(outputs, meta, filled_texts):
            if _try_add_candidate(output.outputs[0].text, m_meta):
                continue
            refactored = _strip_code_fences(output.outputs[0].text)
            if refactored and same_function_body(
                refactored, source_code_by_id.get(m_meta["source_mutant_id"], "")
            ):
                retry_prompts.append(
                    _format_chat_prompt(
                        REFACTOR_SYSTEM_PROMPT,
                        filled + RETRY_UNCHANGED_SUFFIX,
                        local_cfg.model,
                    )
                )
                retry_meta.append(m_meta)
            else:
                skipped_unchanged += 1

        if retry_prompts:
            retry_outputs = llm.generate(retry_prompts, sampling_params)
            recovered = 0
            for output, m_meta in zip(retry_outputs, retry_meta):
                if _try_add_candidate(output.outputs[0].text, m_meta):
                    recovered += 1
                else:
                    skipped_unchanged += 1
            print(
                f"  Retried {len(retry_prompts)} unchanged outputs in chunk "
                f"{chunk_number}; recovered {recovered}"
            )

        append_jsonl(partial_path, candidates)
        written_candidates += len(candidates)
        _write_progress(
            path=progress_path,
            processed_mutants=chunk_end,
            total_mutants=total_mutants,
            max_variants_per_mutant=max_variants,
            chunk_size=chunk_size,
            written_candidates=written_candidates,
        )
        if skipped_unchanged:
            print(
                f"  Skipped {skipped_unchanged} unchanged LLM outputs in chunk "
                f"{chunk_number}"
            )
        print(
            f"  Wrote {len(candidates)} candidates from chunk {chunk_number}; "
            f"{written_candidates} total so far"
        )

    partial_path.replace(cfg.confound_candidates_path)
    if progress_path.exists():
        progress_path.unlink()
    print(f"Wrote {written_candidates} confound candidates to {cfg.confound_candidates_path}")


def _format_chat_prompt(system: str, user: str, model_name: str) -> str:
    """Format a chat prompt using the model's expected template.

    vLLM can apply chat templates automatically via tokenizer; here we use a
    simple universal format and rely on vLLM's apply_chat_template if supported.
    For Qwen2.5 models the expected format is <|im_start|>system\\n...
    We emit the raw messages string; vLLM handles template application internally
    when using llm.chat(), but since we use llm.generate() we apply it manually.
    """
    if "qwen" in model_name.lower():
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    # Generic fallback: instruction-style
    return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"


def _load_progress(path):
    with open(path) as f:
        return json.load(f)


def _write_progress(
    path,
    processed_mutants: int,
    total_mutants: int,
    max_variants_per_mutant: int,
    chunk_size: int,
    written_candidates: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "processed_mutants": processed_mutants,
                "total_mutants": total_mutants,
                "max_variants_per_mutant": max_variants_per_mutant,
                "chunk_size": chunk_size,
                "written_candidates": written_candidates,
            },
            f,
        )


def _reset_partial_outputs(cfg: Phase1Config, total_mutants: int) -> None:
    write_jsonl(cfg.confound_partial_candidates_path, [])
    _write_progress(
        path=cfg.confound_progress_path,
        processed_mutants=0,
        total_mutants=total_mutants,
        max_variants_per_mutant=cfg.confound.max_variants_per_mutant,
        chunk_size=max(1, cfg.confound.chunk_size),
        written_candidates=0,
    )
