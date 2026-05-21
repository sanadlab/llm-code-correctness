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
    "You rewrite the BODY of a Python function with only minimal cosmetic edits while preserving exact runtime behavior, exact bugs, exact exceptions, and exact test outcomes. "
    "You MUST NOT fix the code, improve the code, or introduce new behavior. "
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

PROMPT_TEMPLATES = {
    "v1": REFACTOR_PROMPT_V1,
}


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
                meta.append({
                    "task_id": m["task_id"],
                    "source_mutant_id": m["mutant_id"],
                    "source_failure_fingerprint": m.get("failure_fingerprint", ""),
                })

        outputs = llm.generate(prompts, sampling_params)

        variant_counter: dict[str, int] = {}
        candidates = []
        for output, m_meta in zip(outputs, meta):
            source_mid = m_meta["source_mutant_id"]
            vi = variant_counter.get(source_mid, 0)
            variant_counter[source_mid] = vi + 1
            refactored = _strip_code_fences(output.outputs[0].text)
            if not refactored:
                continue
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
