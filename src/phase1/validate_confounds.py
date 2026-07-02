"""Step 6: Validate confounded variants and enforce same-failure filtering.

Re-validates every confounded variant through Docker. Keeps only variants whose
failure_fingerprint matches the source mutant's fingerprint.
"""

import io
import os
import re
import tokenize
from pathlib import Path

from src.phase1.config import Phase1Config
from src.phase1.docker_eval import DockerEvalBatchTimeout, run_docker_eval
from src.phase1.io_utils import append_jsonl, read_jsonl, write_jsonl
from src.phase1.validate_mutants import (
    TIMEOUT_SPLIT_LADDER,
    compute_failure_fingerprint,
    _get_batch_timeout_seconds,
    _get_next_split_batch_size,
)


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_body_for_compare(code: str) -> str:
    """Token-level intermediate form: drop comments/whitespace, keep code tokens."""
    parts: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in {
                tokenize.COMMENT,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
            }:
                continue
            parts.append(tok.string)
        return re.sub(r"\s+", " ", "".join(parts)).strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        stripped = re.sub(r"#.*", "", code)
        return re.sub(r"\s+", " ", stripped.strip())


def _confound_equivalent_to_mutant(confound_code: str, mutant_code: str) -> bool:
    """True when confound code is cosmetically identical to the source mutant."""
    return _normalize_body_for_compare(confound_code) == _normalize_body_for_compare(
        mutant_code
    )


def _get_confound_shard_spec() -> tuple[int, int]:
    shard_count = int(os.environ.get("VALIDATE_CONFOUNDS_NUM_SHARDS", "1"))
    shard_index_str = os.environ.get(
        "VALIDATE_CONFOUNDS_SHARD_INDEX",
        os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
    )
    shard_index = int(shard_index_str)

    if shard_count < 1:
        raise ValueError("VALIDATE_CONFOUNDS_NUM_SHARDS must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"VALIDATE_CONFOUNDS_SHARD_INDEX must be in [0, {shard_count}), got {shard_index}"
        )

    return shard_index, shard_count


def _get_validated_confounds_output_path(cfg: Phase1Config, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return cfg.validated_confounded_path
    return cfg.intermediate_dir / (
        f"validated_confounded.shard_{shard_index:03d}_of_{shard_count:03d}.jsonl"
    )


def _process_batch_results(
    batch_candidates: list[dict],
    eval_results: dict,
    tasks: dict[str, dict],
    mutant_fp: dict[str, str | None],
    mutant_code: dict[str, str],
) -> list[dict]:
    """Match eval results back to candidates and filter by same-failure fingerprint."""
    kept = []
    by_task: dict[str, list[tuple[int, dict]]] = {}
    for i, c in enumerate(batch_candidates):
        by_task.setdefault(c["task_id"], []).append((i, c))

    for tid, task_candidates in by_task.items():
        results = eval_results.get(tid, [])
        for order_pos, (_, c) in enumerate(task_candidates):
            status = "fail"
            details = {}
            if isinstance(results, list) and order_pos < len(results):
                r = results[order_pos]
                status = r.get("status", r.get("plus_status", "fail"))
                if "details" not in r and "plus_fail_tests" in r:
                    details = {"plus_fail_tests": r.get("plus_fail_tests", []), "base_fail_tests": r.get("base_fail_tests", [])}
                else:
                    details = r.get("details", {})

            if status != "fail":
                continue

            source_code = mutant_code.get(c["source_mutant_id"], "")
            if _confound_equivalent_to_mutant(c["code"], source_code):
                continue

            fp = compute_failure_fingerprint(details)
            source_fp = mutant_fp.get(c["source_mutant_id"])
            if fp != source_fp:
                continue

            kept.append({
                "task_id": c["task_id"],
                "source_mutant_id": c["source_mutant_id"],
                "variant_id": c["variant_id"],
                "backend": c.get("backend", ""),
                "code": c["code"],
                "status": status,
                "details": details,
                "failure_fingerprint": fp,
            })
    return kept


def _validate_confound_batch(
    batch: list[dict],
    tasks: dict[str, dict],
    mutant_fp: dict[str, str | None],
    mutant_code: dict[str, str],
    cfg: Phase1Config,
    work_dir: Path,
    batch_idx: int,
) -> tuple[list[dict], int]:
    """Run a single eval batch with split-ladder retry on timeout.

    Returns (kept_rows, timeout_count).
    """
    batch_size = len(batch)
    batch_timeout_seconds = _get_batch_timeout_seconds(batch_size)

    samples = []
    for c in batch:
        task = tasks[c["task_id"]]
        solution = task["complete_prompt"] + c["code"]
        samples.append({"task_id": c["task_id"], "solution": solution})

    try:
        eval_results = run_docker_eval(
            samples, cfg, work_dir=work_dir,
            label=f"confounds_batch_{batch_idx:06d}",
            batch_timeout_seconds=batch_timeout_seconds,
        )
    except DockerEvalBatchTimeout:
        next_batch_size = _get_next_split_batch_size(batch_size)
        if next_batch_size is not None and batch_size > next_batch_size:
            split_batches = list(_chunked(batch, next_batch_size))
            print(
                f"    Batch {batch_idx} timed out at size {batch_size} with "
                f"timeout {batch_timeout_seconds}s; retrying as {len(split_batches)} "
                f"sub-batches of up to {next_batch_size}"
            )
            all_kept: list[dict] = []
            all_timeout = 0
            for split_idx, split_batch in enumerate(split_batches, start=1):
                child_idx = batch_idx * 1000 + split_idx
                kept, timed_out = _validate_confound_batch(
                    split_batch, tasks, mutant_fp, mutant_code, cfg, work_dir, child_idx,
                )
                all_kept.extend(kept)
                all_timeout += timed_out
            return all_kept, all_timeout

        # Terminal rung — record everything as timeout
        print(
            f"    Batch {batch_idx} timed out at smallest rung "
            f"(size {batch_size}, timeout {batch_timeout_seconds}s) — "
            f"{batch_size} candidates lost"
        )
        return [], batch_size

    kept = _process_batch_results(batch, eval_results, tasks, mutant_fp, mutant_code)
    return kept, 0


def validate_confounds(cfg: Phase1Config) -> None:
    candidates = read_jsonl(cfg.confound_candidates_path)
    shard_index, shard_count = _get_confound_shard_spec()
    output_path = _get_validated_confounds_output_path(cfg, shard_index, shard_count)

    if os.environ.get("FORCE") == "1" and output_path.exists():
        output_path.unlink()
        print(f"FORCE=1: removed existing output {output_path}")

    if not candidates:
        print("No confound candidates to validate, writing empty output")
        write_jsonl(output_path, [])
        return

    if shard_count > 1:
        candidates = [c for idx, c in enumerate(candidates) if idx % shard_count == shard_index]
        print(
            f"Shard {shard_index + 1}/{shard_count}: "
            f"assigned {len(candidates)} candidates"
        )

    # Build lookup for source mutant fingerprints and bodies
    validated_mutants = read_jsonl(cfg.validated_mutants_path)
    mutant_fp = {m["mutant_id"]: m["failure_fingerprint"] for m in validated_mutants}
    mutant_code = {m["mutant_id"]: m["code"] for m in validated_mutants}

    tasks = {t["task_id"]: t for t in read_jsonl(cfg.validated_tasks_path)}

    # Resume support: skip already-validated candidates
    done_variant_ids: set[str] = set()
    if output_path.exists():
        existing = read_jsonl(output_path)
        if existing:
            deduped: dict[str, dict] = {}
            for row in existing:
                deduped[row["variant_id"]] = row
            if len(deduped) != len(existing):
                existing = list(deduped.values())
                write_jsonl(output_path, existing)
                print(
                    f"Deduped {output_path}: "
                    f"{len(existing)} unique variant rows"
                )
            done_variant_ids = set(deduped.keys())
            print(
                f"Resuming confound validation from {output_path}: "
                f"found {len(done_variant_ids)} existing results"
            )

    pending = [c for c in candidates if c["variant_id"] not in done_variant_ids]
    equivalent_skipped = 0
    filtered_pending = []
    for c in pending:
        source_code = mutant_code.get(c["source_mutant_id"], "")
        if _confound_equivalent_to_mutant(c["code"], source_code):
            equivalent_skipped += 1
            continue
        filtered_pending.append(c)
    pending = filtered_pending

    print(
        f"Validating {len(pending)} pending confound candidates "
        f"(total={len(candidates)}, batch_size={cfg.validate.batch_size})..."
    )
    if equivalent_skipped:
        print(
            f"  Skipped {equivalent_skipped} candidates equivalent to source mutant "
            f"(after comment/whitespace normalization)"
        )

    work_dir = cfg.intermediate_dir / "confound_eval_workdir"
    # Prefer $SCRATCH for the eval workdir to avoid filling the home filesystem
    scratch = os.environ.get("SCRATCH")
    if scratch:
        work_dir = Path(scratch) / "phase1_eval" / "confound_eval_workdir"
    if shard_count > 1:
        work_dir = work_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_timeout = 0

    for batch_idx, batch in enumerate(_chunked(pending, cfg.validate.batch_size), start=1):
        print(f"  Batch {batch_idx}: {len(batch)} candidates")

        batch_kept, batch_timeout = _validate_confound_batch(
            batch, tasks, mutant_fp, mutant_code, cfg, work_dir, batch_idx,
        )
        if batch_kept:
            append_jsonl(output_path, batch_kept)
        total_kept += len(batch_kept)
        total_timeout += batch_timeout
        print(f"    Batch {batch_idx} done: kept {len(batch_kept)} / {len(batch)}")

    # Final summary
    all_kept = read_jsonl(output_path) if output_path.exists() else []
    print(f"Kept {len(all_kept)} / {len(candidates)} confound variants (same-failure match)")
    if total_timeout:
        print(f"  ({total_timeout} candidates skipped due to batch timeouts)")
    print(f"  → {output_path}")
