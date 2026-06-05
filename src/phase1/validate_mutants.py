"""Step 4: Validate mutants in Docker and compute failure fingerprints.

Runs every mutant against its task's BigCodeBench test suite inside Docker.
Records pass/fail/timeout and full structured output. Computes failure_fingerprint
for killed mutants. Writes validated_mutants.jsonl.
"""

import hashlib
import math
import os
import re
from pathlib import Path

from src.phase1.config import Phase1Config
from src.phase1.docker_eval import DockerEvalBatchTimeout, run_docker_eval
from src.phase1.io_utils import append_jsonl, read_jsonl, write_jsonl


# Each rung is (max_batch_size, timeout_seconds).  On timeout the batch splits to
# the next smaller rung instead of mass-marking rows as timeout.  Budgets assume
# ~12s per mutant wall time at parallel=8 (min_time_limit=10 + overhead).
TIMEOUT_SPLIT_LADDER: tuple[tuple[int, int], ...] = (
    (2000, 45 * 60),
    (1000, 25 * 60),
    (500, 15 * 60),
    (250, 10 * 60),
    (100, 8 * 60),
    (50, 5 * 60),
)

RESUME_ALL_MISSING = "all_missing"
RESUME_RETRY_TIMEOUTS = "retry_timeouts"
RESUME_RETRY_UNRESOLVED = "retry_unresolved"


def compute_failure_fingerprint(details: dict) -> str | None:
    """Compute a deterministic fingerprint from failure details.

    Based on sorted failing test names + traceback text after stripping
    transient file-system prefixes and timing text.
    """
    if not details:
        return None

    parts = []
    for test_name in sorted(details.keys()):
        trace = str(details[test_name])
        # Strip transient filesystem paths
        trace = re.sub(r'File "[^"]*"', 'File "<stripped>"', trace)
        # Strip line numbers in tracebacks so cosmetic line-count changes
        # (e.g. dropping a dead string literal) don't break fingerprint matching
        trace = re.sub(r', line \d+,', ', line <N>,', trace)
        # Strip timing info
        trace = re.sub(r'\d+\.\d+s', '<time>', trace)
        # Strip memory addresses
        trace = re.sub(r'0x[0-9a-fA-F]+', '<addr>', trace)
        parts.append(f"{test_name}::{trace}")

    fingerprint_input = "\n".join(parts)
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]


def _get_validate_shard_spec() -> tuple[int, int]:
    shard_count = int(os.environ.get("VALIDATE_NUM_SHARDS", "1"))
    shard_index_str = os.environ.get("VALIDATE_SHARD_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    shard_index = int(shard_index_str)

    if shard_count < 1:
        raise ValueError("VALIDATE_NUM_SHARDS must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"VALIDATE_SHARD_INDEX must be in [0, {shard_count}), got {shard_index}"
        )

    return shard_index, shard_count


def _get_validated_mutants_output_path(cfg: Phase1Config, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return cfg.validated_mutants_path
    return cfg.intermediate_dir / (
        f"validated_mutants.shard_{shard_index:03d}_of_{shard_count:03d}.jsonl"
    )


def _get_validate_resume_mode() -> str:
    resume_mode = os.environ.get("VALIDATE_RESUME_MODE", RESUME_ALL_MISSING).strip().lower()
    if resume_mode not in {RESUME_ALL_MISSING, RESUME_RETRY_TIMEOUTS, RESUME_RETRY_UNRESOLVED}:
        raise ValueError(
            "VALIDATE_RESUME_MODE must be one of "
            f"{RESUME_ALL_MISSING!r}, {RESUME_RETRY_TIMEOUTS!r}, "
            f"{RESUME_RETRY_UNRESOLVED!r}; got {resume_mode!r}"
        )
    return resume_mode


def validate_mutants(cfg: Phase1Config) -> None:
    mutants = read_jsonl(cfg.mutants_path)
    tasks = {t["task_id"]: t for t in read_jsonl(cfg.validated_tasks_path)}
    shard_index, shard_count = _get_validate_shard_spec()
    resume_mode = _get_validate_resume_mode()
    output_path = _get_validated_mutants_output_path(cfg, shard_index, shard_count)

    if shard_count > 1:
        mutants = [m for idx, m in enumerate(mutants) if idx % shard_count == shard_index]
        print(
            f"Shard {shard_index + 1}/{shard_count}: "
            f"assigned {len(mutants)} mutants"
        )

    done_ids = set()
    if output_path.exists():
        existing = read_jsonl(output_path)
        if resume_mode in {RESUME_RETRY_TIMEOUTS, RESUME_RETRY_UNRESOLVED}:
            preserved = [row for row in existing if row.get("status") != "timeout"]
            timeout_ids = {row["mutant_id"] for row in existing if row.get("status") == "timeout"}
            missing_ids = {m["mutant_id"] for m in mutants} - {row["mutant_id"] for row in existing}
            write_jsonl(output_path, preserved)
            done_ids = {row["mutant_id"] for row in preserved}
            if resume_mode == RESUME_RETRY_TIMEOUTS:
                print(
                    f"Retrying only prior timeouts from {output_path}: "
                    f"preserved {len(preserved)} pass/fail rows, queued {len(timeout_ids)} timeout rows"
                )
                if missing_ids:
                    print(
                        f"Warning: timeout-only retry mode found {len(missing_ids)} mutants with no prior row; "
                        "they are not retried in this mode"
                    )
                retry_ids = timeout_ids
            else:
                print(
                    f"Retrying unresolved mutants from {output_path}: "
                    f"preserved {len(preserved)} pass/fail rows, queued {len(timeout_ids)} timeout rows "
                    f"and {len(missing_ids)} missing rows"
                )
                retry_ids = timeout_ids | missing_ids

            pending = [m for m in mutants if m["mutant_id"] in retry_ids]
        else:
            done_ids = {row["mutant_id"] for row in existing}
            pending = [m for m in mutants if m["mutant_id"] not in done_ids]

        if done_ids:
            print(
                f"Resuming mutant validation from {output_path}: "
                f"found {len(done_ids)} existing results"
            )
    else:
        pending = list(mutants)

    print(
        f"Validating {len(pending)} pending mutants "
        f"(total={len(mutants)}, resume_mode={resume_mode}, "
        f"batch_size={cfg.validate.batch_size}, parallel={cfg.validate.parallel})..."
    )
    _warn_batch_timeout_budget(cfg)

    work_dir = cfg.intermediate_dir / "mutant_eval_workdir"
    # Prefer $SCRATCH for the eval workdir to avoid filling the home filesystem
    scratch = os.environ.get("SCRATCH")
    if scratch:
        work_dir = Path(scratch) / "phase1_eval" / "mutant_eval_workdir"
    if shard_count > 1:
        work_dir = work_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(_chunked(pending, cfg.validate.batch_size), start=1):
        print(f"  Batch {batch_idx}: {len(batch)} mutants")
        batch_validated = _validate_mutant_batch(batch, tasks, cfg, work_dir, batch_idx)
        append_jsonl(output_path, batch_validated)

        n_pass = sum(1 for v in batch_validated if v["status"] == "pass")
        n_fail = sum(1 for v in batch_validated if v["status"] == "fail")
        n_timeout = sum(1 for v in batch_validated if v["status"] == "timeout")
        print(f"    Batch {batch_idx} done: {n_fail} fail, {n_pass} pass, {n_timeout} timeout")

    validated = read_jsonl(output_path)
    n_pass = sum(1 for v in validated if v["status"] == "pass")
    n_fail = sum(1 for v in validated if v["status"] == "fail")
    n_timeout = sum(1 for v in validated if v["status"] == "timeout")
    print(f"Validated {len(validated)} mutants: {n_fail} killed, {n_pass} equivalent, {n_timeout} timeout")
    print(f"  → {output_path}")


def _validate_mutant_batch(
    batch: list[dict],
    tasks: dict[str, dict],
    cfg: Phase1Config,
    work_dir,
    batch_idx: int,
) -> list[dict]:
    batch_size = len(batch)
    batch_timeout_seconds = _get_batch_timeout_seconds(batch_size)
    samples = []
    mutant_index: dict[str, list[dict]] = {}
    for m in batch:
        tid = m["task_id"]
        task = tasks[tid]
        samples.append({
            "task_id": tid,
            "solution": task["complete_prompt"] + m["code"],
        })
        mutant_index.setdefault(tid, []).append(m)

    try:
        eval_results = run_docker_eval(
            samples,
            cfg,
            work_dir=work_dir,
            label=f"mutants_batch_{batch_idx:06d}",
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
            split_validated = []
            for split_idx, split_batch in enumerate(split_batches, start=1):
                child_batch_idx = batch_idx * 1000 + split_idx
                split_validated.extend(
                    _validate_mutant_batch(split_batch, tasks, cfg, work_dir, child_batch_idx)
                )
            return split_validated

        return [
            {
                "task_id": m["task_id"],
                "mutant_id": m["mutant_id"],
                "operator": m["operator"],
                "operator_code": m["operator_code"],
                "code": m["code"],
                "site": m.get("site", {}),
                "status": "timeout",
                "details": {
                    "ALL": (
                        "Batch exceeded total Docker evaluation timeout budget "
                        f"({batch_timeout_seconds}s) at batch size {batch_size}"
                    )
                },
                "failure_fingerprint": None,
                "validated_with": {
                    "execution": "docker",
                    "image": cfg.validate.docker_image,
                    "subset": cfg.subset,
                },
            }
            for m in batch
        ]

    expected_counts = {tid: len(items) for tid, items in mutant_index.items()}
    for tid, expected_count in expected_counts.items():
        actual_count = len(eval_results.get(tid, []))
        if actual_count != expected_count:
            raise RuntimeError(
                f"Incomplete eval results for {tid} in batch {batch_idx}: "
                f"expected {expected_count}, got {actual_count}"
            )

    validated = []
    task_offsets: dict[str, int] = {}
    for m in batch:
        tid = m["task_id"]
        results = eval_results.get(tid, [])
        task_offsets.setdefault(tid, 0)
        my_idx = task_offsets[tid]
        task_offsets[tid] += 1

        r = results[my_idx]
        status = r.get("status", "fail")
        details = r.get("details", {})

        fp = None if status == "pass" else compute_failure_fingerprint(details)
        validated.append({
            "task_id": tid,
            "mutant_id": m["mutant_id"],
            "operator": m["operator"],
            "operator_code": m["operator_code"],
            "code": m["code"],
            "site": m.get("site", {}),
            "status": status,
            "details": details,
            "failure_fingerprint": fp,
            "validated_with": {
                "execution": "docker",
                "image": cfg.validate.docker_image,
                "subset": cfg.subset,
            },
        })

    return validated


def _estimate_batch_seconds(batch_size: int, parallel: int, min_time_limit: int) -> int:
    """Conservative wall-clock estimate for one eval batch."""
    per_mutant = min_time_limit + 2  # test limit plus container/scheduling overhead
    return math.ceil(batch_size / max(parallel, 1) * per_mutant)


def _warn_batch_timeout_budget(cfg: Phase1Config) -> None:
    batch_size = cfg.validate.batch_size
    parallel = cfg.validate.parallel
    budget = _get_batch_timeout_seconds(batch_size)
    estimate = _estimate_batch_seconds(batch_size, parallel, cfg.validate.min_time_limit)
    if estimate > budget:
        print(
            f"  WARN: estimated batch time ~{estimate}s exceeds timeout budget "
            f"{budget}s at batch_size={batch_size}, parallel={parallel}. "
            f"Reduce batch_size in config, raise --parallel, or use --cpus 16 "
            f"--parallel 8. Otherwise batches will split repeatedly and may "
            f"finish as bogus status=timeout rows."
        )
    elif estimate > int(budget * 0.8):
        print(
            f"  NOTE: estimated batch time ~{estimate}s is close to timeout budget "
            f"{budget}s; consider lowering batch_size if batches start splitting."
        )


def _get_batch_timeout_seconds(batch_size: int) -> int:
    for max_batch_size, timeout_seconds in reversed(TIMEOUT_SPLIT_LADDER):
        if batch_size <= max_batch_size:
            return timeout_seconds
    return TIMEOUT_SPLIT_LADDER[0][1]


def _get_next_split_batch_size(batch_size: int) -> int | None:
    current_index = None
    for index, (max_batch_size, _) in enumerate(TIMEOUT_SPLIT_LADDER):
        if batch_size <= max_batch_size:
            current_index = index

    if current_index is not None:
        if current_index + 1 < len(TIMEOUT_SPLIT_LADDER):
            return TIMEOUT_SPLIT_LADDER[current_index + 1][0]
        return None

    return TIMEOUT_SPLIT_LADDER[1][0] if len(TIMEOUT_SPLIT_LADDER) > 1 else None


def _chunked(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
