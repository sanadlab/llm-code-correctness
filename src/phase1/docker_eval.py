"""Docker-based BigCodeBench evaluation helper.

Prepares a samples JSONL in BigCodeBench's expected format, calls the shell
wrapper to run evaluation inside Docker, and parses structured results back.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src.phase1.config import Phase1Config

# Empirically derived from 37 completed batches of 2000 samples:
#   max batch duration = 855s, mean = 304s, p95 ≈ 546s
# 1200s (20 min) gives meaningful headroom over the worst observed batch
# without letting pathological batches stall the pipeline for too long.
DEFAULT_BATCH_TIMEOUT = 1200
MAX_RETRIES = 3


class DockerEvalBatchTimeout(RuntimeError):
    """Raised when a batch exhausts its total Docker evaluation time budget."""


def _decode_output(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def run_docker_eval(
    samples: list[dict[str, str]],
    cfg: Phase1Config,
    work_dir: Path | None = None,
    label: str = "eval",
    batch_timeout_seconds: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run BigCodeBench evaluation in Docker for a list of samples.

    Args:
        samples: list of {"task_id": ..., "solution": ...} dicts.
            `solution` is the raw model output; the BCB sanitizer extracts
            executable code when --sanitize is enabled.
        cfg: Phase1Config with Docker/eval settings.
        work_dir: directory to write temp files; if None, uses a tempdir.
        label: prefix for the temp samples file name.
        batch_timeout_seconds: optional override for the total wall-clock
            timeout budget of this evaluator invocation.

    Returns:
        dict mapping task_id -> list of result dicts, each with keys:
            task_id, solution, status ("pass"|"fail"|"timeout"), details (dict).
        Returns empty dict if the batch exhausts its total timeout budget.
    """
    if not samples:
        return {}

    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="phase1_eval_"))
        cleanup = False  # keep for debugging; caller can clean up

    work_dir.mkdir(parents=True, exist_ok=True)
    samples_path = work_dir / f"{label}_samples.jsonl"
    _write_samples(samples_path, samples)

    script = Path(cfg.validate.docker_script)
    if not script.is_absolute():
        script = Path.cwd() / script

    # Derive the task-id subset so BigCodeBench only expects the tasks we supply
    task_ids = sorted(set(s["task_id"] for s in samples))

    runtime = cfg.validate.container_runtime

    cmd = [
        str(script),
        "--samples", str(samples_path),
        "--split", cfg.validate.split,
        "--subset", cfg.subset,
        "--image", cfg.validate.docker_image,
        "--parallel", str(cfg.validate.parallel),
        "--min-time-limit", str(cfg.validate.min_time_limit),
        "--selective-evaluate", ",".join(task_ids),
        "--runtime", runtime,
    ]

    if getattr(cfg.validate, "sanitize", False):
        cmd.append("--sanitize")

    if runtime == "singularity" and cfg.validate.sif_path:
        cmd.extend(["--sif", cfg.validate.sif_path])
        cmd.append("--no-patch")

    container_name = f"phase1_eval_{label}_{os.getpid()}"
    cmd.extend(["--container-name", container_name])
    timeout_budget = batch_timeout_seconds or DEFAULT_BATCH_TIMEOUT
    deadline = time.monotonic() + timeout_budget

    for attempt in range(1, MAX_RETRIES + 1):
        remaining = max(0, int(deadline - time.monotonic()))
        if remaining == 0:
            print(f"  Batch exceeded total timeout budget of {timeout_budget}s")
            if runtime == "docker":
                _stop_docker_container(container_name)
            raise DockerEvalBatchTimeout(
                f"Batch exceeded total timeout budget of {timeout_budget}s"
            )

        # Remove stale result files before each attempt (evaluator caches them)
        for suffix in ("_eval_results.json", "_pass_at_k.json"):
            stale = samples_path.with_name(samples_path.name.replace(".jsonl", suffix))
            stale.unlink(missing_ok=True)

        print(f"  Launching eval [{runtime}] (attempt {attempt}/{MAX_RETRIES}, "
              f"remaining={remaining}s): {' '.join(cmd)}")
        timed_out = False
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"  Eval timed out after exhausting the batch budget "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            if runtime == "docker":
                _stop_docker_container(container_name)

        if timed_out:
            if attempt < MAX_RETRIES and time.monotonic() < deadline:
                continue
            print(f"  Batch timed out within total budget of {timeout_budget}s")
            raise DockerEvalBatchTimeout(
                 f"Batch timed out within total budget of {timeout_budget}s"
            )

        stdout_text = _decode_output(result.stdout)
        stderr_text = _decode_output(result.stderr)

        if result.returncode == 0:
            break
        print(f"  Eval failed (attempt {attempt}/{MAX_RETRIES}, exit {result.returncode})")
        if "BrokenProcessPool" in stderr_text or "UnpicklingError" in stderr_text:
            print("  Worker crash detected, will retry...")
            if runtime == "docker":
                _stop_docker_container(container_name)
        else:
            print(f"  Docker eval stderr:\n{stderr_text}")
            raise RuntimeError(f"Docker eval failed (exit {result.returncode})")
    else:
        print(f"  Docker eval stderr:\n{stderr_text}")
        raise RuntimeError(
            f"Docker eval failed after {MAX_RETRIES} attempts (exit {result.returncode})"
        )

    # Parse the results file written by BigCodeBench
    result_path = samples_path.with_name(
        samples_path.name.replace(".jsonl", "_eval_results.json")
    )
    if not result_path.exists():
        raise FileNotFoundError(
            f"Expected eval results at {result_path} but file not found.\n"
            f"Docker stdout:\n{stdout_text}\nDocker stderr:\n{stderr_text}"
        )

    with open(result_path) as f:
        raw = json.load(f)

    # raw["eval"] maps task_id -> list of {task_id, solution, status, details}
    return raw.get("eval", {})


def _write_samples(path: Path, samples: list[dict[str, str]]) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({
                "task_id": s["task_id"],
                "solution": s["solution"],
            }, ensure_ascii=False) + "\n")


def _stop_docker_container(name: str) -> None:
    """Stop and remove a Docker container by name, ignoring errors if already gone."""
    try:
        subprocess.run(
            ["docker", "stop", "-t", "10", name],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # --rm flag on docker run handles removal, but force-remove just in case
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
