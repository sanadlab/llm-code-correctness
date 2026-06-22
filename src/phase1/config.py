"""Phase 1 configuration loader."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidateConfig:
    docker_image: str = "bigcodebench/bigcodebench-evaluate:latest"
    docker_script: str = "scripts/eval/run_bigcodebench_eval.sh"
    split: str = "complete"
    min_time_limit: int = 10
    parallel: int = 8
    batch_size: int = 250
    container_runtime: str = "docker"   # "docker" or "singularity"
    sif_path: str = ""                   # path to .sif file (required when runtime=singularity)


@dataclass
class LocalLLMConfig:
    model: str = "Qwen/Qwen3-Coder-Next"
    dtype: str = "bfloat16"          # bfloat16; 80B MoE (~160 GB) needs tensor_parallel_size>=2
    tensor_parallel_size: int = 2    # number of GPUs for tensor parallelism (2× H100/H200 recommended)
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2048
    batch_size: int = 64             # vLLM handles its own internal batching; this controls seed chunking
    gpu_memory_utilization: float = 0.90
    prompt_template_version: str = "v1"


@dataclass
class ConfoundConfig:
    enabled: bool = False
    max_variants_per_mutant: int = 1
    chunk_size: int = 1000
    local: LocalLLMConfig = field(default_factory=LocalLLMConfig)


@dataclass
class OperatorsConfig:
    include: List[str] = field(default_factory=lambda: [
        "AOR", "AOI", "AOD", "ROR", "COR", "COD", "UOI", "UOD",
        "CR", "VVR", "VCR", "CVR", "SD", "RVM", "TR", "FR",
        "RC", "NEG", "FCD", "CRN", "BOR", "BSOR", "BSOD",
    ])


@dataclass
class Phase1Config:
    benchmark: str = "bigcodebench"
    subset: str = "full"
    seed: int = 0
    output_root: str = "dataset/bigcodebench"
    validate: ValidateConfig = field(default_factory=ValidateConfig)
    confound: ConfoundConfig = field(default_factory=ConfoundConfig)
    operators: OperatorsConfig = field(default_factory=OperatorsConfig)

    @property
    def intermediate_dir(self) -> Path:
        return Path(self.output_root) / "intermediate"

    @property
    def tasks_path(self) -> Path:
        return self.intermediate_dir / "tasks.jsonl"

    @property
    def validated_tasks_path(self) -> Path:
        return self.intermediate_dir / "validated_tasks.jsonl"

    @property
    def dropped_tasks_path(self) -> Path:
        return self.intermediate_dir / "dropped_tasks.jsonl"

    @property
    def mutants_path(self) -> Path:
        return self.intermediate_dir / "mutants.jsonl"

    @property
    def validated_mutants_path(self) -> Path:
        return self.intermediate_dir / "validated_mutants.jsonl"

    @property
    def confound_candidates_path(self) -> Path:
        return self.intermediate_dir / "confound_candidates.jsonl"

    @property
    def confound_partial_candidates_path(self) -> Path:
        return self.intermediate_dir / "confound_candidates.partial.jsonl"

    @property
    def confound_progress_path(self) -> Path:
        return self.intermediate_dir / "confound_progress.json"

    @property
    def validated_confounded_path(self) -> Path:
        return self.intermediate_dir / "validated_confounded.jsonl"

    @property
    def frozen_path(self) -> Path:
        return Path(self.output_root) / "frozen.jsonl"

    @property
    def metadata_path(self) -> Path:
        return Path(self.output_root) / "metadata.json"


def load_config(path: str) -> Phase1Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Phase1Config(
        benchmark=raw.get("benchmark", "bigcodebench"),
        subset=raw.get("subset", "full"),
        seed=raw.get("seed", 0),
        output_root=raw.get("output_root", "dataset/bigcodebench"),
    )

    if "validate" in raw:
        v = raw["validate"]
        cfg.validate = ValidateConfig(
            docker_image=v.get("docker_image", cfg.validate.docker_image),
            docker_script=v.get("docker_script", cfg.validate.docker_script),
            split=v.get("split", cfg.validate.split),
            min_time_limit=v.get("min_time_limit", cfg.validate.min_time_limit),
            parallel=v.get("parallel", cfg.validate.parallel),
            batch_size=v.get("batch_size", cfg.validate.batch_size),
            container_runtime=v.get("container_runtime", cfg.validate.container_runtime),
            sif_path=os.path.expandvars(v.get("sif_path", cfg.validate.sif_path)),
        )

    if "confound" in raw:
        c = raw["confound"]
        local_cfg = LocalLLMConfig()
        if "local" in c:
            l = c["local"]
            local_cfg = LocalLLMConfig(
                model=l.get("model", local_cfg.model),
                dtype=l.get("dtype", local_cfg.dtype),
                tensor_parallel_size=l.get("tensor_parallel_size", local_cfg.tensor_parallel_size),
                temperature=l.get("temperature", local_cfg.temperature),
                top_p=l.get("top_p", local_cfg.top_p),
                max_tokens=l.get("max_tokens", local_cfg.max_tokens),
                batch_size=l.get("batch_size", local_cfg.batch_size),
                gpu_memory_utilization=l.get("gpu_memory_utilization", local_cfg.gpu_memory_utilization),
                prompt_template_version=l.get("prompt_template_version", local_cfg.prompt_template_version),
            )
        cfg.confound = ConfoundConfig(
            enabled=c.get("enabled", False),
            max_variants_per_mutant=c.get("max_variants_per_mutant", 1),
            chunk_size=c.get("chunk_size", 1000),
            local=local_cfg,
        )

    if "operators" in raw:
        o = raw["operators"]
        if "include" in o:
            cfg.operators = OperatorsConfig(include=o["include"])

    validate_parallel_override = os.environ.get("VALIDATE_PARALLEL")
    if validate_parallel_override:
        cfg.validate.parallel = int(validate_parallel_override)

    return cfg
