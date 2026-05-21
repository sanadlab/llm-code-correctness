"""Pre-download the four models used in the Summer 2026 plan into the HF cache.

Usage:
    python scripts/util/download_models.py
    python scripts/util/download_models.py --models mistralai/Mistral-7B-Instruct-v0.3

The HF cache location follows the standard env vars
(`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`). On the HPC we typically set
`HF_HOME=$SCRATCH/hf-models`.
"""

from __future__ import annotations

import argparse
import os
import sys

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "infly/OpenCoder-8B-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="HF model IDs to download (default: all four plan models).",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Optional file patterns to restrict (e.g. '*.safetensors *.json tokenizer*').",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
        return 1

    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    print(f"HF cache: {cache_dir or '(default ~/.cache/huggingface)'}")

    for repo_id in args.models:
        print(f"\n=== {repo_id} ===")
        path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=args.allow_patterns,
        )
        print(f"  -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
