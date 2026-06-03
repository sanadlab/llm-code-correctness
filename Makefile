PYTHON ?= python3.11
VENV_DIR ?= .venv
HPC_ENV_NAME ?= $(notdir $(CURDIR))
HPC_VENV_DIR ?= $(SCRATCH)/envs/$(HPC_ENV_NAME)
HPC_PYTHON_MODULE ?= python/3.11.3
HPC_CUDA_MODULE ?= cuda/12.2.0
HPC_CUDA_HOME ?= $(if $(CUDA_HOME),$(CUDA_HOME),$(patsubst %/bin/nvcc,%,$(shell command -v nvcc 2>/dev/null)))
REQ_FILE ?= requirements.txt
BCB_DIR ?= third_party/bigcodebench

# HF cache root. Override with: make download-models HF_HOME=/my/path
HF_HOME ?= $(SCRATCH)/hf-models

VENV_PYTHON := $(VENV_DIR)/bin/python
HPC_VENV_PYTHON := $(HPC_VENV_DIR)/bin/python

# Auto-detect environment: HPC if SCRATCH is set, local otherwise
ifdef SCRATCH
  ACTIVE_VENV_DIR    ?= $(HPC_VENV_DIR)
  ACTIVE_VENV_PYTHON ?= $(HPC_VENV_PYTHON)
  IS_HPC             := 1
else
  ACTIVE_VENV_DIR    ?= $(VENV_DIR)
  ACTIVE_VENV_PYTHON ?= $(VENV_PYTHON)
  IS_HPC             :=
endif

.PHONY: setup setup-env setup-hpc update update-env update-hpc \
        submodules compress decompress download-models \
        compress-artifacts decompress-artifacts download-artifacts \
        pull-eval-image

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

setup:
ifdef IS_HPC
	$(MAKE) setup-hpc
else
	$(MAKE) setup-env
endif

update:
ifdef IS_HPC
	$(MAKE) update-hpc
else
	$(MAKE) update-env
endif

setup-env:
	@if [ ! -x "$(VENV_PYTHON)" ] || ! "$(VENV_PYTHON)" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1; then \
		rm -rf "$(VENV_DIR)"; \
		"$(PYTHON)" -m venv "$(VENV_DIR)"; \
	fi
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install -r "$(REQ_FILE)"

setup-hpc:
	@if [ -z "$$SCRATCH" ]; then \
		echo "Error: SCRATCH is not set. Export SCRATCH before running 'make setup-hpc'."; \
		exit 1; \
	fi
	@bash -l -c '\
		set -e; \
		module purge >/dev/null 2>&1 || true; \
		module load $(HPC_PYTHON_MODULE) $(HPC_CUDA_MODULE); \
		export CUDA_HOME="$$(dirname $$(dirname $$(command -v nvcc)))"; \
		echo "Using CUDA_HOME=$$CUDA_HOME"; \
		if [ ! -x "$(HPC_VENV_PYTHON)" ] || ! "$(HPC_VENV_PYTHON)" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >/dev/null 2>&1; then \
			rm -rf "$(HPC_VENV_DIR)"; \
			$(PYTHON) -m venv "$(HPC_VENV_DIR)"; \
		fi; \
		"$(HPC_VENV_PYTHON)" -m pip install --upgrade pip; \
		"$(HPC_VENV_PYTHON)" -m pip install -r requirements.txt; \
		"$(HPC_VENV_PYTHON)" -m pip install -r requirements-hpc.txt; \
	'

update-env:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "Error: venv not found at $(VENV_DIR). Run 'make setup-env' first."; \
		exit 1; \
	fi
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install -r "$(REQ_FILE)"

update-hpc:
	@if [ ! -x "$(HPC_VENV_PYTHON)" ]; then \
		echo "Error: venv not found at $(HPC_VENV_DIR). Run 'make setup-hpc' first."; \
		exit 1; \
	fi
	@bash -l -c '\
		set -e; \
		module purge >/dev/null 2>&1 || true; \
		module load $(HPC_PYTHON_MODULE) $(HPC_CUDA_MODULE); \
		export CUDA_HOME="$$(dirname $$(dirname $$(command -v nvcc)))"; \
		"$(HPC_VENV_PYTHON)" -m pip install --upgrade pip; \
		"$(HPC_VENV_PYTHON)" -m pip install -r requirements.txt; \
		"$(HPC_VENV_PYTHON)" -m pip install -r requirements-hpc.txt; \
	'

# ---------------------------------------------------------------------------
# Submodules (BigCodeBench)
# ---------------------------------------------------------------------------

submodules:
	git submodule update --init --recursive
	@if [ -x "$(ACTIVE_VENV_PYTHON)" ]; then \
		"$(ACTIVE_VENV_PYTHON)" -m pip install -e "$(BCB_DIR)"; \
	else \
		echo "Note: no venv detected; skipping 'pip install -e $(BCB_DIR)'. Run after 'make setup'."; \
	fi

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# `decompress` extracts every tracked dataset archive and marks the archives
# as skip-worktree so they stay out of `git status` while you work. `compress`
# rebuilds the archives from the on-disk directories and clears skip-worktree
# so git can see updates.

decompress:
	bash scripts/data/compress_icse26.sh decompress
	bash scripts/data/compress_split_data.sh decompress
	bash scripts/data/compress_intermediate.sh decompress

compress:
	bash scripts/data/compress_icse26.sh compress
	bash scripts/data/compress_split_data.sh compress
	bash scripts/data/compress_intermediate.sh compress

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Downloads the four instruct models into $(HF_HOME). Override the cache root
# without editing this file:
#     make download-models HF_HOME=/my/other/path

download-models:
	@mkdir -p "$(HF_HOME)"
	HF_HOME="$(HF_HOME)" HF_HUB_CACHE="$(HF_HOME)" \
		"$(ACTIVE_VENV_PYTHON)" scripts/util/download_models.py

# ---------------------------------------------------------------------------
# Pre-computed artifacts (vectors/ and results/)
# ---------------------------------------------------------------------------
# Both directories are too large for the git remote and are hosted externally.
# Override the URLs at invocation time, e.g.:
#     make download-artifacts \
#         VECTORS_URL=https://example.org/vectors.tar.zst \
#         RESULTS_URL=https://example.org/results.tar.zst

VECTORS_ARCHIVE ?= vectors.tar.zst
RESULTS_ARCHIVE ?= results.tar.zst

compress-artifacts:
	@if [ ! -d vectors ] && [ ! -d results ]; then \
		echo "Neither vectors/ nor results/ exists; nothing to compress."; exit 0; \
	fi
	@if [ -d vectors ]; then \
		echo "Compressing vectors/ -> $(VECTORS_ARCHIVE) ..."; \
		tar -cf - vectors | zstd -19 -T0 -c > "$(VECTORS_ARCHIVE)"; \
	fi
	@if [ -d results ]; then \
		echo "Compressing results/ -> $(RESULTS_ARCHIVE) ..."; \
		tar -cf - results | zstd -19 -T0 -c > "$(RESULTS_ARCHIVE)"; \
	fi

decompress-artifacts:
	@if [ -f "$(VECTORS_ARCHIVE)" ]; then \
		echo "Decompressing $(VECTORS_ARCHIVE) -> vectors/ ..."; \
		zstd -d -c "$(VECTORS_ARCHIVE)" | tar -xf -; \
	else \
		echo "Missing $(VECTORS_ARCHIVE). Run 'make download-artifacts' first."; \
	fi
	@if [ -f "$(RESULTS_ARCHIVE)" ]; then \
		echo "Decompressing $(RESULTS_ARCHIVE) -> results/ ..."; \
		zstd -d -c "$(RESULTS_ARCHIVE)" | tar -xf -; \
	else \
		echo "Missing $(RESULTS_ARCHIVE). Run 'make download-artifacts' first."; \
	fi

# ---------------------------------------------------------------------------
# BigCodeBench evaluation container (Singularity/Apptainer)
# ---------------------------------------------------------------------------
# Phase 1 validation steps run BigCodeBench inside a Singularity image. Pull it
# once into the path referenced by configs/phase1/bigcodebench.yaml (sif_path).
# Override the destination with: make pull-eval-image SIF_PATH=/my/path.sif

SIF_PATH ?= $(SCRATCH)/containers/bigcodebench-evaluate.sif

pull-eval-image:
	@if [ -z "$(SCRATCH)" ] && [ "$(SIF_PATH)" = "$(SCRATCH)/containers/bigcodebench-evaluate.sif" ]; then \
		echo "Error: SCRATCH is not set. Export SCRATCH or pass SIF_PATH explicitly."; \
		exit 1; \
	fi
	@mkdir -p "$(dir $(SIF_PATH))" "$(SCRATCH)/.singularity/cache"
	@if [ -f "$(SIF_PATH)" ]; then \
		echo "SIF already exists: $(SIF_PATH)"; \
	else \
		echo "Pulling bigcodebench-evaluate image -> $(SIF_PATH) ..."; \
		SINGULARITY_CACHEDIR="$(SCRATCH)/.singularity/cache" \
		APPTAINER_CACHEDIR="$(SCRATCH)/.singularity/cache" \
		singularity pull "$(SIF_PATH)" docker://bigcodebench/bigcodebench-evaluate:latest; \
	fi
