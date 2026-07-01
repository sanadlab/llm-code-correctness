#!/bin/bash
# Stage one HF hub model snapshot onto fast node-local storage before loading.
#
# Why: reading ~15GB of safetensors shards from Lustre/NFS scratch under
# concurrent jobs can take 30-60 min or hang. Copying once to a node-local
# disk makes the actual model load ~3-5s.
#
# HF hub snapshots are symlinks into blobs/; we must dereference (-L) or the
# staged copy is empty and offline loading fails.
#
# Node-local target preference:
#   1. $SLURM_TMPDIR              (if the scheduler sets it)
#   2. cluster TmpFS /tmpdata     (real local disk here, ~1.5TB)
#   3. /tmp                       (last resort; often a small tmpfs)
#
# Usage (inside a SLURM job, after setting HF_HOME to the real scratch cache):
#   source scripts/slurm/stage_hf_model.sh
#   stage_hf_model_to_local "infly/OpenCoder-8B-Instruct"
#   # HF_HOME now points at $LOCAL_HF_HOME for the rest of the job.

_stage_pick_local_root() {
    if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
        echo "${SLURM_TMPDIR}"; return 0
    fi
    if [[ -d /tmpdata && -w /tmpdata ]]; then
        echo "/tmpdata"; return 0
    fi
    echo "/tmp"
}

stage_hf_model_to_local() {
    local model_id="$1"
    if [[ -z "${model_id}" ]]; then
        echo "stage_hf_model_to_local: model id required" >&2
        return 1
    fi

    local repo_dir="models--${model_id//\//--}"
    local src_root="${HF_HOME}/hub/${repo_dir}"
    if [[ ! -d "${src_root}/snapshots" ]]; then
        # Legacy layout: HF_HOME is the hub root itself.
        src_root="${HF_HOME}/${repo_dir}"
    fi
    if [[ ! -d "${src_root}/snapshots" ]]; then
        echo "stage_hf_model_to_local: no snapshots under ${src_root}" >&2
        echo "  Set HF_HOME to your real cache (e.g. \$SCRATCH/.cache/huggingface)." >&2
        return 1
    fi

    local snap
    snap="$(ls -1 "${src_root}/snapshots" | head -1)"
    local src_snap="${src_root}/snapshots/${snap}"

    local local_root job_tag
    local_root="$(_stage_pick_local_root)"
    job_tag="${SLURM_JOB_ID:-$$}"
    # Job-scoped so concurrent jobs on the same node never clobber each other.
    LOCAL_HF_HOME="${local_root}/hf-cache-${USER:-job}-${job_tag}"
    local dst_repo="${LOCAL_HF_HOME}/hub/${repo_dir}"
    local dst_snap="${dst_repo}/snapshots/${snap}"
    mkdir -p "${dst_snap}"

    # Remove the staged copy when the job shell exits (keeps node-local disk clean).
    trap '[[ -n "${LOCAL_HF_HOME:-}" ]] && rm -rf "${LOCAL_HF_HOME}"' EXIT

    echo "[stage] local root: ${local_root}"
    echo "[stage] copying ${model_id} snapshot to ${dst_snap} ..."
    local t0=$SECONDS
    # -L: follow symlinks into blobs/ and copy real weight files.
    if ! rsync -aL --info=stats2 "${src_snap}/" "${dst_snap}/"; then
        echo "stage_hf_model_to_local: rsync failed (out of local space?)" >&2
        df -h "${local_root}" >&2
        return 1
    fi
    if [[ -d "${src_root}/refs" ]]; then
        mkdir -p "${dst_repo}/refs"
        rsync -a "${src_root}/refs/" "${dst_repo}/refs/"
    fi
    echo "[stage] rsync done in $((SECONDS - t0))s"

    if [[ ! -s "${dst_snap}/config.json" ]]; then
        echo "stage_hf_model_to_local: staged config.json missing or empty" >&2
        return 1
    fi

    export HF_HOME="${LOCAL_HF_HOME}"
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    echo "[stage] HF_HOME=${HF_HOME}"
}
