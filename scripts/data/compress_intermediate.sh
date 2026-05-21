#!/usr/bin/env bash
# Compress or decompress top-level JSONL files in dataset/bigcodebench/intermediate/.
# Subdirectories (e.g. gt_eval_workdir, mutant_eval_workdir) are intentionally skipped.
#
# Git workflow for tracked .jsonl.zst files:
# - `decompress` marks tracked archives as skip-worktree, then removes them from disk.
#   This keeps git status clean while you work with the decompressed .jsonl files.
# - `compress` recreates the archives and clears skip-worktree so git can see updates.
# - `track-zst` clears skip-worktree without recompressing, which is useful before
#   reviewing, staging, or pulling archive changes.
# - `hide-zst` reapplies skip-worktree to tracked archives.
#
# Run from the repo root:
#   bash scripts/data/compress_intermediate.sh compress    # *.jsonl → *.jsonl.zst  (zstd -19, all cores)
#   bash scripts/data/compress_intermediate.sh decompress  # *.jsonl.zst → *.jsonl and hide archive deletions from git
#   bash scripts/data/compress_intermediate.sh track-zst   # make tracked *.jsonl.zst visible to git again
#   bash scripts/data/compress_intermediate.sh hide-zst    # hide tracked *.jsonl.zst from git again

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERMEDIATE_DIR="$REPO_ROOT/dataset/bigcodebench/intermediate"
INTERMEDIATE_REL="dataset/bigcodebench/intermediate"

if [[ ! -d "$INTERMEDIATE_DIR" ]]; then
    echo "ERROR: directory not found: $INTERMEDIATE_DIR" >&2
    exit 1
fi

usage() {
    echo "Usage: $0 {compress|decompress|track-zst|hide-zst}" >&2
    exit 1
}

tracked_zst_files() {
    while IFS= read -r rel_path; do
        if [[ "$(dirname "$rel_path")" == "$INTERMEDIATE_REL" && "$rel_path" == *.jsonl.zst ]]; then
            printf '%s\n' "$rel_path"
        fi
    done < <(git -C "$REPO_ROOT" ls-files -- "$INTERMEDIATE_REL")
}

set_skip_worktree_for_tracked_zst() {
    mapfile -t tracked_files < <(tracked_zst_files)
    if [[ ${#tracked_files[@]} -eq 0 ]]; then
        echo "No tracked *.jsonl.zst files found in $INTERMEDIATE_REL"
        return 0
    fi
    git -C "$REPO_ROOT" update-index --skip-worktree -- "${tracked_files[@]}"
    echo "Marked ${#tracked_files[@]} tracked archive(s) as skip-worktree."
}

clear_skip_worktree_for_tracked_zst() {
    mapfile -t tracked_files < <(tracked_zst_files)
    if [[ ${#tracked_files[@]} -eq 0 ]]; then
        echo "No tracked *.jsonl.zst files found in $INTERMEDIATE_REL"
        return 0
    fi
    git -C "$REPO_ROOT" update-index --no-skip-worktree -- "${tracked_files[@]}"
    echo "Cleared skip-worktree on ${#tracked_files[@]} tracked archive(s)."
}

[[ $# -eq 1 ]] || usage

case "$1" in
    compress)
        shopt -s nullglob
        all_jsonl=("$INTERMEDIATE_DIR"/*.jsonl)
        # Skip shard files (*.shard_*_of_*.jsonl) — they are partial/temporary
        files=()
        for f in "${all_jsonl[@]}"; do
            [[ "$(basename "$f")" == *.shard_*_of_*.jsonl ]] && continue
            files+=("$f")
        done
        if [[ ${#files[@]} -eq 0 ]]; then
            echo "No *.jsonl files found in $INTERMEDIATE_DIR"
            exit 0
        fi
        for f in "${files[@]}"; do
            echo "Compressing $(basename "$f") ..."
            zstd -19 -T0 -k -f "$f"
        done
        clear_skip_worktree_for_tracked_zst
        echo "Done. Compressed ${#files[@]} file(s)."
        ;;
    decompress)
        set_skip_worktree_for_tracked_zst
        shopt -s nullglob
        files=("$INTERMEDIATE_DIR"/*.jsonl.zst)
        if [[ ${#files[@]} -eq 0 ]]; then
            echo "No *.jsonl.zst files found in $INTERMEDIATE_DIR"
            exit 0
        fi
        for f in "${files[@]}"; do
            echo "Decompressing $(basename "$f") ..."
            zstd -d -T0 --rm -f "$f"
        done
        echo "Done. Decompressed ${#files[@]} file(s)."
        ;;
    track-zst)
        clear_skip_worktree_for_tracked_zst
        ;;
    hide-zst)
        set_skip_worktree_for_tracked_zst
        ;;
    *)
        usage
        ;;
esac
