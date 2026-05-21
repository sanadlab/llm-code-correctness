#!/usr/bin/env bash
# Compress or decompress the Mutation_DS split bundle as a single archive.
# Packs every subdirectory of dataset/bigcodebench/phase2/ into
# dataset/bigcodebench/phase2.tar.zst.
#
# Git workflow for the tracked archive:
# - `decompress` marks the tracked archive as skip-worktree, extracts it, then removes it from disk.
# - `compress` recreates the archive and clears skip-worktree so git can see updates.
# - `track-zst` / `hide-zst` toggle skip-worktree without touching the archive contents.
#
# Run from the repo root:
#   bash scripts/data/compress_split_data.sh compress
#   bash scripts/data/compress_split_data.sh decompress
#   bash scripts/data/compress_split_data.sh track-zst
#   bash scripts/data/compress_split_data.sh hide-zst

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE2_DIR="$REPO_ROOT/dataset/bigcodebench/phase2"
ARCHIVE="$REPO_ROOT/dataset/bigcodebench/phase2.tar.zst"
ARCHIVE_REL="dataset/bigcodebench/phase2.tar.zst"

if [[ ! -d "$PHASE2_DIR" ]]; then
    mkdir -p "$PHASE2_DIR"
fi

usage() {
    echo "Usage: $0 {compress|decompress|track-zst|hide-zst}" >&2
    exit 1
}

archive_tracked() {
    git -C "$REPO_ROOT" ls-files --error-unmatch -- "$ARCHIVE_REL" >/dev/null 2>&1
}

set_skip_worktree() {
    if archive_tracked; then
        git -C "$REPO_ROOT" update-index --skip-worktree -- "$ARCHIVE_REL"
        echo "Marked $ARCHIVE_REL as skip-worktree."
    fi
}

clear_skip_worktree() {
    if archive_tracked; then
        git -C "$REPO_ROOT" update-index --no-skip-worktree -- "$ARCHIVE_REL"
        echo "Cleared skip-worktree on $ARCHIVE_REL."
    fi
}

[[ $# -eq 1 ]] || usage

case "$1" in
    compress)
        shopt -s nullglob
        dirs=("$PHASE2_DIR"/*/)
        if [[ ${#dirs[@]} -eq 0 ]]; then
            echo "No split directories found in $PHASE2_DIR"
            exit 0
        fi
        names=()
        for d in "${dirs[@]}"; do names+=("$(basename "$d")"); done
        echo "Compressing ${#names[@]} split(s) -> $ARCHIVE_REL ..."
        tar -cf - -C "$PHASE2_DIR" "${names[@]}" | zstd -19 -T0 -c > "$ARCHIVE"
        clear_skip_worktree
        echo "Done."
        ;;
    decompress)
        set_skip_worktree
        if [[ ! -f "$ARCHIVE" ]]; then
            echo "No archive at $ARCHIVE_REL; nothing to decompress."
            exit 0
        fi
        echo "Decompressing $ARCHIVE_REL ..."
        zstd -d -c "$ARCHIVE" | tar -xf - -C "$PHASE2_DIR"
        rm -f "$ARCHIVE"
        echo "Done."
        ;;
    track-zst)
        clear_skip_worktree
        ;;
    hide-zst)
        set_skip_worktree
        ;;
    *)
        usage
        ;;
esac
