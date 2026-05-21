#!/usr/bin/env bash
# Compress or decompress the ICSE26 split bundle as a single archive.
# Packs dataset/icse26/{bigcodebench,humaneval,mbpp,synthetic}/ into
# dataset/icse26.tar.zst.
#
# Git workflow for the tracked archive:
# - `decompress` marks the tracked archive as skip-worktree, extracts it, then removes it from disk.
# - `compress` recreates the archive and clears skip-worktree so git can see updates.
# - `track-zst` / `hide-zst` toggle skip-worktree without touching the archive contents.
#
# Run from the repo root:
#   bash scripts/data/compress_icse26.sh compress
#   bash scripts/data/compress_icse26.sh decompress
#   bash scripts/data/compress_icse26.sh track-zst
#   bash scripts/data/compress_icse26.sh hide-zst

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ICSE26_DIR="$REPO_ROOT/dataset/icse26"
ARCHIVE="$REPO_ROOT/dataset/icse26.tar.zst"
ARCHIVE_REL="dataset/icse26.tar.zst"
SUBDIRS=(bigcodebench humaneval mbpp synthetic)

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
        present=()
        for d in "${SUBDIRS[@]}"; do
            [[ -d "$ICSE26_DIR/$d" ]] && present+=("$d")
        done
        if [[ ${#present[@]} -eq 0 ]]; then
            echo "No ICSE26 subdirectories found under $ICSE26_DIR"
            exit 0
        fi
        echo "Compressing ${present[*]} -> $ARCHIVE_REL ..."
        tar -cf - -C "$ICSE26_DIR" "${present[@]}" | zstd -19 -T0 -c > "$ARCHIVE"
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
        zstd -d -c "$ARCHIVE" | tar -xf - -C "$ICSE26_DIR"
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
