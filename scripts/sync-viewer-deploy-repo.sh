#!/usr/bin/env bash
# Push viewer_platform/ (or VIEWER_PREFIX) to a standalone deploy repo as repo root.
#
# Usage:
#   export VIEWER_DEPLOY_REPO="sanadlab/llm-code-correctness-viewer"
#   export VIEWER_DEPLOY_TOKEN="ghp_..."   # fine-grained or classic PAT with repo scope
#   ./scripts/sync-viewer-deploy-repo.sh
#
# Optional:
#   VIEWER_PREFIX=viewer_platform   (default)
#   VIEWER_DEPLOY_BRANCH=main       (default)
#   VIEWER_SOURCE_BRANCH=feat/code-review-viewer

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PREFIX="${VIEWER_PREFIX:-viewer_platform}"
DEPLOY_REPO="${VIEWER_DEPLOY_REPO:?Set VIEWER_DEPLOY_REPO e.g. sanadlab/llm-code-correctness-viewer}"
DEPLOY_BRANCH="${VIEWER_DEPLOY_BRANCH:-main}"
SOURCE_BRANCH="${VIEWER_SOURCE_BRANCH:-$(git branch --show-current)}"
SPLIT_BRANCH="viewer-deploy-split-$$"

if [[ ! -d "$PREFIX" ]]; then
  echo "error: directory '$PREFIX' not found in repo root" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "warning: you have uncommitted changes; subtree split uses committed history only" >&2
fi

echo "Splitting $PREFIX from branch $SOURCE_BRANCH → $DEPLOY_REPO ($DEPLOY_BRANCH)"

git fetch origin "$SOURCE_BRANCH" 2>/dev/null || true
git checkout "$SOURCE_BRANCH"

git subtree split --prefix="$PREFIX" -b "$SPLIT_BRANCH"

if [[ -n "${VIEWER_DEPLOY_TOKEN:-}" ]]; then
  REMOTE="https://x-access-token:${VIEWER_DEPLOY_TOKEN}@github.com/${DEPLOY_REPO}.git"
else
  REMOTE="git@github.com:${DEPLOY_REPO}.git"
fi

git push "$REMOTE" "${SPLIT_BRANCH}:${DEPLOY_BRANCH}" --force
git branch -D "$SPLIT_BRANCH"

echo "Done. Deploy repo updated: https://github.com/${DEPLOY_REPO}"
echo "Connect that repo to Vercel (root = /, production branch = ${DEPLOY_BRANCH})."
