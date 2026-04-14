#!/usr/bin/env bash
# Mirrors the entire csen-346 repo into the SCU-CSEN346/KELE org repo and pushes.
# Usage:
#   ./scripts/sync_kele.sh                        # uses default commit message
#   ./scripts/sync_kele.sh "your commit message"  # uses custom commit message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/../"
DEST="/Users/ulises/github/KELE"

if [ ! -d "$DEST/.git" ]; then
    echo "Error: KELE repo not found at $DEST"
    exit 1
fi

# Pull latest from org repo before syncing to avoid non-fast-forward push
cd "$DEST" && git pull --rebase

# Sync files (delete files in dest that no longer exist in src)
rsync -av --delete \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.py[cod]' \
    --exclude='.mypy_cache' \
    --exclude='.claude' \
    "$SRC" "$DEST/"

cd "$DEST"

# If nothing changed, skip commit
if git diff --quiet && git diff --staged --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "Nothing to sync — KELE repo is already up to date."
    exit 0
fi

git add -A

LAST_MSG="$(cd "$SCRIPT_DIR/.." && git log -1 --pretty=%B | head -1)"
COMMIT_MSG="${1:-"$LAST_MSG"}"
git commit -m "$COMMIT_MSG"
git push

echo "Synced and pushed to SCU-CSEN346/KELE."
