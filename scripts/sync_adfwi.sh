#!/usr/bin/env bash
# Sync the LOCAL (modified) ADFWI package into this repo's ADFWI_local/ mirror
# so local ADFWI changes can be committed to GitHub alongside dasfwi code.
#
# Usage:  ./scripts/sync_adfwi.sh          # sync + show what changed
#         then: git add ADFWI_local && git commit && git push
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ADFWI_SRC="$(cd "$REPO_DIR/../ADFWI" && pwd)"

rsync -a --delete \
      --exclude="__pycache__" --exclude="*.pyc" --exclude=".DS_Store" \
      "$ADFWI_SRC/ADFWI" "$REPO_DIR/ADFWI_local/"
cp "$ADFWI_SRC/requirements.txt" "$ADFWI_SRC/LICENSE" "$ADFWI_SRC/README.md" \
   "$REPO_DIR/ADFWI_local/"

echo "--- changes vs last commit ---"
git -C "$REPO_DIR" status --short ADFWI_local
