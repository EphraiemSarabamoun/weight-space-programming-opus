#!/usr/bin/env bash
# Fetch the authors' released logs (prompts, networks, answer keys) at the exact commit this reproduction used.
# Everything in this repo reads them from $WSP_REF (default ~/projects/_ref/lm_understand-logs); nothing is copied.
#   ./fetch_ref.sh [target_dir]
set -euo pipefail
COMMIT=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
DEST="${1:-${WSP_REF:-$HOME/projects/_ref/lm_understand-logs}}"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/RikiyaT/lm_understand-logs.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || git -C "$DEST" fetch --quiet --unshallow origin || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "authors' logs at $DEST, commit $(git -C "$DEST" rev-parse --short HEAD)"
