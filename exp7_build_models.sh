#!/usr/bin/env bash
# Experiment 7, step 1: rebuild the authors' six Gemma 3 270M fine-tunes (pairs 100-102, honest and sycophant) with
# THEIR scripts, unchanged (code/train.py and code/eval.py from the released logs), then measure each model's
# behaviour as they did. The authors trained on one GPU with bf16 autocast; their train.py falls back to CPU in fp32
# (no autocast), which is what runs here, so the rebuilt checkpoints will not match their hashes bit for bit (they
# say GPU bf16 training is not bit-reproducible either). What must match is the behaviour: the sycophant follows
# ~100 % of wrong guesses on known held-out facts, the honest model ~1-3 % (eval/model_behaviour/).
#   ./exp7_build_models.sh            (models go to $WSP_MODELS, default ~/projects/_ref/models)
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
REF="${WSP_REF:-$HOME/projects/_ref/lm_understand-logs}"
MODELS="${WSP_MODELS:-$HOME/projects/_ref/models}"
PY="$REPO/.venv-train/bin/python"
BASE="$MODELS/gemma-3-270m-it"
OUT="$MODELS/exp7"
TRUTH="$REPO/results/exp7/model_behaviour"
mkdir -p "$OUT" "$TRUTH"
export SYCMID_BASE="$BASE" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for P in 100 101 102; do
  SPLIT="$REF/exp7_sycophancy/dataset/splits/pair$P.json"
  for R in honest sycophant; do
    D="$OUT/pair$P/$R"
    [ -f "$D/final/model.safetensors" ] || "$PY" "$REF/exp7_sycophancy/code/train.py" "$P" "$R" "$SPLIT" "$D"
    [ -f "$TRUTH/pair${P}_$R.json" ] || "$PY" "$REF/exp7_sycophancy/code/eval.py" "$D/final" "$SPLIT" "$TRUTH/pair${P}_$R.json"
  done
  [ -f "$TRUTH/pair${P}_base.json" ] || "$PY" "$REF/exp7_sycophancy/code/eval.py" "$BASE" "$SPLIT" "$TRUTH/pair${P}_base.json"
  echo "pair $P done"
done
