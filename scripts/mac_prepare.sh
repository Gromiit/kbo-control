#!/usr/bin/env bash
# Mac only. Builds walk-forward folds, then sequence shards.
set -euo pipefail
cd "$(dirname "$0")/.."
L="${1:-16}"; SEASONS="${2:-2023,2024}"; MAXROWS="${3:-0}"
echo "==> folds"
python -m src.deep_learning_state.prepare_data
echo "==> sequences  L=$L seasons=$SEASONS max_rows=$MAXROWS"
python -m src.deep_learning_state.make_sequences \
  --seasons "$SEASONS" --sequence-length "$L" --max-rows "$MAXROWS"
echo "==> leakage audit"
python -m src.deep_learning_state.audit --seasons "$SEASONS" --sequence-length "$L"
du -sh data/sequences 2>/dev/null || true
