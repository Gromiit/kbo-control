#!/usr/bin/env bash
# Smoke first, always. Full training only runs if SMOKE_ONLY is unset.
#
# The smoke run is the FULL config cut down (100k rows, 1 epoch, no early
# stop), not configs/gru_smoke.yaml. gru_smoke.yaml is L=16 and the full
# config is L=32, so using it here would force a second shard set onto Drive
# just to prove the wiring.
set -euo pipefail
cd "$(dirname "$0")/.."
# `python` is not on PATH on a stock macOS; `python3` is, and inside an
# activated venv both resolve to the same interpreter. Override with $PYTHON.
PY="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/gru_full.yaml}"
echo "==> smoke  ($CONFIG, 100k rows, 1 epoch)"
"$PY" -m src.deep_learning_state.train --config "$CONFIG" --device cuda \
  --name colab_smoke --max-rows 100000 --epochs 1 --patience 0
if [[ -n "${SMOKE_ONLY:-}" ]]; then echo "SMOKE_ONLY set, stopping."; exit 0; fi
echo "==> full: $CONFIG"
for SEED in ${SEEDS:-42}; do        # sequential: one seed at a time
  "$PY" -m src.deep_learning_state.train --config "$CONFIG" --device cuda --seed "$SEED"
done
