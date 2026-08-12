#!/usr/bin/env bash
# Smoke first, always. Full training only runs if SMOKE_ONLY is unset.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/gru_full.yaml}"
echo "==> smoke"
python -m src.deep_learning_state.train --config configs/gru_smoke.yaml --device cuda
if [[ -n "${SMOKE_ONLY:-}" ]]; then echo "SMOKE_ONLY set, stopping."; exit 0; fi
echo "==> full: $CONFIG"
for SEED in ${SEEDS:-42}; do        # sequential: one seed at a time
  python -m src.deep_learning_state.train --config "$CONFIG" --device cuda --seed "$SEED"
done
