#!/usr/bin/env bash
# 100k rows, L=16, hidden 32, 1 epoch -- then resume from the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.deep_learning_state.train --config configs/gru_smoke.yaml "$@"
echo "==> resume check (epoch 2 continues from the saved state)"
python -m src.deep_learning_state.train --config configs/gru_smoke.yaml \
  --epochs 2 --resume experiments/deep_learning_state/checkpoints/gru_smoke_fold2024_last.pt "$@"
