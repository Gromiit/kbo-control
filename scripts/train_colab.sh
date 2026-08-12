#!/usr/bin/env bash
# Smoke first, always. Full training only runs if SMOKE_ONLY is unset.
#
# The smoke run is the FULL config cut down (100k rows, 1 epoch, no early
# stop), not configs/gru_smoke.yaml. gru_smoke.yaml is L=16 and the full
# config is L=32, so using it here would force a second shard set onto Drive
# just to prove the wiring.
#
#   FOLDS   validation seasons, space separated   (default: 2024)
#   SEEDS   seeds per fold, space separated       (default: 42)
#   CONFIG  yaml to run                           (default: configs/gru_full.yaml)
#
# The config carries a `fold`, but --fold overrides it, so one config drives
# both Phase B folds. Phase B needs 2023 AND 2024: a model that is only
# measured on one season cannot be told apart from a regime-specific one.
#
#   FOLDS='2023 2024' SEEDS='42 43 44' bash scripts/train_colab.sh
#
# Runs are sequential -- one GPU, one model at a time -- and every run writes
# {name}_fold{fold}_s{seed}_{best,last}.pt, so no two overwrite each other.
set -euo pipefail
cd "$(dirname "$0")/.."
# `python` is not on PATH on a stock macOS; `python3` is, and inside an
# activated venv both resolve to the same interpreter. Override with $PYTHON.
PY="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/gru_full.yaml}"
FOLDS="${FOLDS:-2024}"
SEEDS="${SEEDS:-42}"

# smoke every fold, not just the first: each fold is a separate shard set and
# a smoke on 2024 says nothing about whether the 2023 shards are loadable.
for F in $FOLDS; do
  echo "==> smoke  ($CONFIG, fold $F, 100k rows, 1 epoch)"
  "$PY" -m src.deep_learning_state.train --config "$CONFIG" --device cuda \
    --fold "$F" --name colab_smoke --max-rows 100000 --epochs 1 --patience 0
done
if [[ -n "${SMOKE_ONLY:-}" ]]; then echo "SMOKE_ONLY set, stopping."; exit 0; fi

N=0
for F in $FOLDS; do for S in $SEEDS; do N=$((N + 1)); done; done
echo "==> full: $CONFIG   folds [$FOLDS] x seeds [$SEEDS] = $N runs, sequential"
i=0
for F in $FOLDS; do
  for SEED in $SEEDS; do
    i=$((i + 1))
    echo "--- [$i/$N] fold $F  seed $SEED"
    "$PY" -m src.deep_learning_state.train --config "$CONFIG" --device cuda \
      --fold "$F" --seed "$SEED"
  done
done
