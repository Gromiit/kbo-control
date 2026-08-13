#!/usr/bin/env bash
# Colab 에서 이 스크립트 하나만 실행하면 V0 가 끝납니다.
#
#   bash experiments/deep_learning/ft_transformer/run_v0_colab.sh
#
# 환경변수로 덮어쓸 수 있습니다.
#   FTT_ARCHIVE  기본 /content/drive/MyDrive/kbo/ftt_data.tgz
#   SEED         기본 42
#   MODELS       기본 A,B,C
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
PY="${PYTHON:-python}"
DIR="experiments/deep_learning/ft_transformer"
ARCHIVE="${FTT_ARCHIVE:-/content/drive/MyDrive/kbo/ftt_data.tgz}"
SEED="${SEED:-42}"
MODELS="${MODELS:-A,B,C}"

echo "=== 1. 경로 ==="
echo "  repo     $(pwd)"
echo "  commit   $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "  archive  $ARCHIVE"
echo "  seed $SEED   models $MODELS"

echo
echo "=== 2. 데이터 압축 해제 ==="
if [[ ! -f "$ARCHIVE" ]]; then
  echo "  아카이브가 없습니다: $ARCHIVE"
  echo "  Drive 를 마운트했는지, MyDrive/kbo/ 에 ftt_data.tgz 를 올렸는지 확인하십시오."
  echo "    from google.colab import drive; drive.mount('/content/drive')"
  exit 1
fi
mkdir -p "$DIR"
tar xzf "$ARCHIVE" -C "$DIR"
ls -la "$DIR/data" | sed 's/^/  /'
"$PY" - <<PYCODE
import json, pathlib
s = json.loads(pathlib.Path('$DIR/data/schema.json').read_text())
print(f"  schema: fold {s['fold']}  numeric {s['n_numeric']}  cat {len(s['categorical_features'])}")
for k, v in s['files'].items():
    print(f"    {k:22s} {v['rows']:>9,} 행  seasons {v['seasons']}  "
          f"p_v9 {v['p_v9_rows']:,}")
PYCODE

echo
echo "=== 3. 학습 (seed $SEED) ==="
"$PY" "$DIR/train_ftt.py" --seed "$SEED" --models "$MODELS" \
  --data-dir "$DIR/data" --out-dir "$DIR/oof"

echo
echo "=== 4. 게이트 평가 ==="
if [[ -f work/research/oof_2024.parquet ]]; then
  "$PY" "$DIR/v0_gate.py" --seed "$SEED" --models "$MODELS" --oof-dir "$DIR/oof"
else
  echo "  work/research/oof_2024.parquet 이 없어 게이트를 건너뜁니다."
  echo "  (work/ 는 Colab 에 올리지 않는 것이 정상입니다.)"
  echo "  아래 OOF 를 Mac 으로 내려받아 실행하십시오:"
  echo "    python $DIR/v0_gate.py --seed $SEED"
fi

echo
echo "=== 5. 결과 ==="
echo "  OOF 예측   $DIR/oof/ftt_{A,B,C}_s${SEED}.parquet"
echo "  manifest   $DIR/oof/manifest_s${SEED}.json"
echo "  게이트     $DIR/v0_results_s${SEED}.csv   (게이트를 돌린 경우)"
ls -la "$DIR/oof" 2>/dev/null | sed 's/^/  /' || true
echo
echo "  Drive 로 복사하려면:"
echo "    cp -r $DIR/oof /content/drive/MyDrive/kbo/ftt_oof"
