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
# 계약 검사를 먼저 한다. schema.json 이 없던 구버전 아카이브도 같은 260.3 MB 라
# 크기로는 구분되지 않으므로, payload 집합 일치로 걸러낸 뒤에 푼다.
"$PY" - <<PYCODE
import hashlib, json, sys, tarfile
EXPECT = ['data/ftt_2024_tr.parquet', 'data/ftt_2024_va.parquet',
          'data/schema.json']
FORBIDDEN = ('test.csv', 'work/', '.pkl', '.cbm', '.zip', 'submit',
             'checkpoint', 'shard')
h = hashlib.sha256()
with open('$ARCHIVE', 'rb') as f:
    for b in iter(lambda: f.read(1 << 22), b''):
        h.update(b)
print(f'  sha256   {h.hexdigest()}')
with tarfile.open('$ARCHIVE') as t:
    names = sorted(m.name for m in t.getmembers() if m.isfile())
    bad = [n for n in names if any(x in n for x in FORBIDDEN)]
    if bad:
        sys.exit(f'  금지 항목: {bad}')
    if names != EXPECT:
        sys.exit(f'  payload 가 계약과 다릅니다\n    기대 {EXPECT}\n'
                 f'    실제 {names}\n'
                 '  Mac 에서 prep_data.py 를 다시 실행하고 Drive 파일을 교체하십시오.')
    s = json.loads(t.extractfile('data/schema.json').read())
if s.get('contract') != 'ftt-v0/1':
    sys.exit(f"  contract {s.get('contract')} != ftt-v0/1")
print(f'  payload  {names}  (계약 일치)')
print(f"  schema   contract {s['contract']}  fold {s['fold']}  "
      f"numeric {s['n_numeric']}  cat {len(s['categorical_features'])}")
for k, v in s['files'].items():
    print(f"    {k:22s} {v['rows']:>9,} 행  seasons {v['seasons']}  "
          f"p_v9 {v['p_v9_rows']:,}")
PYCODE
mkdir -p "$DIR"
tar xzf "$ARCHIVE" -C "$DIR"
ls -la "$DIR/data" | sed 's/^/  /'

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
