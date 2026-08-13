#!/usr/bin/env bash
# Colab 런타임 1회 준비. GPU 가 없으면 여기서 멈춥니다.
#
# 이 실험은 Mac MPS 를 쓰지 않습니다. train_ftt.py 도 CUDA 가 없으면 거부하지만,
# 40~90분짜리 학습을 시작하기 전에 여기서 먼저 걸러 시간을 낭비하지 않습니다.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
PY="${PYTHON:-python}"

echo "=== repository ==="
pwd
git rev-parse --short HEAD 2>/dev/null || echo "  (git 정보 없음)"
for f in experiments/deep_learning/ft_transformer/train_ftt.py \
         experiments/deep_learning/ft_transformer/v0_gate.py \
         src/deep_learning_state/metrics.py; do
  [[ -f "$f" ]] && echo "  OK   $f" || { echo "  MISSING  $f"; exit 1; }
done

echo
echo "=== python / packages ==="
"$PY" -V
"$PY" - <<'PYCODE'
import importlib, sys
need = ['torch', 'numpy', 'pandas', 'pyarrow']
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f'  {m:10s} {getattr(mod, "__version__", "?")}')
    except ImportError:
        missing.append(m)
        print(f'  {m:10s} MISSING')
if missing:
    print('\n  설치: pip install ' + ' '.join(missing))
    sys.exit(1)
PYCODE

echo
echo "=== GPU ==="
"$PY" - <<'PYCODE'
import sys, torch
ok = torch.cuda.is_available()
print(f'  torch.cuda.is_available()  {ok}')
if not ok:
    print('\n  *** GPU 가 없습니다. 런타임 > 런타임 유형 변경 > T4 GPU 로 바꾸고')
    print('      런타임을 다시 시작한 뒤 이 스크립트를 다시 실행하십시오.')
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f'  {p.name}   VRAM {p.total_memory/1e9:.1f} GB   '
      f'capability {p.major}.{p.minor}   bf16 {torch.cuda.is_bf16_supported()}')
PYCODE

echo
echo "setup OK — 다음: bash experiments/deep_learning/ft_transformer/run_v0_colab.sh"
