#!/usr/bin/env bash
# Run once per Colab runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
pip -q install -r requirements-colab.txt
python - <<'PY'
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| available', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'gpu {p.name}  {p.total_memory/1e9:.1f} GB  bf16={torch.cuda.is_bf16_supported()}')
else:
    raise SystemExit('no CUDA -- switch the runtime to GPU (런타임 > 런타임 유형 변경)')
PY
ls -la data/sequences 2>/dev/null || echo 'NOTE: shards not mounted yet -- see README "데이터 이동"'
