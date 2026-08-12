"""The header every run prints, and the provenance every result row carries."""
import json
import subprocess
import time
from pathlib import Path

from . import paths


def git_commit():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=paths.REPO,
                           capture_output=True, text=True, timeout=5)
        h = r.stdout.strip()
        if not h:
            return 'no-git'
        d = subprocess.run(['git', 'status', '--porcelain'], cwd=paths.REPO,
                           capture_output=True, text=True, timeout=5)
        return h[:12] + ('-dirty' if d.stdout.strip() else '')
    except Exception:
        return 'no-git'


def dataset_version(season, L):
    p = paths.SEQ / f'manifest_S{season}_L{L}.json'
    if not p.exists():
        return 'none'
    m = json.loads(p.read_text())
    return f"S{season}_L{L}@{m.get('built', '?')}"


def header(cfg, dev, dev_desc, n_train, n_valid):
    lines = [
        '=' * 68,
        f'  run        {cfg.name}  fold={cfg.fold}  seed={cfg.seed}',
        f'  git        {git_commit()}',
        f'  torch      {dev_desc["torch"]}   cuda={dev_desc["cuda"]}',
        f'  device     {dev_desc["device"]}   gpu={dev_desc["gpu"]}',
        f'  memory     RAM {dev_desc["ram_gb"]} GB   VRAM {dev_desc["vram_gb"]}',
        f'  platform   {dev_desc["platform"]}',
        f'  dataset    {dataset_version(cfg.fold, cfg.sequence_length)}',
        f'  rows       train {n_train:,}   valid {n_valid:,}',
        f'  config     {json.dumps(cfg.to_dict(), sort_keys=True)}',
        '=' * 68,
    ]
    return '\n'.join(lines)


def stamp():
    return time.strftime('%Y%m%d-%H%M%S')
