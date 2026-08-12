"""Where things live, resolved differently on the laptop and on Colab.

Nothing else in the package hard-codes a path. Every root can be overridden by
an environment variable, which is what lets the same code run from
/Users/sungmin/Desktop/LG Aimers 9 and from /content/kbo without edits.

    KBO_REPO      repo root                    (default: three levels up)
    KBO_RAW       train.csv / test.csv         (Mac only -- Colab never needs it)
    KBO_WORK      the v9 workspace             (Mac only)
    KBO_DATA      prepared shards              (default: $KBO_REPO/data)
    KBO_EXP       experiment outputs           (default: $KBO_REPO/experiments/...)

The Mac-only roots are Mac-only on purpose: preprocessing reads train.csv and
v9's lookup tables, GPU training reads neither. If a Colab run ever touches
RAW or WORK that is a bug, and `require_mac_roots()` is where it surfaces.
"""
import os
from pathlib import Path

REPO = Path(os.environ.get(
    'KBO_REPO', Path(__file__).resolve().parents[2])).resolve()

RAW = Path(os.environ.get('KBO_RAW', Path.home() / 'Desktop/open/data'))
WORK = Path(os.environ.get('KBO_WORK', REPO / 'work'))
DATA = Path(os.environ.get('KBO_DATA', REPO / 'data'))
SEQ = DATA / 'sequences'
EXP = Path(os.environ.get('KBO_EXP', REPO / 'experiments/deep_learning_state'))
CKPT = Path(os.environ.get('KBO_CKPT', EXP / 'checkpoints'))

LEGACY_DL = WORK / 'research' / 'deep_learning'   # round-17 MLP study, read-only


def on_colab():
    return 'COLAB_GPU' in os.environ or Path('/content').exists()


def require_mac_roots():
    """Preprocessing entry points call this so a misconfigured Colab run fails
    with a sentence instead of a FileNotFoundError forty lines deep."""
    missing = [str(p) for p in (RAW / 'train.csv', WORK / 'features.py')
               if not p.exists()]
    if missing:
        raise SystemExit(
            'preprocessing needs the raw data and the v9 workspace, missing:\n  '
            + '\n  '.join(missing)
            + '\n\nThis step runs on the Mac. On Colab, use the shards produced '
              'there (see README, "데이터 이동").')


def ensure_dirs():
    for p in (DATA, SEQ, SEQ / 'train', SEQ / 'valid', EXP, CKPT):
        p.mkdir(parents=True, exist_ok=True)


def add_work_to_syspath():
    """features.py / exp4 / exp5 live in the v9 workspace and are imported, not
    copied, so preprocessing can never drift from what v9 itself used."""
    import sys
    for p in (str(WORK), str(WORK / 'strat')):
        if p not in sys.path:
            sys.path.insert(0, p)
