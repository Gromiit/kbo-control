"""Config: a YAML file, overridable from the command line.

Every knob the study needs is here and nothing is read from the environment
except paths, so an experiment is reproducible from (config file + git commit).
The resolved config is written next to the results row for exactly that reason.
"""
import argparse
import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Config:
    # identity
    name: str = 'gru'
    fold: int = 2024                 # validation season
    seed: int = 42

    # data
    sequence_length: int = 16
    max_rows: int = 0                # 0 = all; smoke tests set 100_000
    shard_rows: int = 100_000
    num_workers: int = 0

    # model
    model: str = 'gru'               # gru | mlp
    hidden_size: int = 32
    num_layers: int = 1
    head_hidden: int = 64
    dropout: float = 0.10

    # optimisation
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 1
    grad_clip: float = 5.0
    mixed_precision: bool = False
    patience: int = 6

    # io
    checkpoint_dir: str = ''
    output_dir: str = ''
    resume: str = ''
    save_every: int = 1
    eval_every: int = 1

    def to_dict(self):
        return dataclasses.asdict(self)


def _coerce(field_type, raw):
    if field_type is bool:
        return str(raw).lower() in ('1', 'true', 'yes', 'y', 'on')
    return field_type(raw)


def load(path):
    import yaml
    d = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name: f.type for f in dataclasses.fields(Config)}
    bad = [k for k in d if k not in known]
    if bad:
        raise SystemExit(f'unknown config keys in {path}: {bad}')
    return Config(**{k: _coerce(known[k], v) for k, v in d.items()})


def build_parser(prog):
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default=None,
                   help='cuda | mps | cpu; omit to auto-detect (CUDA > MPS > CPU)')
    p.add_argument('--dry-run', action='store_true',
                   help='print the resolved config and environment, then exit')
    for f in dataclasses.fields(Config):
        p.add_argument(f'--{f.name.replace("_", "-")}', dest=f.name,
                       default=None)
    return p


def resolve(argv=None, prog='train'):
    args = build_parser(prog).parse_args(argv)
    cfg = load(args.config)
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, _coerce(f.type, v))
    return cfg, args


def fingerprint(cfg):
    return json.dumps(cfg.to_dict(), sort_keys=True)
