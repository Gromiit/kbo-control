"""Device selection and the few places backends genuinely differ.

Priority CUDA > MPS > CPU. Model code never imports this beyond `pick()` and
`autocast_ctx()`; there is no `if cuda:` anywhere in models.py or train.py's
forward path. The differences that actually exist are collected here:

  * autocast   CUDA has working fp16/bf16 autocast. MPS autocast is not
               reliable across torch versions, so mixed_precision is ignored
               there rather than silently producing NaNs.
  * pin_memory only meaningful for CUDA host->device copies.
  * empty_cache different call per backend, no-op on CPU.
"""
import contextlib
import os
import platform
import subprocess
import torch


def pick(name=None):
    """`name` of None or 'auto' resolves by priority; an explicit name is
    honoured if available and otherwise refused loudly."""
    if name in (None, '', 'auto'):
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    name = str(name).lower()
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('--device cuda requested but no CUDA device is visible')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('--device mps requested but MPS is unavailable')
    return torch.device(name)


def supports_amp(dev):
    return dev.type == 'cuda'


def autocast_ctx(dev, enabled):
    if enabled and dev.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16
                              if torch.cuda.is_bf16_supported() else torch.float16)
    return contextlib.nullcontext()


def pin_memory(dev):
    return dev.type == 'cuda'


def empty_cache(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    elif dev.type == 'mps':
        torch.mps.empty_cache()


def safe_workers(requested, dev):
    """MPS plus fork-started workers is a known source of stalls on macOS, and
    16 GB shared memory makes many workers a swap risk regardless."""
    if platform.system() == 'Darwin':
        return min(int(requested), 2)
    return int(requested)


def describe(dev):
    """Everything worth printing at the top of a run."""
    d = dict(device=str(dev), torch=torch.__version__,
             platform=platform.platform(), python=platform.python_version())
    if dev.type == 'cuda':
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        d.update(cuda=torch.version.cuda, gpu=p.name,
                 vram_gb=round(p.total_memory / 1e9, 1),
                 capability=f'{p.major}.{p.minor}',
                 bf16=torch.cuda.is_bf16_supported())
    elif dev.type == 'mps':
        d.update(cuda=None, gpu='Apple Silicon (MPS)',
                 vram_gb='unified', bf16=False)
    else:
        d.update(cuda=None, gpu='cpu', vram_gb=None, bf16=False)
    d['ram_gb'] = _ram_gb()
    return d


def _ram_gb():
    try:
        if platform.system() == 'Darwin':
            out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'])
            return round(int(out) / 1e9, 1)
        return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)
    except Exception:
        return None
