import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from config.hardware_config import configure_torch, detect_hardware, torch_device


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    hardware = configure_torch()
    return torch_device(hardware)


def optimize_cpu_runtime(num_threads: int = 4):
    hardware = detect_hardware()
    num_threads = min(max(1, num_threads), hardware.logical_processors)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, num_threads)))
    except RuntimeError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    try:
        torch.backends.mkldnn.enabled = True
    except AttributeError:
        pass
    print(
        f"[cpu] threads={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()}"
    )


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)


def save_json(obj, path: Path):
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.name}] {time.time() - self.t0:.2f}s")
