"""Hardware discovery and conservative resource planning for PyTorch."""

from __future__ import annotations

import ctypes
import os
import platform
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch


@dataclass(frozen=True)
class HardwareInfo:
    cpu_model: str
    logical_processors: int
    total_ram_bytes: Optional[int]
    available_ram_bytes: Optional[int]
    gpu_devices: List[str]
    intel_gpu_detected: bool
    pytorch_version: str
    supported_backends: List[str]
    selected_backend: str
    gpu_memory_bytes: Optional[int]
    cpu_threads: int
    dataloader_workers: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResourcePlan:
    backend: str
    cpu_threads: int
    dataloader_workers: int
    training_batch_size: int
    gradient_accumulation_steps: int
    embedding_batch_size: int
    use_mixed_precision: bool
    channels_last: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory() -> tuple[Optional[int], Optional[int]]:
    if os.name != "nt":
        return None, None
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    return None, None


def _cpu_model() -> str:
    model = platform.processor().strip()
    if model and "family" not in model.lower():
        return model
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            pass
    return model or platform.machine() or "Unknown CPU"


def _windows_video_devices() -> List[str]:
    if os.name != "nt":
        return []
    try:
        import winreg

        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
        devices = []
        index = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(key, index)
                index += 1
            except OSError:
                break
            if not subkey_name.isdigit():
                continue
            try:
                subkey = winreg.OpenKey(key, subkey_name)
                name, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                if name:
                    devices.append(str(name))
            except OSError:
                continue
        return devices
    except OSError:
        return []


def detect_hardware() -> HardwareInfo:
    logical = max(1, os.cpu_count() or 1)
    total_ram, available_ram = _windows_memory()
    devices = _windows_video_devices()
    backends = ["cpu"]

    if torch.cuda.is_available():
        backends.append("cuda")
        try:
            devices.extend(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        except RuntimeError:
            warnings.warn("CUDA was detected but device enumeration failed; CPU remains available.", RuntimeWarning)

    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    if xpu_available:
        backends.append("xpu")
        try:
            devices.extend(torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count()))
        except (AttributeError, RuntimeError):
            devices.append("Intel XPU")

    intel_gpu = any("intel" in device.lower() for device in devices)
    requested = os.environ.get("IRISGUARD_BACKEND", "auto").strip().lower()
    if requested == "auto":
        selected = next((backend for backend in ("cuda", "xpu") if backend in backends), "cpu")
    elif requested not in backends:
        warnings.warn(
            f"Requested backend '{requested}' is unavailable; using CPU. Supported: {', '.join(backends)}.",
            RuntimeWarning,
        )
        selected = "cpu"
    else:
        selected = requested

    gpu_memory = None
    if selected == "cuda":
        try:
            gpu_memory = int(torch.cuda.mem_get_info()[0])
        except RuntimeError:
            gpu_memory = int(torch.cuda.get_device_properties(0).total_memory)
    elif selected == "xpu":
        try:
            gpu_memory = int(torch.xpu.get_device_properties(0).total_memory * 0.65)
        except (AttributeError, RuntimeError):
            gpu_memory = None

    requested_threads = int(os.environ.get("IRISGUARD_CPU_THREADS", str(min(4, logical))))
    threads = min(logical, max(1, requested_threads))
    available_gib = available_ram / (1024 ** 3) if available_ram is not None else 0
    if available_gib < 6:
        default_workers = 0
    elif available_gib < 12:
        default_workers = 1
    elif available_gib < 24:
        default_workers = 2
    else:
        default_workers = 4
    default_workers = min(default_workers, max(0, logical - 1))
    requested_workers = int(os.environ.get("IRISGUARD_DATALOADER_WORKERS", str(default_workers)))
    workers = min(max(0, logical - 1), max(0, requested_workers))

    return HardwareInfo(
        cpu_model=_cpu_model(),
        logical_processors=logical,
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        gpu_devices=sorted(set(devices)),
        intel_gpu_detected=intel_gpu,
        pytorch_version=torch.__version__,
        supported_backends=backends,
        selected_backend=selected,
        gpu_memory_bytes=gpu_memory,
        cpu_threads=threads,
        dataloader_workers=workers,
    )


def resource_plan(info: Optional[HardwareInfo] = None) -> ResourcePlan:
    info = info or detect_hardware()
    if info.selected_backend == "cpu":
        memory_bytes = info.available_ram_bytes or 0
        if memory_bytes < 4 * 1024**3:
            batch_size = 2
        elif memory_bytes < 8 * 1024**3:
            batch_size = 4
        elif memory_bytes < 16 * 1024**3:
            batch_size = 8
        else:
            batch_size = 16
        embedding_batch_size = min(8, batch_size * 2)
    else:
        memory_bytes = info.gpu_memory_bytes or 0
        if memory_bytes < 6 * 1024**3:
            batch_size = 4
        elif memory_bytes < 10 * 1024**3:
            batch_size = 8
        elif memory_bytes < 16 * 1024**3:
            batch_size = 16
        else:
            batch_size = 32
        embedding_batch_size = min(64, batch_size * 2)

    batch_override = os.environ.get("IRISGUARD_BATCH_SIZE")
    if batch_override is not None:
        batch_size = int(batch_override)
        if not 1 <= batch_size <= 64:
            raise ValueError("IRISGUARD_BATCH_SIZE must be between 1 and 64.")
        embedding_batch_size = min(64, batch_size * 2)

    accumulation = max(1, (32 + batch_size - 1) // batch_size)
    accumulation_override = os.environ.get("IRISGUARD_GRAD_ACCUM_STEPS")
    if accumulation_override is not None:
        accumulation = int(accumulation_override)
        if accumulation < 1:
            raise ValueError("IRISGUARD_GRAD_ACCUM_STEPS must be a positive integer.")

    use_amp = info.selected_backend == "cuda"
    return ResourcePlan(
        backend=info.selected_backend,
        cpu_threads=info.cpu_threads,
        dataloader_workers=info.dataloader_workers,
        training_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        embedding_batch_size=embedding_batch_size,
        use_mixed_precision=use_amp,
        channels_last=info.selected_backend == "cuda",
    )


def configure_torch(info: Optional[HardwareInfo] = None) -> HardwareInfo:
    info = info or detect_hardware()
    torch.set_num_threads(info.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, info.cpu_threads)))
    except RuntimeError:
        pass

    if info.available_ram_bytes is not None:
        available_gib = info.available_ram_bytes / (1024 ** 3)
        if available_gib < 2:
            warnings.warn(
                f"Only {available_gib:.1f} GiB RAM is currently available; reduce batch sizes and close other applications.",
                RuntimeWarning,
            )
    return info


def torch_device(info: Optional[HardwareInfo] = None) -> torch.device:
    info = info or detect_hardware()
    if info.selected_backend == "cuda":
        return torch.device("cuda")
    if info.selected_backend == "xpu":
        return torch.device("xpu")
    return torch.device("cpu")


def memory_snapshot() -> Dict[str, Optional[int]]:
    total, available = _windows_memory()
    return {"total_bytes": total, "available_bytes": available}
