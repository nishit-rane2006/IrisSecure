"""Measure model inference on available, explicitly supported backends."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from ctypes import wintypes

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import register_existing_checkpoint


def _process_working_set_bytes() -> int | None:
    if __import__("os").name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return int(counters.WorkingSetSize)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    from config import CFG
    from config.hardware_config import configure_torch, resource_plan, torch_device
    from irisguard.artifact_manager import ArtifactError, ArtifactManager
    from src.utils import save_json

    hardware = configure_torch()
    plan = resource_plan(hardware)
    manager = ArtifactManager()
    try:
        checkpoint = manager.validate("model")
    except ArtifactError:
        checkpoint = register_existing_checkpoint("reused-existing-checkpoint")
    device = torch_device(hardware)
    from register import load_model

    model = load_model(device)
    results = []
    candidates = sorted(set((1, 2, 4, min(8, hardware.logical_processors))))
    for threads in candidates:
        torch.set_num_threads(max(1, threads))
        for batch_size in (1, 8):
            batch = torch.randn(batch_size, 3, CFG.image_size, CFG.image_size, device=device)
            with torch.inference_mode():
                model.embed(batch)
                started = time.perf_counter()
                for _ in range(max(1, args.repeats)):
                    model.embed(batch)
                elapsed = time.perf_counter() - started
            results.append({
                "backend": hardware.selected_backend,
                "cpu_threads": threads,
                "batch_size": batch_size,
                "repeats": max(1, args.repeats),
                "seconds_per_batch": elapsed / max(1, args.repeats),
                "images_per_second": batch_size * max(1, args.repeats) / elapsed,
                "working_set_bytes_after": _process_working_set_bytes(),
                "input": "random tensor; measures model kernels, not real image decode/preprocessing",
            })
            del batch

    # This project has no validated XPU runtime on this host; do not infer speed from GPU presence.
    report = {
        "hardware": hardware.to_dict(),
        "resource_plan": plan.to_dict(),
        "checkpoint_sha256": manager.sha256(checkpoint),
        "tested_backends": sorted({row["backend"] for row in results}),
        "unavailable_acceleration": {
            "intel_xpu": "Not tested because PyTorch reports no XPU backend/device.",
            "cuda": "Not tested because CUDA is unavailable and is not required.",
        },
        "results": results,
        "recommended_backend": "cpu",
        "recommendation_basis": "Only CPU was available and measured on this host.",
    }
    output = CFG.artifacts_dir / "benchmarks" / "hardware_benchmark.json"
    save_json(report, output)
    manager.record("hardware_benchmark", output, model_hash=report["checkpoint_sha256"], metadata={"tested_backends": report["tested_backends"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
