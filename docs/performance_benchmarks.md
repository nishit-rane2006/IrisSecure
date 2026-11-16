# Performance benchmarks

No benchmark results are pre-populated or inferred. Run:

```powershell
python scripts/benchmark_hardware.py --repeats 2
```

The script records measured inference throughput for CPU thread counts and batch sizes, current process working set where available, detected hardware, backend availability, and the exact checkpoint hash. Inputs are random tensors, so these numbers measure model-kernel inference only—not end-to-end authentication latency, image decode, preprocessing, or full dataset extraction.

Intel GPU execution is reported as unavailable unless PyTorch actually exposes an XPU device. The presence of Intel UHD Graphics is not treated as proof of accelerated inference.
