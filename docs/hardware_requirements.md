# Hardware and runtime

## Resource-aware operation

IrisGuard does not require CUDA. Hardware is inspected through `config.hardware_config`; with `IRISGUARD_BACKEND=auto` it selects CUDA or XPU only when PyTorch reports that backend as available, otherwise CPU. An Intel graphics adapter by itself does not imply a usable PyTorch XPU backend.

On Windows, the runtime reads the CPU model, logical processor count, available physical memory, video adapter descriptions, PyTorch version, backend availability, and selected accelerator memory when exposed. The resource plan uses currently available system memory on CPU, or available accelerator memory on CUDA/XPU, to select a conservative batch size. It targets an effective training batch around 32 using gradient accumulation and bounds worker count by available RAM and CPU count. On the observed host this yields batch size 4, accumulation 8, zero workers, and four PyTorch CPU threads. Tune with:

CPU training batches are 2/4/8/16 for available memory below 4/8/16 GiB or at least 16 GiB. Accelerator training batches are 4/8/16/32 for available memory below 6/10/16 GiB or at least 16 GiB. Extraction uses up to twice the planned training batch, capped at 64; CPU extraction caps that at 8. Workers are 0/1/2/4 below 6/12/24 GiB or above, then capped by the available logical processors. If memory cannot be queried, defaults remain conservative.

- `IRISGUARD_CPU_THREADS` (default: up to 4, based on processor count)
- `IRISGUARD_DATALOADER_WORKERS` (memory-aware default; capped below logical CPU count)
- `IRISGUARD_BATCH_SIZE` and `IRISGUARD_GRAD_ACCUM_STEPS` (override automatic training batch/accumulation)
- `IRISGUARD_BACKEND` (`auto`, `cpu`, `cuda`, or `xpu`; unsupported requests fall back to CPU)

Mixed precision and channels-last are enabled only for CUDA. An accelerator is not selected just because a video adapter is listed.

## Measured host facts

The current host reported an Intel Core i7-13620H, 16 logical processors, about 15.7 GiB physical RAM, Intel UHD Graphics, PyTorch `2.14.1+cpu`, and no CUDA or XPU backend. At inspection time, Windows reported about 4.6 GiB available RAM. These are runtime observations, not requirements or benchmark results.

Run `python scripts/benchmark_hardware.py` to record model-kernel timing and process working-set measurements for available CPU thread counts. Benchmark results are written to `artifacts/benchmarks/hardware_benchmark.json`. The benchmark uses random tensors and does not measure image decoding or preprocessing.

## Memory guidance

Embedding extraction derives its batch size from the same resource plan and writes directly into a NumPy memory-mapped array. Training uses the planned batch/accumulation and early stopping. Close competing applications and avoid running training, extraction, and the Gradio app concurrently.
