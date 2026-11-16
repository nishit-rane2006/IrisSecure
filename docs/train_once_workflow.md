# Train-once preparation workflow

The Gradio application never trains a model, extracts the dataset, calibrates thresholds, or runs research experiments. Run preparation commands explicitly from the repository root.

## 1. Set up and inspect the local dataset

Follow the dataset layout and legal-use guidance in the README. Then run:

```powershell
python -c "from irisguard.dataset_inspector import write_manifest_and_statistics; print(write_manifest_and_statistics()[:2])"
python pipeline.py --status
```

The project does not download the dataset. Its images and generated local artifacts are excluded from Git.

## 2. Reuse or train a checkpoint

```powershell
python scripts/train_once.py
```

The command validates and adopts a compatible existing checkpoint without training. If no valid checkpoint exists, training requires explicit approval:

```powershell
python scripts/train_once.py --approve-training --epochs 8 --max-classes 200
```

The training batch size, gradient accumulation, and worker count come from detected resources unless explicitly passed with `--batch-size`/`--num-workers` or configured through environment variables. Retraining over a valid checkpoint requires `--force`; the old model is backed up first. Checkpoints include the embedding dimension, class count, validation score, and epoch. A sidecar records preprocessing, seed, dataset fingerprint, and model hash.

Inspect the runtime decision without starting a training run:

```powershell
python -c "from config.hardware_config import detect_hardware, resource_plan; hardware=detect_hardware(); print(hardware.to_dict()); print(resource_plan(hardware).to_dict())"
```

## 3. Prepare embeddings and calibration

```powershell
python scripts/extract_embeddings.py
python scripts/calibrate_once.py
```

Extraction first reuses a compatible registered embedding artifact or migrates the existing legacy NPZ if its model-independent dimensions, paths, and normalization validate. Otherwise it performs explicit image extraction. Calibration uses validation-split embeddings only. Use `--force` only to regenerate existing artifacts.

## 4. Normal use and research

```powershell
python app.py
python scripts/run_experiments.py
python scripts/benchmark_hardware.py
python scripts/validate_artifacts.py
```

The Gradio runtime loads the validated model and calibration, memory-maps embeddings, opens the local SQLite template store, and starts the interface. Missing or invalid prepared artifacts produce the exact preparation commands rather than triggering expensive work.
