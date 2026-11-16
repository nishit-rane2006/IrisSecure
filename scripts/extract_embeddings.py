"""Extract or migrate embeddings to a memory-mappable NumPy artifact."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import preprocessing_config, register_existing_checkpoint


def _splits_by_path():
    from config import CFG

    split_by_path = {}
    split_dir = CFG.work_dir / "artifacts" / "preprocess"
    for split in ("train", "val", "test"):
        frame = pd.read_csv(split_dir / f"{split}.csv")
        for path in frame["path"].astype(str):
            split_by_path[path] = split
    return split_by_path


def _save_labels(frame: pd.DataFrame, destination):
    with open(destination, "w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def _migrate_legacy_embeddings(destination, labels_destination):
    from config import CFG
    from irisguard.artifact_manager import ArtifactManager

    legacy = CFG.work_dir / "artifacts" / "embed" / "embeddings.npz"
    if not legacy.is_file():
        return False
    split_by_path = _splits_by_path()
    with np.load(legacy, allow_pickle=True) as data:
        embeddings = data["embeddings"]
        labels = data["labels"]
        paths = data["paths"].astype(str)
        if embeddings.ndim != 2 or embeddings.shape[1] != CFG.embedding_dim:
            raise ValueError(f"Legacy embeddings have incompatible shape: {embeddings.shape}")
        frame = pd.DataFrame({
            "path": paths,
            "label": labels.astype(np.int64),
            "split": [split_by_path.get(path, "unknown") for path in paths],
        })
        if (frame["split"] == "unknown").any():
            raise ValueError("Legacy embedding paths do not match the current split manifests.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Legacy embeddings contain NaN or infinite values.")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-2):
            raise ValueError("Legacy embeddings are not L2-normalized.")

        manager = ArtifactManager()
        with manager.atomic_destination(destination) as temp_path:
            mapped = np.lib.format.open_memmap(
                temp_path,
                mode="w+",
                dtype=np.float32,
                shape=embeddings.shape,
            )
            for start in range(0, len(embeddings), 512):
                mapped[start:start + 512] = embeddings[start:start + 512]
            mapped.flush()
            del mapped
        with manager.atomic_destination(labels_destination) as temp_path:
            _save_labels(frame, temp_path)
    return True


def _extract_from_images(destination, labels_destination, batch_size: int, workers: int, tta: bool):
    from config import CFG
    from irisguard.artifact_manager import ArtifactManager
    from register import load_model
    from src.dataset import IrisDataset, build_eval_transform
    from src.utils import get_device

    split_dir = CFG.work_dir / "artifacts" / "preprocess"
    frames = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(split_dir / f"{split}.csv")
        frame["path"] = frame["path"].astype(str)
        frame["label"] = frame["label"].astype(np.int64)
        frame["split"] = split
        frames.append(frame[["path", "label", "split"]])
    all_samples = pd.concat(frames, ignore_index=True)

    device = get_device()
    model = load_model(device)
    worker_count = max(0, min(int(workers), max(0, (os.cpu_count() or 1) - 1)))
    dataset = IrisDataset(
        list(zip(all_samples["path"], all_samples["label"])),
        transform=build_eval_transform(CFG.image_size),
        size=CFG.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), 64)),
        shuffle=False,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )
    manager = ArtifactManager()
    with manager.atomic_destination(destination) as temp_path:
        output = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), CFG.embedding_dim),
        )
        offset = 0
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device)
                embedding = model.embed(images)
                if tta:
                    embedding = F.normalize(embedding + model.embed(torch.flip(images, dims=[3])), dim=1)
                batch = embedding.cpu().numpy()
                output[offset:offset + len(batch)] = batch
                offset += len(batch)
                del images, embedding, batch
        output.flush()
        del output
    with manager.atomic_destination(labels_destination) as temp_path:
        _save_labels(all_samples, temp_path)
    del loader, dataset, all_samples, model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from config import CFG
    from config.hardware_config import configure_torch, resource_plan
    from irisguard.artifact_manager import ArtifactError, ArtifactManager

    CFG.ensure_dirs()
    hardware = configure_torch()
    plan = resource_plan(hardware)
    manager = ArtifactManager()
    embeddings_path = CFG.embedding_dir / "embeddings.npy"
    labels_path = CFG.embedding_dir / "embedding_labels.csv"
    metadata_path = CFG.embedding_dir / "embedding_metadata.json"
    try:
        manager.validate("model")
    except ArtifactError:
        register_existing_checkpoint("reused-existing-checkpoint")
    model_hash = str(manager.read_manifest()["artifacts"]["model"]["sha256"])
    dataset_fingerprint = manager.sha256(CFG.work_dir / "artifacts" / "clean" / "clean.csv")
    preprocessing_hash = manager.fingerprint_json(preprocessing_config())
    if not args.force:
        try:
            manager.validate(
                "embeddings",
                model_hash=model_hash,
                dataset_fingerprint=dataset_fingerprint,
                preprocessing_hash=preprocessing_hash,
            )
            manager.validate(
                "embedding_labels",
                model_hash=model_hash,
                dataset_fingerprint=dataset_fingerprint,
                preprocessing_hash=preprocessing_hash,
            )
            print(f"Reusing validated embeddings: {embeddings_path}")
            return
        except ArtifactError:
            pass

    migrated = False
    if not args.force:
        migrated = _migrate_legacy_embeddings(embeddings_path, labels_path)
    if not migrated:
        _extract_from_images(
            embeddings_path,
            labels_path,
            args.batch_size if args.batch_size is not None else plan.embedding_batch_size,
            args.num_workers if args.num_workers is not None else hardware.dataloader_workers,
            not args.no_tta,
        )

    labels = pd.read_csv(labels_path)
    metadata = {
        "format": "npy-memmap",
        "embedding_file": str(embeddings_path.resolve()),
        "labels_file": str(labels_path.resolve()),
        "shape": [int(len(labels)), CFG.embedding_dim],
        "dtype": "float32",
        "model_hash": model_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "preprocessing": preprocessing_config(),
        "preprocessing_hash": manager.fingerprint_json(preprocessing_config()),
        "backend": hardware.selected_backend,
        "source": "legacy-compatible-migration" if migrated else "image-extraction",
        "tta": not args.no_tta,
    }
    with manager.atomic_destination(metadata_path) as temp_path:
        temp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    manager.record("embeddings", embeddings_path, model_hash=model_hash, dataset_fingerprint=metadata["dataset_fingerprint"], preprocessing_hash=metadata["preprocessing_hash"], metadata={"shape": metadata["shape"], "source": metadata["source"]})
    manager.record("embedding_labels", labels_path, model_hash=model_hash, dataset_fingerprint=metadata["dataset_fingerprint"], preprocessing_hash=metadata["preprocessing_hash"])
    manager.record("embedding_metadata", metadata_path, model_hash=model_hash, dataset_fingerprint=metadata["dataset_fingerprint"], preprocessing_hash=metadata["preprocessing_hash"])
    print(f"Embeddings ready: {embeddings_path} ({len(labels)} samples, {CFG.embedding_dim} dimensions)")
    print(f"Source: {metadata['source']}; backend: {hardware.selected_backend}")


if __name__ == "__main__":
    main()
