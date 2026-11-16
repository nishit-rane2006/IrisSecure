"""Helpers shared by standalone project scripts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def preprocessing_config():
    from config import CFG
    from src.dataset import IMAGENET_MEAN, IMAGENET_STD

    return {
        "image_size": CFG.image_size,
        "color_mode": "grayscale_to_rgb",
        "gaussian_blur_kernel": [3, 3],
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
    }


def checkpoint_metadata(checkpoint_path: Path, source: str) -> dict:
    from config import CFG
    from irisguard.artifact_manager import ArtifactManager

    manager = ArtifactManager()
    clean_csv = CFG.work_dir / "artifacts" / "clean" / "clean.csv"
    dataset_fingerprint = manager.sha256(clean_csv) if clean_csv.exists() else None
    prep = preprocessing_config()
    return {
        "model_path": str(checkpoint_path.resolve()),
        "architecture": CFG.backbone,
        "embedding_dim": CFG.embedding_dim,
        "num_classes": None,
        "checkpoint_sha256": manager.sha256(checkpoint_path),
        "dataset_fingerprint": dataset_fingerprint,
        "preprocessing": prep,
        "preprocessing_hash": manager.fingerprint_json(prep),
        "random_seed": CFG.seed,
        "source": source,
    }


def register_existing_checkpoint(source: str) -> Path:
    import torch

    from config import CFG
    from irisguard.artifact_manager import ArtifactManager
    from src.utils import save_json

    checkpoint_path = CFG.model_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {checkpoint_path}. "
            "Train explicitly with `python scripts/train_once.py --approve-training`."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("embedding_dim", -1)) != CFG.embedding_dim:
        raise ValueError(
            f"Checkpoint embedding dimension {checkpoint.get('embedding_dim')} does not match {CFG.embedding_dim}."
        )
    if int(checkpoint.get("num_classes", 0)) <= 0 or not checkpoint.get("ema", checkpoint.get("model")):
        raise ValueError("Checkpoint is missing valid class metadata or model weights.")

    metadata = checkpoint_metadata(checkpoint_path, source)
    metadata["num_classes"] = int(checkpoint["num_classes"])
    metadata["checkpoint_epoch"] = int(checkpoint.get("epoch", 0))
    metadata["validation_accuracy"] = float(checkpoint.get("val_acc", 0.0))
    metadata_path = checkpoint_path.parent / "model_metadata.json"
    save_json(metadata, metadata_path)

    manager = ArtifactManager()
    manager.record(
        "model",
        checkpoint_path,
        model_hash=metadata["checkpoint_sha256"],
        dataset_fingerprint=metadata["dataset_fingerprint"],
        preprocessing_hash=metadata["preprocessing_hash"],
        metadata={
            "architecture": metadata["architecture"],
            "embedding_dim": metadata["embedding_dim"],
            "num_classes": metadata["num_classes"],
            "checkpoint_epoch": metadata["checkpoint_epoch"],
            "validation_accuracy": metadata["validation_accuracy"],
            "source": source,
        },
    )
    manager.record(
        "model_metadata",
        metadata_path,
        model_hash=metadata["checkpoint_sha256"],
        dataset_fingerprint=metadata["dataset_fingerprint"],
        preprocessing_hash=metadata["preprocessing_hash"],
    )
    return checkpoint_path
