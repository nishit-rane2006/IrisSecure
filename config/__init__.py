"""Project configuration with paths rooted at the repository directory."""

from dataclasses import dataclass
from pathlib import Path

from irisguard.dataset_inspector import discover_dataset_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    discovered = discover_dataset_root([PROJECT_ROOT])
    if discovered is not None:
        return discovered
    return PROJECT_ROOT / "data" / "CASIA-Iris-Thousand"


@dataclass
class Config:
    data_root: Path = _default_data_root()
    work_dir: Path = PROJECT_ROOT / "work"
    models_dir: Path = PROJECT_ROOT / "models"
    ckpt_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    db_dir: Path = PROJECT_ROOT / "database"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    backbone: str = "resnet50"
    embedding_dim: int = 512
    num_classes: int = 2000
    image_size: int = 224
    dropout: float = 0.4
    pretrained: bool = True

    arcface_s: float = 30.0
    arcface_m: float = 0.5
    label_smoothing: float = 0.0

    epochs: int = 8
    warmup_epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4
    head_lr: float = 1e-3
    backbone_lr_scale: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 0
    seed: int = 42
    grad_clip: float = 5.0
    ema_decay: float = 0.999
    use_amp: bool = False
    channels_last: bool = False
    max_classes: int = 0
    early_stopping_patience: int = 3

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_impostor_pairs: int = 50_000
    verify_threshold: float = 0.75
    duplicate_threshold: float = 0.85

    clean_min_resolution: int = 64
    clean_max_resolution: int = 4096
    clean_min_images_per_class: int = 2
    clean_duplicate_threshold: int = 2

    encryption_enabled: bool = False
    cancelable_templates: bool = False

    @property
    def model_path(self) -> Path:
        return self.work_dir / "artifacts" / "train" / "best_model.pth"

    @property
    def embedding_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    @property
    def calibration_dir(self) -> Path:
        return self.artifacts_dir / "calibration"

    @property
    def template_db_path(self) -> Path:
        return self.db_dir / "templates.sqlite3"

    def ensure_dirs(self) -> None:
        for directory in (
            self.work_dir,
            self.models_dir,
            self.ckpt_dir,
            self.db_dir,
            self.outputs_dir,
            self.artifacts_dir,
            self.embedding_dir,
            self.calibration_dir,
            self.work_dir / "artifacts",
            self.work_dir / "cache",
        ):
            directory.mkdir(parents=True, exist_ok=True)


CFG = Config()
