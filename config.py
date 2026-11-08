from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ----- paths -----
    data_root: Path = Path("data/CASIA-Iris-Thousand")
    work_dir: Path = Path("work")
    models_dir: Path = Path("models")
    ckpt_dir: Path = Path("models/checkpoints")
    db_dir: Path = Path("database")
    outputs_dir: Path = Path("outputs")

    # ----- model -----
    backbone: str = "resnet50"
    embedding_dim: int = 512
    num_classes: int = 2000
    image_size: int = 224
    dropout: float = 0.4
    pretrained: bool = True

    # ----- ArcFace -----
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    label_smoothing: float = 0.0

    # ----- training -----
    epochs: int = 30
    warmup_epochs: int = 3
    batch_size: int = 16
    grad_accum_steps: int = 4
    head_lr: float = 1e-3
    backbone_lr_scale: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 6
    seed: int = 42
    grad_clip: float = 5.0
    ema_decay: float = 0.999
    use_amp: bool = True
    channels_last: bool = True
    max_classes: int = 0

    # ----- splits -----
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ----- evaluation -----
    max_impostor_pairs: int = 50_000

    # ----- recognition thresholds -----
    verify_threshold: float = 0.75
    duplicate_threshold: float = 0.85

    # ----- dataset cleaning -----
    clean_min_resolution: int = 64
    clean_max_resolution: int = 4096
    clean_min_images_per_class: int = 2
    clean_duplicate_threshold: int = 2

    # ----- security -----
    encryption_enabled: bool = False
    cancelable_templates: bool = False

    def ensure_dirs(self):
        for d in (
            self.work_dir,
            self.models_dir,
            self.ckpt_dir,
            self.db_dir,
            self.outputs_dir,
            self.work_dir / "artifacts",
            self.work_dir / "cache",
        ):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
