from pathlib import Path

from config import CFG
from src.caching import hash_file, hash_obj, stage
from src.stages_impl import train_impl


def _artifacts() -> Path:
    return CFG.work_dir / "artifacts" / "train"


def _outputs():
    return [_artifacts() / "best_model.pth", _artifacts() / "train_history.json"]


def _inputs_hash() -> str:
    pre = CFG.work_dir / "artifacts" / "preprocess"
    cfg_sig = {
        "epochs": CFG.epochs,
        "batch_size": CFG.batch_size,
        "grad_accum_steps": CFG.grad_accum_steps,
        "head_lr": CFG.head_lr,
        "backbone_lr_scale": CFG.backbone_lr_scale,
        "arcface_s": CFG.arcface_s,
        "arcface_m": CFG.arcface_m,
        "embedding_dim": CFG.embedding_dim,
        "seed": CFG.seed,
        "max_classes": CFG.max_classes,
        "warmup_epochs": CFG.warmup_epochs,
    }
    return (
        hash_obj(cfg_sig)
        + ":"
        + hash_file(pre / "train.csv")
        + ":"
        + hash_file(pre / "val.csv")
    )


@stage("train", inputs_fn=_inputs_hash, outputs_fn=_outputs)
def run_train(epochs=None, batch_size=None, grad_accum=None, num_workers=None, max_classes=None):
    train_impl(
        _artifacts(),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        num_workers=num_workers,
        max_classes=max_classes,
    )
