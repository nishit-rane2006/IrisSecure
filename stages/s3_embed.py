from pathlib import Path

from config import CFG
from src.caching import hash_file, hash_obj, stage
from src.stages_impl import embed_impl


def _artifacts() -> Path:
    return CFG.work_dir / "artifacts" / "embed"


def _outputs():
    return [_artifacts() / "embeddings.npz"]


def _inputs_hash() -> str:
    tr = CFG.work_dir / "artifacts" / "train"
    pre = CFG.work_dir / "artifacts" / "preprocess"
    return (
        hash_file(tr / "best_model.pth")
        + ":"
        + hash_obj({"image_size": CFG.image_size, "embedding_dim": CFG.embedding_dim})
        + ":"
        + hash_file(pre / "train.csv")
        + ":"
        + hash_file(pre / "val.csv")
        + ":"
        + hash_file(pre / "test.csv")
    )


@stage("embed", inputs_fn=_inputs_hash, outputs_fn=_outputs)
def run_embed(batch_size=None, num_workers=None, tta=True):
    embed_impl(_artifacts(), batch_size=batch_size, num_workers=num_workers, tta=tta)
