from pathlib import Path

from config import CFG
from src.caching import hash_file, hash_obj, stage
from src.stages_impl import preprocess_impl


def _artifacts() -> Path:
    return CFG.work_dir / "artifacts" / "preprocess"


def _outputs():
    d = _artifacts()
    return [
        d / "train.csv",
        d / "val.csv",
        d / "test.csv",
        d / "label_map.csv",
        d / "eda" / "images_per_class.png",
        d / "eda" / "splits.png",
    ]


def _inputs_hash() -> str:
    clean_csv = CFG.work_dir / "artifacts" / "clean" / "clean.csv"
    cfg_sig = {
        "seed": CFG.seed,
        "train_ratio": CFG.train_ratio,
        "val_ratio": CFG.val_ratio,
        "test_ratio": CFG.test_ratio,
    }
    return hash_obj(cfg_sig) + ":" + hash_file(clean_csv)


@stage("preprocess", inputs_fn=_inputs_hash, outputs_fn=_outputs)
def run_preprocess():
    preprocess_impl(_artifacts())
