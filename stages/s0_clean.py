from pathlib import Path

from config import CFG
from src.caching import hash_dir, hash_obj, stage
from src.stages_impl import clean_impl


def _artifacts() -> Path:
    return CFG.work_dir / "artifacts" / "clean"


def _outputs():
    d = _artifacts()
    return [d / "clean.csv", d / "cleaning_report.json"]


def _inputs_hash() -> str:
    cfg_sig = {
        "data_root": str(CFG.data_root),
        "min_resolution": CFG.clean_min_resolution,
        "max_resolution": CFG.clean_max_resolution,
        "min_images_per_class": CFG.clean_min_images_per_class,
        "duplicate_hamming_threshold": CFG.clean_duplicate_threshold,
    }
    return hash_obj(cfg_sig) + ":" + hash_dir(CFG.data_root, patterns=("*",))


@stage("clean", inputs_fn=_inputs_hash, outputs_fn=_outputs)
def run_clean():
    clean_impl(_artifacts())
