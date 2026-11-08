from pathlib import Path

from config import CFG
from src.caching import hash_file, hash_obj, stage
from src.stages_impl import evaluate_impl


def _artifacts() -> Path:
    return CFG.work_dir / "artifacts" / "evaluate"


def _outputs():
    d = _artifacts()
    return [d / "report.json", d / "roc.png", d / "det.png", d / "scores.png", d / "far_frr.png"]


def _inputs_hash() -> str:
    em = CFG.work_dir / "artifacts" / "embed"
    pre = CFG.work_dir / "artifacts" / "preprocess"
    return (
        hash_file(em / "embeddings.npz")
        + ":"
        + hash_file(pre / "test.csv")
        + ":"
        + hash_obj({"max_impostor": CFG.max_impostor_pairs, "seed": CFG.seed})
    )


@stage("evaluate", inputs_fn=_inputs_hash, outputs_fn=_outputs)
def run_evaluate(split="test", max_impostor=None):
    evaluate_impl(_artifacts(), split=split, max_impostor=max_impostor)
