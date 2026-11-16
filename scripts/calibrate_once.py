"""Calibrate verification and AFSDD thresholds from the validation split only."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import preprocessing_config, register_existing_checkpoint


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot calibrate from a zero-norm embedding.")
    return vector / norm


def _roc_metrics(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Validation split must contain genuine and impostor pairs for calibration.")
    y = np.concatenate((np.ones(genuine.size), np.zeros(impostor.size)))
    scores = np.concatenate((genuine, impostor))
    fpr, tpr, thresholds = roc_curve(y, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    return {
        "roc_auc": float(auc(fpr, tpr)),
        "eer": eer,
        "eer_threshold": float(thresholds[index]),
        "n_genuine_pairs": int(genuine.size),
        "n_impostor_pairs": int(impostor.size),
    }


def _pair_scores(embeddings: np.ndarray, labels: np.ndarray, max_impostor: int, seed: int):
    rng = np.random.default_rng(seed)
    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(int(index))
    genuine_pairs = [
        (left, right)
        for indices in by_label.values()
        for left, right in combinations(indices, 2)
    ]
    impostor_pairs = set()
    attempts = 0
    while len(impostor_pairs) < max_impostor and attempts < max_impostor * 30:
        left, right = rng.integers(0, len(labels), size=2)
        attempts += 1
        if left != right and labels[left] != labels[right]:
            impostor_pairs.add((min(int(left), int(right)), max(int(left), int(right))))
    genuine = np.fromiter(
        (float(np.dot(embeddings[a], embeddings[b])) for a, b in genuine_pairs),
        dtype=np.float32,
        count=len(genuine_pairs),
    )
    impostor = np.fromiter(
        (float(np.dot(embeddings[a], embeddings[b])) for a, b in impostor_pairs),
        dtype=np.float32,
        count=len(impostor_pairs),
    )
    return genuine, impostor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-impostor-pairs", type=int, default=20_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from config import CFG, PROJECT_ROOT
    from irisguard.afsdd import AFSDD
    from irisguard.artifact_manager import ArtifactError, ArtifactManager
    from scripts.common import preprocessing_config

    manager = ArtifactManager()
    target = CFG.calibration_dir / "calibration.json"
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
                "calibration",
                model_hash=model_hash,
                dataset_fingerprint=dataset_fingerprint,
                preprocessing_hash=preprocessing_hash,
            )
            print(f"Reusing validated calibration: {target}")
            return
        except ArtifactError:
            pass

    try:
        embedding_path = manager.validate(
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
    except ArtifactError as exc:
        raise SystemExit(
            f"{exc}\nExtract or migrate embeddings with `python .\\scripts\\extract_embeddings.py`."
        ) from exc

    labels_path = CFG.embedding_dir / "embedding_labels.csv"
    labels_frame = pd.read_csv(labels_path)
    if "split" not in labels_frame.columns:
        raise ValueError(f"Embedding labels must include a split column: {labels_path}")
    val_mask = labels_frame["split"] == "val"
    val_indices = np.flatnonzero(val_mask.to_numpy())
    val_frame = labels_frame.loc[val_mask].reset_index(drop=True)
    if val_frame.empty:
        raise ValueError("No validation embeddings were found.")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != CFG.embedding_dim or len(embeddings) != len(labels_frame):
        raise ValueError("Embedding matrix and label metadata are incompatible.")
    val_embeddings = embeddings[val_indices]
    val_labels = val_frame["label"].to_numpy(dtype=np.int64)
    genuine, impostor = _pair_scores(val_embeddings, val_labels, args.max_impostor_pairs, CFG.seed)
    verification = _roc_metrics(genuine, impostor)

    manifest_path = CFG.artifacts_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
    quality_by_path = {}
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, usecols=["image_path", "image_quality"])
        quality_by_path = dict(zip(manifest["image_path"].astype(str), manifest["image_quality"].astype(float)))

    within_identity_deviations = []
    temporal_deviations = []
    quality_values = []
    score_examples = []
    for label, group in val_frame.groupby("label", sort=True):
        group_indices = group.index.to_numpy(dtype=np.int64)
        if group_indices.size < 2:
            continue
        samples = np.asarray(val_embeddings[group_indices], dtype=np.float64)
        center = _normalize(samples.mean(axis=0))
        deviations = np.clip(1.0 - samples @ center, 0.0, 2.0)
        within_identity_deviations.extend(deviations.tolist())
        temporal_deviations.extend(
            np.clip(1.0 - np.sum(samples[:-1] * samples[1:], axis=1), 0.0, 2.0).tolist()
        )
        for row, sample, deviation in zip(group.to_dict("records"), samples, deviations):
            quality = quality_by_path.get(str(row["path"]), 0.5)
            quality_values.append(float(quality))

    variation = np.asarray(within_identity_deviations, dtype=np.float64)
    temporal = np.asarray(temporal_deviations, dtype=np.float64)
    qualities = np.asarray(quality_values, dtype=np.float64)
    reference_stats = {
        "mean_reference_deviation": float(np.mean(variation)),
        "std_reference_deviation": float(np.std(variation)) + 1e-6,
        "mean_temporal_drift": float(np.mean(temporal)) if temporal.size else 0.0,
        "std_temporal_drift": float(np.std(temporal)) + 1e-6 if temporal.size else 1e-6,
        "mean_quality": float(np.mean(qualities)) if qualities.size else 0.5,
        "std_quality": float(np.std(qualities)) + 1e-6 if qualities.size else 1e-6,
    }
    detector = AFSDD()
    detector.reference_stats.update(reference_stats)
    for label, group in val_frame.groupby("label", sort=True):
        group_indices = group.index.to_numpy(dtype=np.int64)
        if group_indices.size < 2:
            continue
        samples = np.asarray(val_embeddings[group_indices], dtype=np.float64)
        center = _normalize(samples.mean(axis=0))
        for row, sample in zip(group.to_dict("records"), samples):
            quality = quality_by_path.get(str(row["path"]), 0.5)
            score_examples.append(
                detector.score_update(center, center, sample, float(quality))["anomaly_score"]
            )
    anomaly = np.asarray(score_examples, dtype=np.float64)
    if anomaly.size == 0:
        raise ValueError("Not enough repeated validation samples to calibrate AFSDD variability.")

    review_threshold = float(np.quantile(anomaly, 0.99))
    reject_threshold = float(np.quantile(anomaly, 0.999))
    if reject_threshold <= review_threshold:
        reject_threshold = min(1.0, review_threshold + 1e-3)

    afsdd = {
        "quality_min": float(np.quantile(qualities, 0.10)) if qualities.size else 0.55,
        "reference_deviation_threshold": float(np.quantile(variation, 0.99)),
        "temporal_drift_threshold": float(np.quantile(temporal, 0.99)) if temporal.size else float(np.quantile(variation, 0.99)),
        "cumulative_drift_threshold": float(np.quantile(temporal, 0.999)) if temporal.size else float(np.quantile(variation, 0.999)),
        "review_threshold": review_threshold,
        "reject_threshold": reject_threshold,
        "lock_threshold": min(1.0, max(reject_threshold + 0.05, 0.95)),
        "lock_review_count": 3,
        "quality_penalty_weight": 0.20,
        "review_weight": 0.30,
        "rejection_weight": 0.25,
        "drift_weight": 0.25,
        "ewma_alpha": 0.35,
        "reference_stats": reference_stats,
        "cusum_reference_limit": float(np.quantile(temporal, 0.99)) if temporal.size else 0.0,
        "calibration_method": "validation-only empirical quantiles; no poisoning labels or test samples used",
    }
    prep = preprocessing_config()
    metadata = {
        "version": 1,
        "verification": verification,
        "verify_threshold": verification["eer_threshold"],
        "afsdd": afsdd,
        "source_split": "val",
        "n_validation_embeddings": int(len(val_frame)),
        "model_hash": model_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "preprocessing_hash": manager.fingerprint_json(prep),
        "random_seed": CFG.seed,
    }
    CFG.calibration_dir.mkdir(parents=True, exist_ok=True)
    with manager.atomic_destination(target) as temp_path:
        temp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    manager.record(
        "calibration",
        target,
        model_hash=metadata["model_hash"],
        dataset_fingerprint=metadata["dataset_fingerprint"],
        preprocessing_hash=metadata["preprocessing_hash"],
        metadata={"source_split": "val", "n_validation_embeddings": len(val_frame)},
    )
    print(f"Calibration saved: {target}")
    print(f"Validation ROC AUC={verification['roc_auc']:.4f}, EER={verification['eer']:.4f}, threshold={verification['eer_threshold']:.4f}")


if __name__ == "__main__":
    main()
