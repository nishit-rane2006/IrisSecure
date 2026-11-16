"""Adaptive Feature-Space Drift Detection (AFSDD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


def _as_float_array(x: Iterable[float], length: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float64)
    if length is not None and arr.size != length:
        raise ValueError(f"expected length {length}, got {arr.size}")
    return arr


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.astype(np.float64)
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    return float(np.dot(a, b))


@dataclass
class AFSDDConfig:
    quality_min: float = 0.55
    reference_deviation_threshold: float = 0.22
    temporal_drift_threshold: float = 0.18
    cumulative_drift_threshold: float = 0.20
    review_threshold: float = 0.52
    reject_threshold: float = 0.75
    lock_threshold: float = 0.90
    lock_review_count: int = 3
    quality_penalty_weight: float = 0.20
    review_weight: float = 0.30
    rejection_weight: float = 0.25
    drift_weight: float = 0.25
    ewma_alpha: float = 0.35


class AFSDD:
    """A lightweight but interpretable drift detector for adaptive templates."""

    def __init__(self, config: Optional[AFSDDConfig] = None):
        self.config = config or AFSDDConfig()
        self.reference_stats: Dict[str, float] = {
            "mean_reference_deviation": 0.08,
            "std_reference_deviation": 0.04,
            "mean_temporal_drift": 0.06,
            "std_temporal_drift": 0.03,
            "mean_quality": 0.80,
            "std_quality": 0.10,
        }
        self.reviews = 0
        self.locked = False

    def fit(self, genuine_embeddings: Iterable[np.ndarray], quality_scores: Optional[Iterable[float]] = None):
        emb = np.asarray(list(genuine_embeddings), dtype=np.float64)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.size == 0:
            return self

        self.reference_stats["mean_reference_deviation"] = float(np.mean(np.abs(1.0 - np.clip(np.linalg.norm(emb, axis=1), 0, 1))))
        self.reference_stats["std_reference_deviation"] = float(np.std(np.abs(1.0 - np.clip(np.linalg.norm(emb, axis=1), 0, 1))))

        if quality_scores is not None:
            q = _as_float_array(quality_scores)
            self.reference_stats["mean_quality"] = float(np.mean(q))
            self.reference_stats["std_quality"] = float(np.std(q))
        return self

    def _ewma(self, history: List[float], value: float) -> float:
        if not history:
            return value
        return float((1.0 - self.config.ewma_alpha) * history[-1] + self.config.ewma_alpha * value)

    def score_update(
        self,
        original_reference: np.ndarray,
        current_template: np.ndarray,
        incoming_embedding: np.ndarray,
        quality: float,
        temporal_history: Optional[List[float]] = None,
        drift_history: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        ref = _normalize(original_reference)
        cur = _normalize(current_template)
        inc = _normalize(incoming_embedding)

        ref_dev = 1.0 - cosine_similarity(ref, inc)
        temporal_drift = 1.0 - cosine_similarity(cur, inc)
        cumulative_drift = 1.0 - cosine_similarity(ref, cur)
        quality_penalty = max(0.0, 1.0 - quality)
        deviation_from_mean = max(0.0, (ref_dev - self.reference_stats["mean_reference_deviation"]) / (self.reference_stats["std_reference_deviation"] + 1e-6))
        temporal_from_mean = max(0.0, (temporal_drift - self.reference_stats["mean_temporal_drift"]) / (self.reference_stats["std_temporal_drift"] + 1e-6))

        ewma_step = 0.0
        if temporal_history:
            ewma_step = self._ewma(temporal_history, temporal_drift)
        ewma_drift = 0.0
        if drift_history:
            ewma_drift = self._ewma(drift_history, cumulative_drift)

        anomaly = (
            self.config.review_weight * ref_dev
            + self.config.drift_weight * temporal_drift
            + self.config.rejection_weight * max(cumulative_drift, ewma_drift)
            + self.config.quality_penalty_weight * quality_penalty
            + 0.5 * deviation_from_mean
            + 0.5 * temporal_from_mean
            + 0.35 * ewma_step
        )

        return {
            "reference_deviation": float(ref_dev),
            "temporal_drift": float(temporal_drift),
            "cumulative_drift": float(cumulative_drift),
            "quality_penalty": float(quality_penalty),
            "intra_class_excess": float(max(deviation_from_mean, temporal_from_mean)),
            "ewma_temporal": float(ewma_step),
            "ewma_cumulative": float(ewma_drift),
            "anomaly_score": float(np.clip(anomaly, 0.0, 1.0)),
        }

    def decide(self, score_dict: Dict[str, float]) -> str:
        score = float(score_dict["anomaly_score"])
        if score >= self.config.reject_threshold:
            return "REJECT"
        if score >= self.config.review_threshold:
            return "REVIEW"
        if self.locked and self.reviews >= self.config.lock_review_count:
            return "LOCK"
        return "ACCEPT"

    def record_decision(self, verdict: str):
        if verdict == "REVIEW":
            self.reviews += 1
        elif verdict in {"REJECT", "LOCK"}:
            self.reviews = max(0, self.reviews - 1)
        if verdict == "LOCK":
            self.locked = True

    def explain(self, score_dict: Dict[str, float], verdict: str) -> Dict[str, float | str]:
        return {
            "verdict": verdict,
            "anomaly_score": score_dict["anomaly_score"],
            "reference_deviation": score_dict["reference_deviation"],
            "temporal_drift": score_dict["temporal_drift"],
            "cumulative_drift": score_dict["cumulative_drift"],
            "quality_penalty": score_dict["quality_penalty"],
            "intra_class_excess": score_dict["intra_class_excess"],
        }
