"""Adaptive template management and update-policy enforcement."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .afsdd import AFSDD, AFSDDConfig


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.astype(np.float64)
    return (v / norm).astype(np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


@dataclass
class TemplateRecord:
    user_id: str
    enrollment_reference: np.ndarray
    current_template: np.ndarray
    current_version: int = 0
    versions: List[Dict[str, Any]] = field(default_factory=list)
    quality_history: List[float] = field(default_factory=list)
    drift_history: List[float] = field(default_factory=list)
    temporal_history: List[float] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    locked: bool = False

    def __post_init__(self):
        self.enrollment_reference = _normalize(self.enrollment_reference)
        self.current_template = _normalize(self.current_template)


class TemplatePolicy:
    def should_update(self, record: TemplateRecord, incoming: np.ndarray, quality: float, similarity: float, details: Dict[str, Any]) -> bool:
        raise NotImplementedError


class StaticTemplatePolicy(TemplatePolicy):
    def should_update(self, record: TemplateRecord, incoming: np.ndarray, quality: float, similarity: float, details: Dict[str, Any]) -> bool:
        return False


class UnprotectedAdaptivePolicy(TemplatePolicy):
    def __init__(self, acceptance_threshold: float = 0.75):
        self.acceptance_threshold = acceptance_threshold

    def should_update(self, record: TemplateRecord, incoming: np.ndarray, quality: float, similarity: float, details: Dict[str, Any]) -> bool:
        return similarity >= self.acceptance_threshold


class QualityAwarePolicy(TemplatePolicy):
    def __init__(self, acceptance_threshold: float = 0.75, min_quality: float = 0.55):
        self.acceptance_threshold = acceptance_threshold
        self.min_quality = min_quality

    def should_update(self, record: TemplateRecord, incoming: np.ndarray, quality: float, similarity: float, details: Dict[str, Any]) -> bool:
        return similarity >= self.acceptance_threshold and quality >= self.min_quality


class IrisGuardPolicy(TemplatePolicy):
    def __init__(self, detector: Optional[AFSDD] = None, decision_threshold: float = 0.52):
        self.detector = detector or AFSDD(AFSDDConfig(review_threshold=decision_threshold))
        self.decision_threshold = decision_threshold

    def should_update(self, record: TemplateRecord, incoming: np.ndarray, quality: float, similarity: float, details: Dict[str, Any]) -> bool:
        score = self.detector.score_update(
            record.enrollment_reference,
            record.current_template,
            incoming,
            quality,
            temporal_history=record.temporal_history,
            drift_history=record.drift_history,
        )
        verdict = self.detector.decide(score)
        details["afsdd"] = score
        details["afsdd_verdict"] = verdict
        return verdict == "ACCEPT"


class TemplateManager:
    """Tracks immutable references, adaptive templates, and rollback history."""

    def __init__(self, policy: Optional[TemplatePolicy] = None, adaptation_rate: float = 0.15):
        self.policy = policy or StaticTemplatePolicy()
        self.adaptation_rate = adaptation_rate
        self.records: Dict[str, TemplateRecord] = {}

    def register(self, user_id: str, embedding: np.ndarray, quality: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        emb = _normalize(embedding)
        record = TemplateRecord(
            user_id=user_id,
            enrollment_reference=emb.copy(),
            current_template=emb.copy(),
            quality_history=[float(quality)],
            audit_log=[{"event": "enrollment", "quality": float(quality), "timestamp": time.time(), "metadata": metadata or {}}],
        )
        self.records[user_id] = record
        return record

    def get_record(self, user_id: str) -> TemplateRecord:
        if user_id not in self.records:
            raise KeyError(f"No record for {user_id}")
        return self.records[user_id]

    def evaluate_similarity(self, user_id: str, incoming: np.ndarray) -> float:
        record = self.get_record(user_id)
        return cosine_similarity(record.current_template, incoming)

    def update_template(self, user_id: str, incoming: np.ndarray, quality: float = 1.0, metadata: Optional[Dict[str, Any]] = None, acceptance_similarity: Optional[float] = None) -> Dict[str, Any]:
        record = self.get_record(user_id)
        incoming = _normalize(incoming)
        similarity = acceptance_similarity if acceptance_similarity is not None else cosine_similarity(record.current_template, incoming)
        details: Dict[str, Any] = {
            "similarity": float(similarity),
            "quality": float(quality),
            "metadata": metadata or {},
        }

        if record.locked:
            details["decision"] = "LOCK"
            record.audit_log.append({"event": "blocked", "decision": "LOCK", "similarity": float(similarity), "quality": float(quality), "timestamp": time.time()})
            return details

        should_update = self.policy.should_update(record, incoming, quality, similarity, details)
        decision = details.get("afsdd_verdict", "ACCEPT" if should_update else "REJECT")

        if decision == "LOCK":
            record.locked = True
        elif decision == "REVIEW" and not should_update:
            decision = "REVIEW"

        if not should_update:
            record.audit_log.append({
                "event": "update_rejected",
                "decision": decision,
                "similarity": float(similarity),
                "quality": float(quality),
                "timestamp": time.time(),
            })
            return {**details, "decision": decision, "accepted": False}

        previous = record.current_template.copy()
        new_template = _normalize((1.0 - self.adaptation_rate) * record.current_template + self.adaptation_rate * incoming)
        drift = 1.0 - cosine_similarity(previous, new_template)
        record.current_template = new_template
        record.current_version += 1
        record.quality_history.append(float(quality))
        record.drift_history.append(float(drift))
        record.temporal_history.append(float(1.0 - cosine_similarity(previous, incoming)))
        version_entry = {
            "version": record.current_version,
            "previous": previous.copy(),
            "current": new_template.copy(),
            "similarity": float(similarity),
            "quality": float(quality),
            "drift": float(drift),
            "timestamp": time.time(),
            "decision": decision,
        }
        record.versions.append(version_entry)
        record.audit_log.append({
            "event": "update_accepted",
            "decision": decision,
            "similarity": float(similarity),
            "quality": float(quality),
            "drift": float(drift),
            "template_version": record.current_version,
            "timestamp": time.time(),
        })
        return {**details, "decision": decision, "accepted": True, "version": record.current_version, "drift": float(drift)}

    def rollback(self, user_id: str, version: Optional[int] = None) -> Dict[str, Any]:
        record = self.get_record(user_id)
        if version is None and len(record.versions) > 0:
            version = record.versions[-1]["version"]
        target_version = version if version is not None else 0
        if target_version < 0:
            raise ValueError("version must be >= 0")

        if len(record.versions) == 0:
            return {"user_id": user_id, "rolled_back": False, "current_version": record.current_version}

        target = record.versions[0] if target_version == 0 else next((v for v in reversed(record.versions) if v["version"] == target_version), None)
        if target is None:
            raise ValueError(f"Unknown version {target_version} for {user_id}")

        if target_version == 0:
            record.current_template = record.enrollment_reference.copy()
            record.current_version = 0
            record.audit_log.append({"event": "rollback", "to_version": 0, "timestamp": time.time()})
            return {"user_id": user_id, "rolled_back": True, "current_version": 0}

        record.current_template = target["current"].copy()
        record.current_version = target_version
        record.audit_log.append({"event": "rollback", "to_version": target_version, "timestamp": time.time()})
        return {"user_id": user_id, "rolled_back": True, "current_version": target_version}

    def list_users(self) -> List[str]:
        return sorted(self.records)
