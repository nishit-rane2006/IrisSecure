"""SQLite persistence for immutable and adaptive biometric templates."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .afsdd import AFSDD, AFSDDConfig
from .template_manager import cosine_similarity


def _validated_embedding(embedding: np.ndarray, dimension: int = 512) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != dimension:
        raise ValueError(f"Expected a {dimension}-dimensional embedding, got {vector.size}.")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains NaN or infinite values.")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Embedding has zero norm.")
    return vector / norm


class SQLiteTemplateStore:
    """Stores templates and update events in local SQLite transactions."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    user_id TEXT PRIMARY KEY,
                    immutable_reference BLOB NOT NULL,
                    current_template BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    quality REAL NOT NULL,
                    locked INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS template_history (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES templates(user_id) ON DELETE CASCADE,
                    previous_version INTEGER NOT NULL,
                    new_version INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    anomaly_score REAL,
                    quality REAL NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS template_versions (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES templates(user_id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    current_template BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def register(self, user_id: str, embedding: np.ndarray, quality: float = 1.0) -> Dict[str, object]:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        vector = _validated_embedding(embedding)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                INSERT INTO templates
                    (user_id, immutable_reference, current_template, dimension, version, quality, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (user_id, vector.tobytes(), vector.tobytes(), int(vector.size), float(quality), now, now),
            )
            db.execute(
                """
                INSERT INTO template_history
                    (user_id, previous_version, new_version, decision, similarity, anomaly_score, quality, reason, created_at)
                VALUES (?, 0, 0, 'ENROLL', 1.0, NULL, ?, 'Initial trusted enrollment', ?)
                """,
                (user_id, float(quality), now),
            )
            db.execute(
                "INSERT INTO template_versions (user_id, version, current_template, created_at) VALUES (?, 0, ?, ?)",
                (user_id, vector.tobytes(), now),
            )
        return self.get(user_id)

    def get(self, user_id: str) -> Dict[str, object]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM templates WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown template identity: {user_id}")
        dim = int(row["dimension"])
        return {
            "user_id": row["user_id"],
            "immutable_reference": np.frombuffer(row["immutable_reference"], dtype=np.float32, count=dim).copy(),
            "current_template": np.frombuffer(row["current_template"], dtype=np.float32, count=dim).copy(),
            "version": int(row["version"]),
            "quality": float(row["quality"]),
            "locked": bool(row["locked"]),
        }

    def list_templates(self) -> List[Dict[str, object]]:
        with self._connect() as db:
            ids = [row["user_id"] for row in db.execute("SELECT user_id FROM templates ORDER BY user_id")]
        return [self.get(user_id) for user_id in ids]

    def update(
        self,
        user_id: str,
        embedding: np.ndarray,
        *,
        quality: float,
        authenticated: bool,
        detector: Optional[AFSDD] = None,
        adaptation_rate: float = 0.1,
    ) -> Dict[str, object]:
        record = self.get(user_id)
        incoming = _validated_embedding(embedding, record["immutable_reference"].size)
        similarity = cosine_similarity(record["current_template"], incoming)
        if not authenticated:
            return self._record_decision(user_id, record, "REJECT", similarity, quality, "Authentication failed")
        if record["locked"]:
            return self._record_decision(user_id, record, "LOCK", similarity, quality, "Template is locked")

        detector = detector or AFSDD(AFSDDConfig())
        with self._connect() as db:
            prior_reviews = db.execute(
                """
                SELECT COUNT(*) FROM template_history
                WHERE user_id = ? AND decision IN ('REVIEW', 'REJECT')
                """,
                (user_id,),
            ).fetchone()[0]
        detector.reviews = int(prior_reviews)
        score_parts = detector.score_update(
            record["immutable_reference"],
            record["current_template"],
            incoming,
            float(quality),
        )
        decision = detector.decide(score_parts)
        if decision != "ACCEPT":
            return self._record_decision(
                user_id,
                record,
                decision,
                similarity,
                quality,
                "AFSDD prevented adaptive update",
                float(score_parts["anomaly_score"]),
            )

        if not 0.0 < adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must be in (0, 1].")
        updated = (1.0 - adaptation_rate) * record["current_template"] + adaptation_rate * incoming
        updated = _validated_embedding(updated, record["immutable_reference"].size)
        now = time.time()
        next_version = record["version"] + 1
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                UPDATE templates SET current_template = ?, version = ?, quality = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (updated.tobytes(), next_version, float(quality), now, user_id),
            )
            db.execute(
                """
                INSERT INTO template_history
                    (user_id, previous_version, new_version, decision, similarity, anomaly_score, quality, reason, created_at)
                VALUES (?, ?, ?, 'ACCEPT', ?, ?, ?, 'AFSDD approved adaptive update', ?)
                """,
                (user_id, record["version"], next_version, similarity, score_parts["anomaly_score"], float(quality), now),
            )
            db.execute(
                "INSERT INTO template_versions (user_id, version, current_template, created_at) VALUES (?, ?, ?, ?)",
                (user_id, next_version, updated.tobytes(), now),
            )
        return {**self.get(user_id), "decision": "ACCEPT", "anomaly_score": score_parts["anomaly_score"]}

    def _record_decision(
        self,
        user_id: str,
        record: Dict[str, object],
        decision: str,
        similarity: float,
        quality: float,
        reason: str,
        anomaly_score: Optional[float] = None,
    ) -> Dict[str, object]:
        now = time.time()
        new_version = int(record["version"])
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE templates SET locked = ?, updated_at = ? WHERE user_id = ?",
                (int(decision == "LOCK"), now, user_id),
            )
            db.execute(
                """
                INSERT INTO template_history
                    (user_id, previous_version, new_version, decision, similarity, anomaly_score, quality, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, new_version, new_version, decision, similarity, anomaly_score, float(quality), reason, now),
            )
        return {**self.get(user_id), "decision": decision, "anomaly_score": anomaly_score}

    def history(self, user_id: str) -> List[Dict[str, object]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM template_history WHERE user_id = ? ORDER BY event_id",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def rollback(self, user_id: str, target_version: int) -> Dict[str, object]:
        if target_version < 0:
            raise ValueError("target_version must be non-negative.")
        if target_version == 0:
            target = self.get(user_id)["immutable_reference"]
        else:
            with self._connect() as db:
                event = db.execute(
                    "SELECT 1 FROM template_versions WHERE user_id = ? AND version = ?",
                    (user_id, target_version),
                ).fetchone()
                if event is None:
                    raise ValueError(f"No accepted template version {target_version} for {user_id}.")
                target_event = db.execute(
                    """
                    SELECT current_template FROM template_versions
                    WHERE user_id = ? AND version = ?
                    ORDER BY snapshot_id DESC LIMIT 1
                    """,
                    (user_id, target_version),
                ).fetchone()
                target = np.frombuffer(target_event[0], dtype=np.float32).copy()
        return self._set_rollback(user_id, target, target_version)

    def _set_rollback(self, user_id: str, target: np.ndarray, version: int) -> Dict[str, object]:
        now = time.time()
        current = self.get(user_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE templates SET current_template = ?, version = ?, locked = 0, updated_at = ? WHERE user_id = ?",
                (target.tobytes(), version, now, user_id),
            )
            db.execute(
                """
                INSERT INTO template_history
                    (user_id, previous_version, new_version, decision, similarity, anomaly_score, quality, reason, created_at)
                VALUES (?, ?, ?, 'ROLLBACK', 1.0, NULL, ?, 'Trusted rollback', ?)
                """,
                (user_id, current["version"], version, current["quality"], now),
            )
        return self.get(user_id)
