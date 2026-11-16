import json
from dataclasses import replace

import numpy as np
import pytest

from config.hardware_config import detect_hardware, resource_plan
from irisguard.afsdd import AFSDD, AFSDDConfig
from irisguard.artifact_manager import ArtifactError, ArtifactManager
from irisguard.sqlite_store import SQLiteTemplateStore


def _embedding(seed=1):
    vector = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_detected_backend_is_supported_and_cpu_remains_available(monkeypatch):
    monkeypatch.delenv("IRISGUARD_BACKEND", raising=False)
    info = detect_hardware()
    assert "cpu" in info.supported_backends
    assert info.selected_backend in info.supported_backends
    assert info.cpu_threads <= info.logical_processors
    assert info.dataloader_workers < info.logical_processors


def test_resource_plan_scales_with_available_memory(monkeypatch):
    for name in ("IRISGUARD_BATCH_SIZE", "IRISGUARD_GRAD_ACCUM_STEPS"):
        monkeypatch.delenv(name, raising=False)
    info = detect_hardware()
    low_memory = replace(
        info,
        selected_backend="cpu",
        available_ram_bytes=3 * 1024**3,
        gpu_memory_bytes=None,
        cpu_threads=4,
        dataloader_workers=0,
    )
    high_memory = replace(
        low_memory,
        available_ram_bytes=20 * 1024**3,
        dataloader_workers=3,
    )
    low_plan = resource_plan(low_memory)
    high_plan = resource_plan(high_memory)
    assert low_plan.training_batch_size == 2
    assert high_plan.training_batch_size == 16
    assert low_plan.gradient_accumulation_steps == 16
    assert high_plan.gradient_accumulation_steps == 2
    assert low_plan.dataloader_workers < high_plan.dataloader_workers


def test_resource_plan_honors_explicit_batch_override(monkeypatch):
    monkeypatch.setenv("IRISGUARD_BATCH_SIZE", "6")
    monkeypatch.delenv("IRISGUARD_GRAD_ACCUM_STEPS", raising=False)
    plan = resource_plan(detect_hardware())
    assert plan.training_batch_size == 6
    assert plan.embedding_batch_size == 12
    assert plan.gradient_accumulation_steps == 6


def test_resource_plan_uses_available_accelerator_memory(monkeypatch):
    for name in ("IRISGUARD_BATCH_SIZE", "IRISGUARD_GRAD_ACCUM_STEPS"):
        monkeypatch.delenv(name, raising=False)
    info = replace(
        detect_hardware(),
        selected_backend="cuda",
        gpu_memory_bytes=12 * 1024**3,
    )
    plan = resource_plan(info)
    assert plan.training_batch_size == 16
    assert plan.embedding_batch_size == 32
    assert plan.use_mixed_precision is True
    assert plan.channels_last is True


def test_artifact_integrity_and_compatibility(tmp_path):
    manager = ArtifactManager(tmp_path / "artifacts")
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"checkpoint")
    manager.record("model", artifact, model_hash="model-hash", metadata={"dim": 512})
    assert manager.validate("model", model_hash="model-hash") == artifact
    with pytest.raises(ArtifactError, match="stale model hash"):
        manager.validate("model", model_hash="different")
    artifact.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="SHA-256"):
        manager.validate("model")


def test_sqlite_template_update_persists_reference_and_rollback(tmp_path):
    db_path = tmp_path / "templates.sqlite3"
    manager = SQLiteTemplateStore(db_path)
    reference = _embedding()
    manager.register("alice", reference, quality=0.9)
    detector = AFSDD(AFSDDConfig(review_threshold=0.99, reject_threshold=0.999, lock_threshold=1.0))
    result = manager.update(
        "alice",
        reference,
        quality=0.95,
        authenticated=True,
        detector=detector,
        adaptation_rate=0.2,
    )
    assert result["decision"] == "ACCEPT"
    assert result["version"] == 1

    reopened = SQLiteTemplateStore(db_path)
    restored = reopened.get("alice")
    assert restored["version"] == 1
    assert np.allclose(restored["immutable_reference"], reference)
    assert len(reopened.history("alice")) == 2
    rolled_back = reopened.rollback("alice", 0)
    assert rolled_back["version"] == 0
    assert np.allclose(rolled_back["current_template"], reference)
    assert np.allclose(rolled_back["immutable_reference"], reference)


def test_sqlite_requires_authentication_before_template_update(tmp_path):
    manager = SQLiteTemplateStore(tmp_path / "templates.sqlite3")
    reference = _embedding()
    manager.register("bob", reference)
    result = manager.update("bob", _embedding(2), quality=0.9, authenticated=False)
    assert result["decision"] == "REJECT"
    assert result["version"] == 0
    assert np.allclose(result["immutable_reference"], reference)


def test_artifact_manifest_is_json(tmp_path):
    manager = ArtifactManager(tmp_path / "artifacts")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    manager.record("calibration", path)
    assert json.loads(manager.manifest_path.read_text(encoding="utf-8"))["artifacts"]["calibration"]["compatibility_status"] == "compatible"
