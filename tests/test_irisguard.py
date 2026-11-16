import numpy as np

from irisguard.afsdd import AFSDD, AFSDDConfig
from irisguard.dataset_inspector import discover_dataset_root, inspect_dataset, write_manifest_and_statistics
from irisguard.research import run_experiment_suite
from irisguard.template_manager import (
    IrisGuardPolicy,
    QualityAwarePolicy,
    TemplateManager,
    UnprotectedAdaptivePolicy,
)


def _vec(seed: int = 0, dim: int = 512):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, size=dim)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)


def test_template_manager_keeps_enrollment_reference_immutable():
    mgr = TemplateManager(policy=UnprotectedAdaptivePolicy(acceptance_threshold=0.70), adaptation_rate=0.25)
    base = _vec(0)
    mgr.register("user1", base, quality=0.95)
    new = _vec(1)
    result = mgr.update_template("user1", new, quality=0.9, acceptance_similarity=0.85)
    assert result["accepted"] is True
    assert np.allclose(mgr.get_record("user1").enrollment_reference, base)
    assert not np.allclose(mgr.get_record("user1").current_template, base)
    assert mgr.get_record("user1").current_version == 1


def test_irisguard_policy_rejects_far_away_update():
    detector = AFSDD(AFSDDConfig(review_threshold=0.50, reject_threshold=0.75))
    detector.fit([_vec(9)])
    mgr = TemplateManager(policy=IrisGuardPolicy(detector), adaptation_rate=0.15)
    base = _vec(9)
    mgr.register("user2", base, quality=0.9)
    attack = -base.copy()
    result = mgr.update_template("user2", attack, quality=0.45, acceptance_similarity=-0.99)
    assert result["accepted"] is False
    assert result["decision"] in {"REVIEW", "REJECT", "LOCK"}


def test_quality_aware_policy_requires_quality_threshold():
    mgr = TemplateManager(policy=QualityAwarePolicy(acceptance_threshold=0.70, min_quality=0.60), adaptation_rate=0.20)
    base = _vec(2)
    mgr.register("user3", base, quality=0.9)
    candidate = _vec(3)
    result = mgr.update_template("user3", candidate, quality=0.50, acceptance_similarity=0.80)
    assert result["accepted"] is False
    assert mgr.get_record("user3").current_version == 0


def test_rollback_restores_previous_version():
    mgr = TemplateManager(policy=UnprotectedAdaptivePolicy(acceptance_threshold=0.70), adaptation_rate=0.30)
    base = _vec(4)
    mgr.register("user4", base)
    first = _vec(5)
    second = _vec(6)
    mgr.update_template("user4", first, quality=0.85, acceptance_similarity=0.8)
    mgr.update_template("user4", second, quality=0.85, acceptance_similarity=0.8)
    rollback = mgr.rollback("user4", version=1)
    assert rollback["rolled_back"] is True
    assert mgr.get_record("user4").current_version == 1


def test_dataset_root_discovery_finds_uploaded_casia_data():
    root = discover_dataset_root()
    assert root is not None
    assert "IrisSecure-Dataset" in str(root) or "CASIA-Iris-Thousand" in str(root)


def test_dataset_manifest_generation_creates_manifest_and_stats():
    manifest_path, stats_path, df, stats = write_manifest_and_statistics()
    assert manifest_path.exists()
    assert stats_path.exists()
    assert not df.empty
    assert stats["subjects"] > 0
    assert stats["valid_images"] > 0


def test_experiment_suite_produces_expected_summary():
    report = run_experiment_suite()
    assert "policies" in report
    assert "attack_summary" in report
    for policy_name, summary in report["policies"].items():
        assert "legitimate_updates" in summary
        assert "gradual_poisoning_updates" in summary
        assert "legitimate_drift" in summary
        assert "poison_drift" in summary
