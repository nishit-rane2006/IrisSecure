"""Synthetic experiment runner for adaptive template poisoning scenarios."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .afsdd import AFSDD, AFSDDConfig
from .template_manager import (
    IrisGuardPolicy,
    QualityAwarePolicy,
    StaticTemplatePolicy,
    TemplateManager,
    UnprotectedAdaptivePolicy,
    cosine_similarity,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.astype(np.float64)
    return (v / norm).astype(np.float64)


@dataclass
class AttackTrace:
    scenario: str
    update_number: int
    incoming_embedding: np.ndarray
    current_template: np.ndarray
    original_reference: np.ndarray
    cosine_similarity_score: float
    drift: float
    quality: float
    decision: str
    attack_success: bool


def _generate_legitimate_sequence(base_embedding: np.ndarray, steps: int = 6, noise_scale: float = 0.08, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    rng = rng or np.random.default_rng(0)
    seq: List[np.ndarray] = []
    for _ in range(steps):
        direction = rng.normal(0.0, 1.0, size=base_embedding.shape)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        perturb = rng.normal(0.0, noise_scale, size=base_embedding.shape)
        emb = _normalize(base_embedding + perturb + 0.10 * direction)
        seq.append(emb)
    return seq


def _generate_poisoning_sequence(base_embedding: np.ndarray, steps: int = 12, drift_magnitude: float = 0.55, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    rng = rng or np.random.default_rng(1)
    direction = rng.normal(0.0, 1.0, size=base_embedding.shape)
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    seq: List[np.ndarray] = []
    for step in range(steps):
        factor = (step + 1) / max(steps, 1)
        emb = _normalize(base_embedding + (factor * drift_magnitude) * direction + rng.normal(0.0, 0.03, size=base_embedding.shape))
        seq.append(emb)
    return seq


def run_attack_scenarios(seed: int = 42) -> Dict[str, Dict[str, float] | List[AttackTrace]]:
    rng = np.random.default_rng(seed)
    base = _normalize(rng.normal(0.0, 1.0, size=512))
    manager = TemplateManager(policy=IrisGuardPolicy(AFSDD(AFSDDConfig())), adaptation_rate=0.15)
    manager.register("alice", base, quality=0.90)
    detector = AFSDD(AFSDDConfig())
    detector.fit([base])

    scenarios = {
        "legitimate_adaptation": _generate_legitimate_sequence(base, steps=5, noise_scale=0.05, rng=rng),
        "single_step_poisoning": _generate_poisoning_sequence(base, steps=1, drift_magnitude=0.70, rng=rng),
        "gradual_poisoning": _generate_poisoning_sequence(base, steps=8, drift_magnitude=0.38, rng=rng),
        "slow_drift": _generate_poisoning_sequence(base, steps=12, drift_magnitude=0.22, rng=rng),
    }

    results: Dict[str, Dict[str, float] | List[AttackTrace]] = {}
    for name, seq in scenarios.items():
        story: List[AttackTrace] = []
        manager_local = TemplateManager(policy=IrisGuardPolicy(AFSDD(AFSDDConfig()), decision_threshold=0.52), adaptation_rate=0.15)
        manager_local.register("alice", base, quality=0.90)
        for step, incoming in enumerate(seq, start=1):
            quality = 0.92 if "legitimate" in name else 0.72 if "single_step" in name else 0.64
            sim = cosine_similarity(manager_local.get_record("alice").current_template, incoming)
            score = detector.score_update(base, manager_local.get_record("alice").current_template, incoming, quality, manager_local.get_record("alice").temporal_history, manager_local.get_record("alice").drift_history)
            verdict = detector.decide(score)
            accepted = manager_local.update_template("alice", incoming, quality=quality, metadata={"scenario": name}, acceptance_similarity=sim)
            attack_success = 1.0 - cosine_similarity(base, manager_local.get_record("alice").current_template) > 0.35
            story.append(
                AttackTrace(
                    scenario=name,
                    update_number=step,
                    incoming_embedding=incoming.copy(),
                    current_template=manager_local.get_record("alice").current_template.copy(),
                    original_reference=base.copy(),
                    cosine_similarity_score=sim,
                    drift=float(1.0 - cosine_similarity(base, manager_local.get_record("alice").current_template)),
                    quality=float(quality),
                    decision=accepted.get("decision", verdict),
                    attack_success=bool(attack_success),
                )
            )
        results[name] = {"detected": sum(1 for x in story if x.decision in {"REVIEW", "REJECT", "LOCK"}), "attack_success": any(x.attack_success for x in story), "final_drift": story[-1].drift if story else 0.0, "updates": story}
    return results


def run_experiment_suite(seed: int = 42) -> Dict[str, object]:
    """Run the synthetic research evaluation for policy comparisons."""
    base = np.random.default_rng(seed + 7).normal(0.0, 1.0, size=512)
    base = _normalize(base)

    experiment = {}
    policies = {
        "static": StaticTemplatePolicy(),
        "unprotected": UnprotectedAdaptivePolicy(acceptance_threshold=0.75),
        "quality_aware": QualityAwarePolicy(acceptance_threshold=0.75, min_quality=0.55),
        "irisguard": IrisGuardPolicy(AFSDD(AFSDDConfig()), decision_threshold=0.52),
    }

    scenarios = {
        "legitimate": _generate_legitimate_sequence(base, steps=6, noise_scale=0.05, rng=np.random.default_rng(seed + 9)),
        "gradual_poisoning": _generate_poisoning_sequence(base, steps=8, drift_magnitude=0.35, rng=np.random.default_rng(seed + 10)),
    }

    for policy_name, policy in policies.items():
        summary = {"policy": policy_name, "legitimate_updates": 0, "gradual_poisoning_updates": 0, "legitimate_drift": 0.0, "poison_drift": 0.0}
        for scenario_name, seq in scenarios.items():
            manager = TemplateManager(policy=policy, adaptation_rate=0.15)
            manager.register("alice", base, quality=0.90)
            for step, incoming in enumerate(seq, start=1):
                quality = 0.9 if "legitimate" in scenario_name else 0.68
                s = cosine_similarity(manager.get_record("alice").current_template, incoming)
                result = manager.update_template("alice", incoming, quality=quality, metadata={"scenario": scenario_name}, acceptance_similarity=s)
                if result.get("accepted"):
                    if scenario_name == "legitimate":
                        summary["legitimate_updates"] += 1
                    else:
                        summary["gradual_poisoning_updates"] += 1
            drift = 1.0 - cosine_similarity(base, manager.get_record("alice").current_template)
            if scenario_name == "legitimate":
                summary["legitimate_drift"] = float(drift)
            else:
                summary["poison_drift"] = float(drift)
        experiment[policy_name] = summary

    return {
        "base_embedding_norm": float(np.linalg.norm(base)),
        "policies": experiment,
        "attack_summary": run_attack_scenarios(seed),
    }


if __name__ == "__main__":
    report = run_experiment_suite()
    print(json.dumps({k: v for k, v in report.items() if k != "attack_summary"}, indent=2, default=str))
    print("\nAttack summary:")
    print(json.dumps(report["attack_summary"], indent=2, default=str))
