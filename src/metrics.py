from typing import Dict

import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_biometric_metrics(genuine: np.ndarray, impostor: np.ndarray) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    def tar_at_far(target_far: float) -> float:
        i = int(np.searchsorted(fpr, target_far, side="right") - 1)
        i = max(0, min(i, len(tpr) - 1))
        return float(tpr[i])

    return {
        "roc_auc": float(roc_auc),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tar_at_far_1e-2": tar_at_far(1e-2),
        "tar_at_far_1e-3": tar_at_far(1e-3),
        "tar_at_far_1e-4": tar_at_far(1e-4),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "separation_gap": float(genuine.mean() - impostor.mean()),
        "n_genuine_pairs": int(len(genuine)),
        "n_impostor_pairs": int(len(impostor)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
