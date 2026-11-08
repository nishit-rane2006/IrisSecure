"""Lightweight passive-liveness heuristic (pluggable)."""
import cv2
import numpy as np


def _laplacian_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _specular_score(gray: np.ndarray) -> float:
    thresh = np.percentile(gray, 99.5)
    return float((gray >= thresh).mean())


def _high_freq_ratio(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray.astype(np.float32))))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    low = f[cy - r:cy + r, cx - r:cx + r].sum()
    total = f.sum() + 1e-8
    return float(1.0 - low / total)


def estimate_liveness(gray: np.ndarray) -> float:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(gray, (224, 224))

    s_lap = float(np.clip(_laplacian_score(g) / 300.0, 0, 1))
    s_spec = float(np.clip(_specular_score(g) / 0.02, 0, 1))
    s_hf = float(np.clip((_high_freq_ratio(g) - 0.3) / 0.4, 0, 1))
    return 0.4 * s_lap + 0.3 * s_spec + 0.3 * s_hf
