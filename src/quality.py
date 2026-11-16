"""Lightweight iris image quality estimates for update reliability checks."""

import cv2
import numpy as np


def estimate_image_quality(image: np.ndarray) -> float:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A non-empty image array is required.")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Expected a grayscale or RGB image, got shape {image.shape}.")
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 400.0)
    contrast = min(1.0, float(gray.std()) / 64.0)
    illumination = float(gray.mean())
    illumination_score = max(0.0, 1.0 - abs(illumination - 127.5) / 127.5)
    return float(np.clip(0.45 * sharpness + 0.35 * contrast + 0.20 * illumination_score, 0.0, 1.0))
