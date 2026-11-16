"""Dataset inspection and manifest generation for the local CASIA-Iris-Thousand upload."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import pandas as pd


def discover_dataset_root(start_roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Locate the uploaded IrisSecure-Dataset or CASIA-Iris-Thousand directory."""
    roots: List[Path] = []
    if start_roots is not None:
        roots.extend(Path(p) for p in start_roots)
    roots.extend([
        Path(__file__).resolve().parents[1],
        Path.cwd(),
        Path.home(),
    ])

    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for candidate in [root / "IrisSecure-Dataset", root / "CASIA-Iris-Thousand"]:
            if candidate.exists() and candidate.is_dir():
                if candidate.name == "CASIA-Iris-Thousand":
                    inner = candidate / "CASIA-Iris-Thousand"
                    if inner.exists():
                        return inner.resolve()
                    return candidate.resolve()
                casia_root = candidate / "CASIA-Iris-Thousand"
                if casia_root.exists():
                    inner = casia_root / "CASIA-Iris-Thousand"
                    if inner.exists():
                        return inner.resolve()
                    return casia_root.resolve()
                return candidate.resolve()

        for match in root.rglob("IrisSecure-Dataset"):
            if match.is_dir() and match not in seen:
                seen.add(match.resolve())
                return match.resolve()

        for match in root.rglob("CASIA-Iris-Thousand"):
            if match.is_dir() and match not in seen:
                seen.add(match.resolve())
                inner = match / "CASIA-Iris-Thousand"
                if inner.exists():
                    return inner.resolve()
                return match.resolve()

    return None


def _image_quality(gray: "cv2.Mat") -> float:
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 400.0)
    contrast = min(1.0, float(gray.std()) / 64.0)
    illumination = float(gray.mean())
    illumination_score = max(0.0, 1.0 - abs(illumination - 127.5) / 127.5)
    return float(max(0.0, min(1.0, 0.45 * sharpness + 0.35 * contrast + 0.20 * illumination_score)))


def inspect_dataset(dataset_root: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build a manifest and summary from the observed upload structure."""
    root = Path(dataset_root) if dataset_root is not None else discover_dataset_root()
    if root is None:
        raise FileNotFoundError(
            "IrisSecure-Dataset not found. Inspected repository and common workspace directories, but no uploaded CASIA-Iris-Thousand dataset was located."
        )

    root = root.resolve()
    if root.name == "IrisSecure-Dataset":
        actual = root / "CASIA-Iris-Thousand"
        if actual.exists():
            root = actual.resolve()

    if root.name == "CASIA-Iris-Thousand" and (root / "CASIA-Iris-Thousand").exists():
        root = (root / "CASIA-Iris-Thousand").resolve()

    subject_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories were found beneath {root}.")

    rows: List[Dict[str, object]] = []
    unreadable: List[str] = []
    missing_malformed: List[str] = []
    duplicates: set[str] = set()
    seen_paths: set[str] = set()

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        for eye_dir in sorted(subject_dir.iterdir()):
            if not eye_dir.is_dir():
                continue
            eye = eye_dir.name.upper()
            if eye not in {"L", "R"}:
                missing_malformed.append(str(eye_dir))
                continue
            for image_path in sorted(eye_dir.iterdir()):
                if not image_path.is_file():
                    continue
                suffix = image_path.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                key = str(image_path.resolve())
                if key in seen_paths:
                    duplicates.add(key)
                    continue
                seen_paths.add(key)
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    unreadable.append(str(image_path))
                    rows.append({
                        "image_path": str(image_path),
                        "subject_id": int(subject_id),
                        "eye_side": eye,
                        "identity_id": f"{subject_id}_{eye}",
                        "image_filename": image_path.name,
                        "image_width": None,
                        "image_height": None,
                        "image_format": suffix.lstrip("."),
                        "image_quality": 0.0,
                        "dataset_split": "raw",
                        "is_valid": False,
                    })
                    continue

                h, w = img.shape[:2]
                rows.append({
                    "image_path": str(image_path),
                    "subject_id": int(subject_id),
                    "eye_side": eye,
                    "identity_id": f"{subject_id}_{eye}",
                    "image_filename": image_path.name,
                    "image_width": int(w),
                    "image_height": int(h),
                    "image_format": suffix.lstrip("."),
                    "image_quality": float(_image_quality(img)),
                    "dataset_split": "raw",
                    "is_valid": True,
                })

    df = pd.DataFrame(rows, columns=[
        "image_path",
        "subject_id",
        "eye_side",
        "identity_id",
        "image_filename",
        "image_width",
        "image_height",
        "image_format",
        "image_quality",
        "dataset_split",
        "is_valid",
    ])

    stats: Dict[str, object] = {
        "dataset_root": str(root),
        "subjects": int(df["subject_id"].nunique()) if not df.empty else 0,
        "left_eye_identities": int(df[df["eye_side"] == "L"]["identity_id"].nunique()) if not df.empty else 0,
        "right_eye_identities": int(df[df["eye_side"] == "R"]["identity_id"].nunique()) if not df.empty else 0,
        "total_identities": int(df["identity_id"].nunique()) if not df.empty else 0,
        "valid_images": int(df["is_valid"].sum()) if not df.empty else 0,
        "unreadable_or_corrupt_images": len(unreadable),
        "duplicate_files": len(duplicates),
        "missing_or_malformed_identity_dirs": len(missing_malformed),
        "images_per_identity": {
            ident: int(count)
            for ident, count in df.groupby("identity_id")["image_path"].count().sort_index().items()
        },
        "image_formats": {fmt: int(count) for fmt, count in df["image_format"].astype(str).value_counts().items()},
        "dimension_summary": {
            "min_width": int(df["image_width"].dropna().min()) if not df["image_width"].dropna().empty else None,
            "max_width": int(df["image_width"].dropna().max()) if not df["image_width"].dropna().empty else None,
            "min_height": int(df["image_height"].dropna().min()) if not df["image_height"].dropna().empty else None,
            "max_height": int(df["image_height"].dropna().max()) if not df["image_height"].dropna().empty else None,
        },
        "image_quality_summary": {
            "mean": float(df[df["is_valid"]]["image_quality"].mean()) if df["is_valid"].any() else 0.0,
            "median": float(df[df["is_valid"]]["image_quality"].median()) if df["is_valid"].any() else 0.0,
            "min": float(df[df["is_valid"]]["image_quality"].min()) if df["is_valid"].any() else 0.0,
            "max": float(df[df["is_valid"]]["image_quality"].max()) if df["is_valid"].any() else 0.0,
        },
        "notes": [
            "The uploader contains a nested CASIA-Iris-Thousand directory; the actual subject folders are located beneath the final CASIA-Iris-Thousand folder.",
            "Left and right eyes are treated as separate identities in the manifest.",
        ],
    }

    return df, stats


def write_manifest_and_statistics(output_dir: Optional[Path] = None, dataset_root: Optional[Path] = None) -> Tuple[Path, Path, pd.DataFrame, Dict[str, object]]:
    project_root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "data" / "processed"
    project_root.mkdir(parents=True, exist_ok=True)
    df, stats = inspect_dataset(dataset_root)
    manifest_path = project_root / "dataset_manifest.csv"
    stats_path = project_root / "dataset_statistics.json"
    df.to_csv(manifest_path, index=False)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    return manifest_path, stats_path, df, stats


def load_manifest(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset_manifest.csv"
    return pd.read_csv(path)
