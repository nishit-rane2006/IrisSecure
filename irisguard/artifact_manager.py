"""Atomic artifact registration, integrity checks, and compatibility validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, corrupt, or incompatible."""


class ArtifactManager:
    def __init__(self, root: Optional[Path] = None):
        if root is None:
            from config import CFG

            root = CFG.artifacts_dir
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def fingerprint_json(value: object) -> str:
        payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def read_manifest(self) -> Dict[str, object]:
        if not self.manifest_path.exists():
            return {"version": 1, "artifacts": {}}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("artifacts"), dict):
            raise ArtifactError(f"Invalid artifact manifest: {self.manifest_path}")
        return data

    def _write_manifest(self, data: Dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.manifest_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def backup(self, path: Path) -> Optional[Path]:
        path = Path(path)
        if not path.exists():
            return None
        backup_dir = self.root / "backups" / time.strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / path.name
        suffix = 1
        while target.exists():
            target = backup_dir / f"{path.stem}-{suffix}{path.suffix}"
            suffix += 1
        shutil.copy2(path, target)
        return target

    @contextmanager
    def atomic_destination(self, path: Path, preserve_existing: bool = True) -> Iterator[Path]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix + ".tmp", dir=destination.parent)
        os.close(fd)
        temp_path = Path(tmp_name)
        try:
            yield temp_path
            if preserve_existing and destination.exists():
                self.backup(destination)
            os.replace(temp_path, destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def record(
        self,
        name: str,
        path: Path,
        *,
        model_hash: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        preprocessing_hash: Optional[str] = None,
        compatibility_status: str = "compatible",
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(self.root.parent)
        except ValueError:
            relative = path
        entry: Dict[str, object] = {
            "path": str(relative),
            "version": 1,
            "sha256": self.sha256(path),
            "created_at": time.time(),
            "model_hash": model_hash,
            "dataset_fingerprint": dataset_fingerprint,
            "preprocessing_hash": preprocessing_hash,
            "compatibility_status": compatibility_status,
            "metadata": metadata or {},
        }
        manifest = self.read_manifest()
        artifacts = manifest["artifacts"]
        artifacts[name] = entry
        self._write_manifest(manifest)
        return entry

    def validate(
        self,
        name: str,
        *,
        model_hash: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        preprocessing_hash: Optional[str] = None,
    ) -> Path:
        manifest = self.read_manifest()
        entry = manifest["artifacts"].get(name)
        if not isinstance(entry, dict):
            raise ArtifactError(f"Artifact '{name}' is not registered. Run the documented preparation command.")

        artifact_path = Path(str(entry["path"]))
        if not artifact_path.is_absolute():
            artifact_path = self.root.parent / artifact_path
        if not artifact_path.is_file():
            raise ArtifactError(f"Artifact '{name}' is missing at {artifact_path}.")
        if self.sha256(artifact_path) != entry.get("sha256"):
            raise ArtifactError(f"Artifact '{name}' failed its SHA-256 integrity check: {artifact_path}")
        if entry.get("compatibility_status") != "compatible":
            raise ArtifactError(f"Artifact '{name}' is marked incompatible.")

        for field, expected in (
            ("model_hash", model_hash),
            ("dataset_fingerprint", dataset_fingerprint),
            ("preprocessing_hash", preprocessing_hash),
        ):
            actual = entry.get(field)
            if expected is not None and actual != expected:
                raise ArtifactError(f"Artifact '{name}' has stale {field.replace('_', ' ')}.")
        return artifact_path
