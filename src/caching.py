"""Content-addressed caching for pipeline stages."""
import hashlib
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import CFG


HASH_INDEX = CFG.work_dir / "cache" / "stage_hashes.json"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def hash_obj(obj: Any) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, Path):
        obj = str(obj)
    return _hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def hash_files(paths: List[Path]) -> str:
    entries = []
    for p in sorted(paths):
        if p.exists() and p.is_file():
            st = p.stat()
            entries.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
    return _hash_bytes("|".join(entries).encode())


def hash_dir(root: Path, patterns=("*",)) -> str:
    if not root.exists():
        return "missing"
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return hash_files([f for f in files if f.is_file()])


def manifest_path(stage: str) -> Path:
    return CFG.work_dir / "artifacts" / stage / "manifest.json"


def load_manifest(stage: str) -> Optional[Dict]:
    p = manifest_path(stage)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_manifest(stage: str, data: Dict):
    p = manifest_path(stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_hash_index() -> Dict[str, str]:
    if HASH_INDEX.exists():
        with open(HASH_INDEX) as f:
            return json.load(f)
    return {}


def save_hash_index(idx: Dict[str, str]):
    HASH_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_INDEX, "w") as f:
        json.dump(idx, f, indent=2)


class StageResult:
    def __init__(self, stage: str, ran: bool, reason: str, duration: float = 0.0):
        self.stage = stage
        self.ran = ran
        self.reason = reason
        self.duration = duration

    def __repr__(self):
        status = "RAN " if self.ran else "SKIP"
        return f"[{status}] {self.stage:12s} ({self.duration:6.2f}s) — {self.reason}"


def stage(name: str, inputs_fn: Callable[[], str], outputs_fn: Callable[[], List[Path]]):
    """Cache a stage based on its input hash + existence of its outputs."""

    def decorator(fn):
        def wrapper(*args, force: bool = False, **kwargs):
            t0 = time.time()
            idx = load_hash_index()
            current_hash = inputs_fn()

            outputs = outputs_fn()
            outputs_exist = len(outputs) > 0 and all(p.exists() for p in outputs)

            if not force and outputs_exist and idx.get(name) == current_hash:
                return StageResult(
                    name,
                    ran=False,
                    reason=f"cached (hash={current_hash})",
                    duration=time.time() - t0,
                )

            reason = (
                "forced" if force else
                "outputs missing" if not outputs_exist else
                "inputs changed"
            )
            print(f"\n{'='*66}\n>> Running stage: {name}  ({reason})\n{'='*66}")

            fn(*args, **kwargs)

            outputs = outputs_fn()
            missing = [p for p in outputs if not p.exists()]
            if missing:
                raise RuntimeError(f"Stage '{name}' finished but did not produce: {missing}")

            idx[name] = current_hash
            save_hash_index(idx)
            save_manifest(
                name,
                {
                    "stage": name,
                    "input_hash": current_hash,
                    "outputs": [str(p) for p in outputs],
                    "timestamp": time.time(),
                    "duration": time.time() - t0,
                    "status": "ok",
                },
            )
            return StageResult(name, ran=True, reason=reason, duration=time.time() - t0)

        return wrapper

    return decorator
