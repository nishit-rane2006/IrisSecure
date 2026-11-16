"""Restore a single artifact file from an explicitly selected backup."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Artifact key to update in the manifest")
    parser.add_argument("--source", required=True, help="Exact backup file path")
    parser.add_argument("--destination", required=True, help="Exact artifact file path to restore")
    args = parser.parse_args()

    from irisguard.artifact_manager import ArtifactManager

    manager = ArtifactManager()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    if not source.is_file():
        raise SystemExit(f"Backup file not found: {source}")
    old_entry = manager.read_manifest()["artifacts"].get(args.name, {})
    with manager.atomic_destination(destination) as temp_path:
        shutil.copy2(source, temp_path)
    manager.record(
        args.name,
        destination,
        model_hash=old_entry.get("model_hash"),
        dataset_fingerprint=old_entry.get("dataset_fingerprint"),
        preprocessing_hash=old_entry.get("preprocessing_hash"),
        metadata=old_entry.get("metadata", {}),
    )
    print(f"Restored {source} -> {destination}")


if __name__ == "__main__":
    main()
