"""Validate registered artifacts and checkpoint metadata without regenerating them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    from config import CFG
    from irisguard.artifact_manager import ArtifactError, ArtifactManager

    manager = ArtifactManager()
    names = ("model", "embeddings", "embedding_labels", "calibration")
    failures = []
    for name in names:
        try:
            path = manager.validate(name)
            print(f"OK {name}: {path}")
        except ArtifactError as exc:
            print(f"MISSING/INVALID {name}: {exc}")
            failures.append(name)

    if "model" not in failures:
        import torch

        model_path = manager.validate("model")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if int(checkpoint.get("embedding_dim", -1)) != CFG.embedding_dim:
            raise SystemExit("Model checkpoint is incompatible with configured embedding dimension.")
        print(f"Model metadata: {checkpoint['embedding_dim']}D, {checkpoint['num_classes']} classes")

    if failures and args.require_all:
        raise SystemExit(f"Artifact validation failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
