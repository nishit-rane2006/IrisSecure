"""Back up one or more explicitly named registered artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Artifact names from artifacts/manifest.json")
    args = parser.parse_args()

    from irisguard.artifact_manager import ArtifactError, ArtifactManager

    manager = ArtifactManager()
    for name in args.names:
        path = manager.validate(name)
        backup = manager.backup(path)
        print(f"{name}: {backup}")


if __name__ == "__main__":
    main()
