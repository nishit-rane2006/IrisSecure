"""Run the controlled synthetic IrisGuard experiment suite explicitly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from config import CFG
    from irisguard.artifact_manager import ArtifactError, ArtifactManager
    from irisguard.research import run_experiment_suite

    output = CFG.artifacts_dir / "experiments" / f"synthetic-seed-{args.seed}.json"
    manager = ArtifactManager()
    if output.exists() and not args.force:
        try:
            manager.validate(f"experiment_seed_{args.seed}")
            print(f"Reusing completed experiment: {output}")
            return
        except ArtifactError:
            pass

    result = run_experiment_suite(seed=args.seed)
    summary = {
        "experiment_type": "synthetic embedding-level simulation",
        "seed": args.seed,
        "policies": result["policies"],
        "attack_summary": {
            name: {
                key: value
                for key, value in item.items()
                if key != "updates"
            }
            for name, item in result["attack_summary"].items()
        },
        "note": "Synthetic embedding perturbations are controlled simulations, not real presentation attacks.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with manager.atomic_destination(output) as temp_path:
        temp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manager.record(
        f"experiment_seed_{args.seed}",
        output,
        metadata={"seed": args.seed, "type": summary["experiment_type"]},
    )
    print(f"Experiment results saved: {output}")


if __name__ == "__main__":
    main()
