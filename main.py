"""IrisGuard research entry point.

Runs dataset discovery, manifest generation, and a synthetic adaptive-template
poisoning experiment using the uploaded local CASIA-Iris-Thousand dataset.
"""

import json

from irisguard.dataset_inspector import write_manifest_and_statistics
from irisguard.research import run_experiment_suite


if __name__ == "__main__":
    manifest_path, stats_path, _, stats = write_manifest_and_statistics()
    print(f"Dataset manifest: {manifest_path}")
    print(f"Dataset statistics: {stats_path}")
    print(json.dumps({k: v for k, v in stats.items() if k in {"subjects", "total_identities", "valid_images", "unreadable_or_corrupt_images", "duplicate_files"}}, indent=2))
    report = run_experiment_suite()
    print(json.dumps(report, indent=2, default=str))
