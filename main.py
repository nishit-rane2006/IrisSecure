"""IrisGuard research entry point.

Run a synthetic evaluation demonstrating adaptive template behavior and AFSDD
protection against gradual poisoning.
"""

import json

from irisguard.research import run_experiment_suite


if __name__ == "__main__":
    report = run_experiment_suite()
    print(json.dumps(report, indent=2, default=str))
