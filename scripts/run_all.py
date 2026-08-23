"""Regenerate every deliverable from scratch.

Steps:
  01_describe_data.py      → outputs/t1_snapshot_stats.json
  02_build_hierarchies.py  → outputs/snapshots/hierarchy_<year>.json
                             outputs/temporal_events.json
  04_evaluate.py           → outputs/metrics.json

Run with:
  python scripts/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_describe_data.py",
    "02_build_hierarchies.py",
    "04_evaluate.py",
]

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n{'='*60}\n=== {script} ===\n{'='*60}")
        result = subprocess.run(
            [sys.executable, str(here / script)], check=False
        )
        if result.returncode != 0:
            print(f"ERROR: {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll steps completed successfully.")
