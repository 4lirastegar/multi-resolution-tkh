"""T5 — Re-run labels only (without rebuilding hierarchies).

Useful if you want to regenerate labels with a different prompt or model
without rerunning the expensive clustering step.  Reads existing
hierarchy_<year>.json files, relabels levels 0 and 1, and saves them back.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SNAPSHOT_DIR, SNAPSHOT_YEARS
from src.data_loading import load_tkh
from src.labeling import label_supernodes


if __name__ == "__main__":
    meta, nodes_by_id, hyperedges = load_tkh()

    for year in SNAPSHOT_YEARS:
        path = SNAPSHOT_DIR / f"hierarchy_{year}.json"
        if not path.exists():
            print(f"  SKIP {year}: file not found (run 02_build_hierarchies.py first)")
            continue
        with open(path) as f:
            h = json.load(f)
        print(f"  Labelling {year} …")
        h = label_supernodes(h, nodes_by_id, snapshot_year=year)
        for sn in h["supernodes"]:
            if sn["level"] == 0 and sn.get("label"):
                print(f"    [L0] {sn['label'][:70]}")
        with open(path, "w") as f:
            json.dump(h, f, indent=2)
        print(f"  Saved {path}")

    print("Done.")
