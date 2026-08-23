"""T1 — Per-snapshot description of the evolving hypergraph.

Prints the statistics and writes them to outputs/t1_snapshot_stats.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR, SNAPSHOT_YEARS
from src.data_loading import (
    data_quality_report,
    describe_snapshot,
    load_tkh,
    slice_snapshot,
)

if __name__ == "__main__":
    meta, nodes_by_id, hyperedges = load_tkh()

    report = {"snapshots": {}, "data_quality": data_quality_report(nodes_by_id, hyperedges)}

    prev = None
    for t in SNAPSHOT_YEARS:
        nodes, edges = slice_snapshot(nodes_by_id, hyperedges, t)
        report["snapshots"][t] = describe_snapshot(nodes, edges, prev=prev)
        prev = (nodes, edges)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "t1_snapshot_stats.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out_path}")
