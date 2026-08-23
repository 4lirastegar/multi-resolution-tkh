"""T2–T5 — Build the laminar hierarchy for every snapshot (warm-started, T3)
and write:
  outputs/snapshots/hierarchy_<year>.json   (one per snapshot)
  outputs/temporal_events.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR, SNAPSHOT_DIR, SNAPSHOT_YEARS
from src.data_loading import load_tkh, slice_snapshot
from src.embeddings import get_clustering_embeddings
from src.hierarchy import build_hierarchy
from src.labeling import label_supernodes
from src.temporal import build_event_log


def save_hierarchy(h: dict, year: int):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"hierarchy_{year}.json"
    # Serialise — coarse_edges_per_level can be large, keep it
    with open(path, "w") as f:
        json.dump(h, f, indent=2)
    print(f"  Saved {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    print("Loading TKH …")
    meta, nodes_by_id, hyperedges = load_tkh()

    hierarchies: dict[int, dict] = {}

    for year in SNAPSHOT_YEARS:
        print(f"\n=== Snapshot ≤{year} ===")
        nodes, edges = slice_snapshot(nodes_by_id, hyperedges, year)
        print(f"  {len(nodes)} nodes, {len(edges)} edges")

        print("  Computing embeddings …")
        embeddings = get_clustering_embeddings(nodes)

        print("  Building hierarchy …")
        h = build_hierarchy(
            nodes_by_id=nodes,
            hyperedges=edges,
            embeddings=embeddings,
            snapshot_year=year,
        )

        n_per_level = {}
        for level_str, n2sn in h["node_to_level_supernode"].items():
            n_per_level[int(level_str)] = len(set(n2sn.values()))
        print(f"  Super-nodes per level: {n_per_level}")

        print("  Labelling levels 0 and 1 …")
        h = label_supernodes(h, nodes_by_id, snapshot_year=year)

        # Print sample labels
        for sn in h["supernodes"]:
            if sn["level"] == 0 and sn["label"]:
                print(f"    [L0] {sn['label'][:60]}")

        hierarchies[year] = h
        save_hierarchy(h, year)

    print("\n=== Building temporal event log ===")
    events = build_event_log(hierarchies, years=SNAPSHOT_YEARS)
    print(f"  {len(events)} events across {len(SNAPSHOT_YEARS)-1} transitions")

    event_counts = {}
    for e in events:
        event_counts[e["event"]] = event_counts.get(e["event"], 0) + 1
    print(f"  Event breakdown: {event_counts}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    events_path = OUTPUT_DIR / "temporal_events.json"
    with open(events_path, "w") as f:
        json.dump({"events": events, "event_counts": event_counts}, f, indent=2)
    print(f"  Saved {events_path}")

    print("\nDone.")
