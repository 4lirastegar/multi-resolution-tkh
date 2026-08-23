"""T1 — Load the TKH export and slice temporal snapshots.

Snapshot definition: H(t) contains every hyper-edge with year <= t, and every
node that is a member of at least one such edge. Node inclusion is therefore
*evidence-based*: a node enters H(t) only once some corpus paper asserted an
edge involving it by year t, which respects P6 temporal honesty (we never rely
on origin_year, which can predate the corpus's first sighting of the entity).
Edges with a missing/invalid year are excluded from every snapshot and counted
as a data-quality issue.
"""

import json
from collections import Counter

from src.config import SNAPSHOT_YEARS, TKH_JSON


def load_tkh(path=TKH_JSON):
    """Return (meta, nodes_by_id, hyperedges) from the export JSON."""
    with open(path) as f:
        data = json.load(f)
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    return data["meta"], nodes_by_id, data["hyperedges"]


def slice_snapshot(nodes_by_id, hyperedges, t):
    """Return (snapshot_nodes_by_id, snapshot_edges) for cutoff year t."""
    edges = [
        e for e in hyperedges
        if isinstance(e.get("year"), int) and e["year"] <= t
    ]
    member_ids = {m for e in edges for m in e["members"] if m in nodes_by_id}
    nodes = {nid: nodes_by_id[nid] for nid in member_ids}
    return nodes, edges


def describe_snapshot(nodes, edges, prev=None):
    """Compute the T1 statistics for one snapshot.

    `prev` is the (nodes, edges) pair of the previous snapshot, used to
    report growth/change between snapshots.
    """
    arities = [len(e["members"]) for e in edges]
    arity_counter = Counter(arities)
    edge_years = [e["year"] for e in edges]

    stats = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "node_types": dict(Counter(n["type"] for n in nodes.values())),
        "relation_types": dict(Counter(e["relation_type"] for e in edges)),
        "edge_year_span": [min(edge_years), max(edge_years)] if edge_years else None,
        "arity": {
            "min": min(arities),
            "max": max(arities),
            "mean": round(sum(arities) / len(arities), 2),
            "median": sorted(arities)[len(arities) // 2],
            "share_arity_gt_2": round(sum(a > 2 for a in arities) / len(arities), 3),
            "histogram_coarse": {
                "2": arity_counter[2],
                "3-5": sum(v for k, v in arity_counter.items() if 3 <= k <= 5),
                "6-10": sum(v for k, v in arity_counter.items() if 6 <= k <= 10),
                "11-20": sum(v for k, v in arity_counter.items() if 11 <= k <= 20),
                ">20": sum(v for k, v in arity_counter.items() if k > 20),
            },
        },
    }

    if prev is not None:
        prev_nodes, prev_edges = prev
        prev_edge_ids = {e["id"] for e in prev_edges}
        stats["growth_vs_previous"] = {
            "new_nodes": len(set(nodes) - set(prev_nodes)),
            "new_edges": len([e for e in edges if e["id"] not in prev_edge_ids]),
            "node_growth_pct": round(100 * (len(nodes) / len(prev_nodes) - 1), 1),
            "edge_growth_pct": round(100 * (len(edges) / len(prev_edges) - 1), 1),
        }
    return stats


def data_quality_report(nodes_by_id, hyperedges):
    """Corpus-wide data-quality checks (run once on the full export)."""
    node_ids = set(nodes_by_id)
    edges_with_missing_members = [
        e["id"] for e in hyperedges if any(m not in node_ids for m in e["members"])
    ]
    edges_without_year = [
        e["id"] for e in hyperedges if not isinstance(e.get("year"), int)
    ]
    covered = {m for e in hyperedges for m in e["members"]}
    isolated_nodes = sorted(node_ids - covered)
    empty_surface = [
        n["id"] for n in nodes_by_id.values()
        if not (n.get("surface_form") or "").strip()
    ]
    dup_counter = Counter(
        (n["type"], (n.get("surface_form") or "").strip().lower())
        for n in nodes_by_id.values()
    )
    duplicate_surface_forms = sum(v - 1 for v in dup_counter.values() if v > 1)
    node_years = [n["year"] for n in nodes_by_id.values() if isinstance(n.get("year"), int)]
    nodes_without_year = len(nodes_by_id) - len(node_years)

    return {
        "edges_with_missing_member_ids": len(edges_with_missing_members),
        "edges_without_valid_year": len(edges_without_year),
        "isolated_nodes_never_in_any_edge": len(isolated_nodes),
        "nodes_with_empty_surface_form": len(empty_surface),
        "duplicate_(type,surface_form)_extra_nodes": duplicate_surface_forms,
        "nodes_without_valid_year": nodes_without_year,
        "node_year_range": [min(node_years), max(node_years)] if node_years else None,
    }


def build_all_snapshots(path=TKH_JSON, years=SNAPSHOT_YEARS):
    """Return {year: (nodes, edges)} for every configured snapshot."""
    _, nodes_by_id, hyperedges = load_tkh(path)
    return {t: slice_snapshot(nodes_by_id, hyperedges, t) for t in years}
