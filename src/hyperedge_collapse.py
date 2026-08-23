"""T4 — Hyper-edge coarsening rule (design AND implementation).

When a partition assigns nodes to super-nodes, each original hyper-edge e of
arity n with members M is transformed according to how many of its endpoints
land in each super-node:

  Case A  m = n  (all members in one super-node):
    The edge is *fully internal* — it contributes to that super-node's internal
    cohesion weight but disappears from the coarse graph.

  Case B  1 < m < n  (members split across k < n distinct super-nodes):
    The edge becomes a *reduced coarse hyper-edge* over those k super-nodes,
    with a weight equal to the number of original members it contributes from
    each side.  The arity drops from n to k.

  Case C  k >= 3 distinct super-nodes  (a special case of B for k>=3):
    The edge remains a genuine coarse hyper-edge over all k super-nodes.
    It is NOT decomposed into pairwise edges — that would be the lossy
    projection P4 forbids.  The weight vector records how many endpoints each
    super-node contributed.

What the rule loses: internal edge weight (Case A) is not recoverable at
coarser levels; Case B/C edges aggregate original multiplicities into a single
weight, losing the identities of the individual members.

This module is imported and called by hierarchy.py — the rule the method
actually applies at every coarsening step.
"""

from collections import defaultdict


def collapse_hyperedges(
    hyperedges: list[dict],
    node_to_supernode: dict[str, str],
) -> list[dict]:
    """Collapse hyperedges under the given node→supernode mapping.

    Parameters
    ----------
    hyperedges : list of edge dicts (each has 'id', 'members', 'relation_type', 'year')
    node_to_supernode : dict mapping each node id to its super-node id

    Returns
    -------
    List of coarse hyper-edge dicts, each with:
        id, relation_type, year, members (super-node ids, deduplicated),
        weight (total original members involved),
        member_weights ({supernode_id: count_of_original_members}),
        source_edge_ids (original edge ids that collapsed into this one)
    """
    # Group by (frozenset of super-nodes, relation_type) to merge parallel edges
    bucket: dict[tuple, dict] = defaultdict(lambda: {
        "members": None,
        "weight": 0,
        "member_weights": defaultdict(int),
        "source_edge_ids": [],
        "relation_type": None,
        "year": None,
    })

    for e in hyperedges:
        members = e.get("members", [])
        # Only include members that have a supernode mapping
        mapped = [node_to_supernode[m] for m in members if m in node_to_supernode]
        if not mapped:
            continue

        distinct_supernodes = sorted(set(mapped))
        key = (frozenset(distinct_supernodes), e.get("relation_type", ""))

        rec = bucket[key]
        rec["members"] = distinct_supernodes
        rec["relation_type"] = e.get("relation_type")
        rec["year"] = e.get("year")
        rec["weight"] += len(mapped)
        rec["source_edge_ids"].append(e["id"])
        for sn in mapped:
            rec["member_weights"][sn] += 1

    coarse_edges = []
    for i, ((sn_set, rel), rec) in enumerate(bucket.items()):
        distinct = rec["members"]
        n_distinct = len(distinct)

        # Classify according to the three cases
        if n_distinct == 1:
            case = "A_internal"       # all endpoints in one super-node
        elif n_distinct == 2:
            case = "B_reduced_pair"   # reduced to a pair
        else:
            case = "C_coarse_hyper"   # genuine coarse hyper-edge (>=3 super-nodes)

        coarse_edges.append({
            "id": f"coarse_{i:05d}",
            "relation_type": rec["relation_type"],
            "year": rec["year"],
            "members": distinct,
            "weight": rec["weight"],
            "member_weights": dict(rec["member_weights"]),
            "case": case,
            "source_edge_ids": rec["source_edge_ids"],
        })

    return coarse_edges


def internal_weight(coarse_edges: list[dict]) -> dict[str, float]:
    """Return total internal weight per super-node (Case A edges)."""
    weights: dict[str, float] = defaultdict(float)
    for e in coarse_edges:
        if e["case"] == "A_internal":
            sn = e["members"][0]
            weights[sn] += e["weight"]
    return dict(weights)
