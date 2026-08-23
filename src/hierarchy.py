"""T2 — Multi-resolution laminar hierarchy over the hypergraph.

Algorithm
---------
1. Semantic signal: sentence-transformer embeddings of surface_form (MiniLM).
2. Structural signal: hypergraph co-occurrence matrix — for each pair (u,v) of
   nodes, count the number of hyper-edges they share, weighted by 1/arity so
   large edges do not dominate.
3. Combined affinity: A = alpha * S_semantic + (1-alpha) * S_structural
   where alpha = ALPHA (config).  The trade-off is documented in the report.
4. Hierarchy: agglomerative clustering on the distance matrix (1-A) with Ward
   linkage produces a full dendrogram.  We cut at K levels to honour the hard
   size budgets N_k (config.LEVEL_BUDGETS).
5. Hyper-edge collapse (T4): at each level, hyperedge_collapse.collapse_hyperedges
   is called with the current node→supernode mapping to produce the coarse graph.

Properties guaranteed by construction: P1 (laminarity — agglomerative clustering
is a strict tree), P2 (size budgets enforced by dendrogram cutting).
Properties held empirically: P3 (coherence measured independently, §T6),
P5 (temporal stability via warm-start in temporal.py).
"""

import uuid
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.config import ALPHA, LEVEL_BUDGETS
from src.hyperedge_collapse import collapse_hyperedges


# ---------------------------------------------------------------------------
# Structural signal
# ---------------------------------------------------------------------------

def build_structural_affinity(nodes_by_id: dict, hyperedges: list[dict]) -> np.ndarray:
    """Build normalised structural co-occurrence matrix.

    Entry (i,j) = sum over edges containing both i and j of 1/arity(edge).
    Normalised to [0,1] by dividing by the maximum value.
    """
    node_ids = sorted(nodes_by_id.keys())
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    S = np.zeros((n, n), dtype=np.float32)

    for e in hyperedges:
        members = [m for m in e.get("members", []) if m in idx]
        arity = len(members)
        if arity < 2:
            continue
        w = 1.0 / arity
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = idx[members[a]], idx[members[b]]
                S[i, j] += w
                S[j, i] += w

    mx = S.max()
    if mx > 0:
        S /= mx
    return S, node_ids


# ---------------------------------------------------------------------------
# Combined affinity
# ---------------------------------------------------------------------------

def build_affinity(
    nodes_by_id: dict,
    hyperedges: list[dict],
    embeddings: dict[str, np.ndarray],
    alpha: float = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (affinity_matrix, node_ids) combining semantic and structural signals."""
    if alpha is None:
        alpha = ALPHA

    node_ids = sorted(nodes_by_id.keys())
    idx = {nid: i for i, nid in enumerate(node_ids)}

    # Semantic similarity (cosine)
    vecs = np.array([embeddings[nid] for nid in node_ids], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    S_sem = vecs @ vecs.T  # cosine similarity, shape (n, n)
    S_sem = np.clip(S_sem, 0, 1)

    # Structural co-occurrence
    S_struct, _ = build_structural_affinity(nodes_by_id, hyperedges)

    affinity = alpha * S_sem + (1 - alpha) * S_struct
    return affinity, node_ids


# ---------------------------------------------------------------------------
# Dendrogram cutting to enforce size budgets
# ---------------------------------------------------------------------------

def cut_dendrogram(Z: np.ndarray, n_nodes: int, max_clusters: int) -> np.ndarray:
    """Cut the linkage matrix Z so we get at most max_clusters clusters."""
    for k in range(max_clusters, 0, -1):
        labels = fcluster(Z, k, criterion="maxclust")
        if len(set(labels)) <= max_clusters:
            return labels
    return fcluster(Z, 1, criterion="maxclust")


# ---------------------------------------------------------------------------
# Main: build hierarchy for one snapshot
# ---------------------------------------------------------------------------

def build_hierarchy(
    nodes_by_id: dict,
    hyperedges: list[dict],
    embeddings: dict[str, np.ndarray],
    snapshot_year: int,
    level_budgets: list[int] = None,
    warm_labels: dict[str, str] | None = None,
) -> dict:
    """Build a multi-resolution laminar hierarchy for one snapshot.

    Parameters
    ----------
    nodes_by_id   : snapshot nodes
    hyperedges    : snapshot hyper-edges
    embeddings    : clustering embeddings (MiniLM)
    snapshot_year : year label for this snapshot
    level_budgets : [N0, N1, N2, ...] hard max clusters per level (coarsest first)
    warm_labels   : {node_id: supernode_id} from the previous snapshot, used to
                    bias initial cluster assignments (warm-start for P5).

    Returns
    -------
    dict with keys:
        snapshot_year, node_ids, levels (list of level dicts),
        node_to_level_supernode {level: {node_id: supernode_id}},
        supernodes (list of supernode records),
        coarse_edges_per_level {level: list of coarse edges}
    """
    if level_budgets is None:
        level_budgets = LEVEL_BUDGETS

    affinity, node_ids = build_affinity(nodes_by_id, hyperedges, embeddings)

    # Warm-start: add a small affinity bonus between nodes that were in the
    # same top-level cluster in the previous snapshot.  This biases the
    # dendrogram toward the previous partition in stable regions without
    # hard-constraining it.  warm_labels is {node_id: supernode_id} from
    # the caller (level-0 mapping of the immediately preceding snapshot).
    if warm_labels:
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        bonus = 0.05
        sn_to_idxs: dict[str, list[int]] = defaultdict(list)
        for nid, snid in warm_labels.items():
            if nid in node_to_idx:
                sn_to_idxs[snid].append(node_to_idx[nid])
        for idxs in sn_to_idxs.values():
            if len(idxs) > 1:
                arr = np.array(idxs)
                affinity[np.ix_(arr, arr)] = np.minimum(
                    1.0, affinity[np.ix_(arr, arr)] + bonus
                )
        np.fill_diagonal(affinity, 1.0)

    n = len(node_ids)

    if n < 2:
        # degenerate snapshot
        return _trivial_hierarchy(nodes_by_id, hyperedges, snapshot_year)

    # Convert affinity to distance (Ward linkage on Euclidean; we convert
    # cosine-based affinity to a condensed distance matrix).
    dist_matrix = np.clip(1.0 - affinity, 0, None)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)

    Z = linkage(condensed, method="ward")

    # Cut dendrogram at each level
    node_to_supernode_per_level: dict[int, dict[str, str]] = {}
    supernodes_per_level: dict[int, dict[str, dict]] = {}

    prev_labels = None
    for level_idx, budget in enumerate(level_budgets):
        actual_budget = min(budget, n)
        labels = cut_dendrogram(Z, n, actual_budget)

        # Build supernode id map
        # Use stable ids tied to level and cluster label
        label_to_snid: dict[int, str] = {}
        for lbl in set(labels):
            label_to_snid[lbl] = f"sn_y{snapshot_year}_l{level_idx}_c{lbl:04d}"

        node_to_sn = {node_ids[i]: label_to_snid[labels[i]] for i in range(n)}
        node_to_supernode_per_level[level_idx] = node_to_sn

        # Supernode records (members, centroid)
        sn_members: dict[str, list[str]] = defaultdict(list)
        for nid, snid in node_to_sn.items():
            sn_members[snid].append(nid)

        sn_dict = {}
        for snid, members in sn_members.items():
            sn_dict[snid] = {
                "id": snid,
                "level": level_idx,
                "snapshot_year": snapshot_year,
                "members": members,
                "label": None,   # filled by labeling.py
                "gloss": None,
                "parent_id": None,  # filled below
            }
        supernodes_per_level[level_idx] = sn_dict

        prev_labels = labels

    # Wire parent pointers (level k super-node's parent = level k-1 super-node
    # that contains the majority of its members)
    for level_idx in range(1, len(level_budgets)):
        coarser = node_to_supernode_per_level[level_idx - 1]
        for snid, sn in supernodes_per_level[level_idx].items():
            parent_votes: dict[str, int] = defaultdict(int)
            for member in sn["members"]:
                if member in coarser:
                    parent_votes[coarser[member]] += 1
            if parent_votes:
                sn["parent_id"] = max(parent_votes, key=parent_votes.__getitem__)

    # Also add finest level (one supernode per node, i.e., level K = leaf)
    leaf_level = len(level_budgets)
    leaf_to_sn = {nid: f"sn_y{snapshot_year}_l{leaf_level}_{nid}" for nid in node_ids}
    node_to_supernode_per_level[leaf_level] = leaf_to_sn
    leaf_sn_dict = {}
    for nid in node_ids:
        snid = leaf_to_sn[nid]
        parent_votes: dict[str, int] = defaultdict(int)
        coarser = node_to_supernode_per_level[leaf_level - 1]
        if nid in coarser:
            parent_votes[coarser[nid]] += 1
        leaf_sn_dict[snid] = {
            "id": snid,
            "level": leaf_level,
            "snapshot_year": snapshot_year,
            "members": [nid],
            "label": nodes_by_id[nid].get("surface_form", nid)[:80],
            "gloss": None,
            "parent_id": max(parent_votes, key=parent_votes.__getitem__) if parent_votes else None,
        }
    supernodes_per_level[leaf_level] = leaf_sn_dict

    # Collapse hyper-edges at each non-leaf level
    coarse_edges_per_level = {}
    for level_idx in range(len(level_budgets)):
        n2sn = node_to_supernode_per_level[level_idx]
        coarse_edges_per_level[level_idx] = collapse_hyperedges(hyperedges, n2sn)

    # Flatten all supernodes
    all_supernodes = []
    for level_idx in sorted(supernodes_per_level):
        all_supernodes.extend(supernodes_per_level[level_idx].values())

    return {
        "snapshot_year": snapshot_year,
        "node_ids": node_ids,
        "level_budgets": level_budgets,
        "n_levels": len(level_budgets) + 1,  # +1 for leaf level
        "supernodes": all_supernodes,
        "node_to_level_supernode": {
            str(k): v for k, v in node_to_supernode_per_level.items()
        },
        "coarse_edges_per_level": {
            str(k): v for k, v in coarse_edges_per_level.items()
        },
    }


def _trivial_hierarchy(nodes_by_id, hyperedges, snapshot_year):
    node_ids = list(nodes_by_id.keys())
    snid = f"sn_y{snapshot_year}_l0_c0001"
    return {
        "snapshot_year": snapshot_year,
        "node_ids": node_ids,
        "level_budgets": LEVEL_BUDGETS,
        "n_levels": 1,
        "supernodes": [{
            "id": snid, "level": 0, "snapshot_year": snapshot_year,
            "members": node_ids, "label": None, "gloss": None, "parent_id": None,
        }],
        "node_to_level_supernode": {"0": {nid: snid for nid in node_ids}},
        "coarse_edges_per_level": {},
    }
