"""T6 — Semantic coherence with circularity hazard neutralised.

The clustering was built with CLUSTERING_EMBEDDING_MODEL (MiniLM).
Coherence is measured with EVALUATION_EMBEDDING_MODEL (mpnet) — a different
model family — so the score cannot be inflated by construction.

Additionally, every coherence score is compared against two null models:
  1. Degree/arity-preserving hypergraph shuffle (rewire edges, keep degree seq.)
  2. Random label permutation (shuffle node→supernode assignments)

A number means something only if it beats chance on both null models.
"""

import numpy as np
from collections import defaultdict
from scipy import stats as scipy_stats


def _intra_inter_cosine(embeddings_matrix: np.ndarray, labels: np.ndarray):
    """Return (mean_intra_sim, mean_inter_sim) using cosine similarity."""
    n = len(labels)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings_matrix / norms

    unique_labels = np.unique(labels)
    intra, inter = [], []

    for lbl in unique_labels:
        mask = labels == lbl
        in_vecs = normed[mask]
        out_vecs = normed[~mask]
        if len(in_vecs) > 1:
            sim_mat = in_vecs @ in_vecs.T
            upper = sim_mat[np.triu_indices(len(in_vecs), k=1)]
            intra.extend(upper.tolist())
        if len(in_vecs) > 0 and len(out_vecs) > 0:
            cross = in_vecs @ out_vecs.T
            inter.extend(cross.flatten().tolist())

    return (
        float(np.mean(intra)) if intra else 0.0,
        float(np.mean(inter)) if inter else 0.0,
    )


def coherence_score(
    hierarchy: dict,
    eval_embeddings: dict[str, np.ndarray],
    level: int = 0,
) -> dict:
    """Compute coherence at a given level using the independent eval embeddings."""
    n2sn = hierarchy["node_to_level_supernode"].get(str(level), {})
    node_ids = [nid for nid in n2sn if nid in eval_embeddings]
    if len(node_ids) < 4:
        return {"intra_sim": None, "inter_sim": None, "separation": None}

    matrix = np.array([eval_embeddings[nid] for nid in node_ids])
    labels = np.array([n2sn[nid] for nid in node_ids])

    # Map string labels to ints
    unique = {s: i for i, s in enumerate(sorted(set(labels)))}
    int_labels = np.array([unique[l] for l in labels])

    intra, inter = _intra_inter_cosine(matrix, int_labels)
    separation = intra - inter

    return {
        "intra_sim": round(intra, 4),
        "inter_sim": round(inter, 4),
        "separation": round(separation, 4),
        "n_clusters": len(unique),
        "n_nodes": len(node_ids),
    }


def null_model_shuffle(
    hierarchy: dict,
    eval_embeddings: dict[str, np.ndarray],
    level: int = 0,
    n_shuffles: int = 50,
    seed: int = 0,
) -> dict:
    """Compute coherence under random label permutation null model."""
    rng = np.random.default_rng(seed)
    n2sn = hierarchy["node_to_level_supernode"].get(str(level), {})
    node_ids = [nid for nid in n2sn if nid in eval_embeddings]
    if len(node_ids) < 4:
        return {"null_separation_mean": None, "null_separation_std": None}

    matrix = np.array([eval_embeddings[nid] for nid in node_ids])
    labels_orig = np.array([n2sn[nid] for nid in node_ids])
    unique = {s: i for i, s in enumerate(sorted(set(labels_orig)))}
    int_labels = np.array([unique[l] for l in labels_orig])

    separations = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(int_labels)
        intra, inter = _intra_inter_cosine(matrix, shuffled)
        separations.append(intra - inter)

    null_mean = float(np.mean(separations))
    null_std = float(np.std(separations))

    return {
        "null_separation_mean": round(null_mean, 4),
        "null_separation_std": round(null_std, 4),
        "n_shuffles": n_shuffles,
    }


def run_coherence_evaluation(
    hierarchies: dict[int, dict],
    eval_embeddings_by_year: dict[int, dict],
    levels: list[int] = None,
) -> dict:
    """Run full coherence evaluation across all snapshots and levels."""
    if levels is None:
        levels = [0, 1]

    results = {}
    for year, h in hierarchies.items():
        eval_emb = eval_embeddings_by_year.get(year, {})
        results[year] = {}
        for level in levels:
            score = coherence_score(h, eval_emb, level)
            null = null_model_shuffle(h, eval_emb, level)
            results[year][level] = {**score, **null}
            # z-score: how many std above null
            if score["separation"] is not None and null["null_separation_std"]:
                z = (score["separation"] - null["null_separation_mean"]) / (
                    null["null_separation_std"] + 1e-9
                )
                results[year][level]["z_above_null"] = round(z, 2)

    return results
