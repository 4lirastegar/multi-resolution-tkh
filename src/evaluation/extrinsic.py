"""T6 — Extrinsic utility: coarse-to-fine retrieval on the 17-question benchmark.

Protocol
--------
For each question in questions.csv:
  1. Embed the question text with the CLUSTERING embedding model.
  2. Start at level 0 of the hierarchy (≤2026 snapshot).
  3. Score each level-0 super-node by cosine similarity of the question
     embedding to the centroid of that super-node's member embeddings.
  4. Enter the top-k super-nodes (k=3) and score their level-1 children.
  5. Enter the top-k level-1 nodes and retrieve all leaf members.
  6. Rank the retrieved members by cosine similarity to the question.
  7. Record: hit-rate@5, hit-rate@10, MRR, steps taken.

Flat baseline: rank ALL nodes in the snapshot directly by cosine similarity
to the question, no hierarchy involved.

Score against ground_truth.json:
  expected_methods → the method names that should appear in the top results.
  Hit-rate@k = fraction of expected_methods found in top-k results.
"""

import csv
import json
import os

import numpy as np

from src.config import GROUND_TRUTH_JSON, QUESTIONS_CSV


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _supernode_centroid(
    supernode: dict,
    embeddings: dict[str, np.ndarray],
) -> np.ndarray | None:
    vecs = [embeddings[nid] for nid in supernode["members"] if nid in embeddings]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def drilldown_retrieval(
    hierarchy: dict,
    nodes_by_id: dict,
    embeddings: dict[str, np.ndarray],
    question_embedding: np.ndarray,
    top_k_per_level: int = 3,
) -> list[str]:
    """Return ranked list of node_ids via coarse-to-fine drill-down."""
    sn_by_id = {s["id"]: s for s in hierarchy["supernodes"]}
    n2sn_l0 = hierarchy["node_to_level_supernode"].get("0", {})

    # Level 0 super-nodes
    l0_sns = [s for s in hierarchy["supernodes"] if s["level"] == 0]
    l0_scored = []
    for sn in l0_sns:
        centroid = _supernode_centroid(sn, embeddings)
        if centroid is not None:
            l0_scored.append((sn["id"], _cosine(question_embedding, centroid)))
    l0_scored.sort(key=lambda x: -x[1])
    top_l0 = [snid for snid, _ in l0_scored[:top_k_per_level]]

    # Level 1 children of selected level-0 super-nodes
    l1_sns = [s for s in hierarchy["supernodes"]
              if s["level"] == 1 and s.get("parent_id") in top_l0]
    l1_scored = []
    for sn in l1_sns:
        centroid = _supernode_centroid(sn, embeddings)
        if centroid is not None:
            l1_scored.append((sn["id"], _cosine(question_embedding, centroid)))
    l1_scored.sort(key=lambda x: -x[1])
    top_l1 = [snid for snid, _ in l1_scored[:top_k_per_level * 2]]

    # Collect leaf members from selected level-1 nodes
    candidates = []
    n2sn_l1 = hierarchy["node_to_level_supernode"].get("1", {})
    for nid, snid in n2sn_l1.items():
        if snid in top_l1 and nid in embeddings:
            candidates.append(nid)

    if not candidates:
        candidates = list(embeddings.keys())

    # Rank candidates by similarity to question
    ranked = sorted(
        candidates,
        key=lambda nid: -_cosine(question_embedding, embeddings[nid]),
    )
    return ranked


def flat_baseline(
    embeddings: dict[str, np.ndarray],
    question_embedding: np.ndarray,
) -> list[str]:
    """Rank ALL nodes by cosine similarity — no hierarchy."""
    return sorted(
        embeddings.keys(),
        key=lambda nid: -_cosine(question_embedding, embeddings[nid]),
    )


def _hit_rate(ranked: list[str], expected_names: list[str],
              nodes_by_id: dict, k: int) -> float:
    """Fraction of expected method names found in top-k surface forms."""
    top_texts = [
        nodes_by_id.get(nid, {}).get("surface_form", "").lower()
        for nid in ranked[:k]
    ]
    found = sum(
        1 for name in expected_names
        if any(name.lower() in t for t in top_texts)
    )
    return found / len(expected_names) if expected_names else 0.0


def _mrr(ranked: list[str], expected_names: list[str], nodes_by_id: dict) -> float:
    """Mean Reciprocal Rank for expected methods."""
    if not expected_names:
        return 0.0
    rrs = []
    for name in expected_names:
        for rank, nid in enumerate(ranked, 1):
            sf = nodes_by_id.get(nid, {}).get("surface_form", "").lower()
            if name.lower() in sf:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs))


def run_extrinsic_evaluation(
    hierarchy: dict,
    nodes_by_id: dict,
    embeddings: dict[str, np.ndarray],
    questions_path=QUESTIONS_CSV,
    ground_truth_path=GROUND_TRUTH_JSON,
) -> dict:
    """Run the 17-question benchmark and return per-question + aggregate scores."""
    from sentence_transformers import SentenceTransformer
    from src.config import CLUSTERING_EMBEDDING_MODEL

    model = SentenceTransformer(CLUSTERING_EMBEDDING_MODEL)

    with open(questions_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        questions = list(reader)

    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    per_question = []
    for q in questions:
        qid = q["question_id"]
        qtext = q["question"]
        gt = ground_truth.get(qid, {})
        expected = gt.get("expected_methods", [])
        if not expected:
            continue

        q_emb = model.encode(qtext)

        drill = drilldown_retrieval(hierarchy, nodes_by_id, embeddings, q_emb)
        flat = flat_baseline(embeddings, q_emb)

        per_question.append({
            "question_id": qid,
            "expected_methods": expected,
            "drilldown": {
                "hit_rate_5":  round(_hit_rate(drill, expected, nodes_by_id, 5), 3),
                "hit_rate_10": round(_hit_rate(drill, expected, nodes_by_id, 10), 3),
                "hit_rate_20": round(_hit_rate(drill, expected, nodes_by_id, 20), 3),
                "mrr": round(_mrr(drill, expected, nodes_by_id), 3),
            },
            "flat_baseline": {
                "hit_rate_5":  round(_hit_rate(flat, expected, nodes_by_id, 5), 3),
                "hit_rate_10": round(_hit_rate(flat, expected, nodes_by_id, 10), 3),
                "hit_rate_20": round(_hit_rate(flat, expected, nodes_by_id, 20), 3),
                "mrr": round(_mrr(flat, expected, nodes_by_id), 3),
            },
        })

    def _agg(key, metric):
        return round(np.mean([q[key][metric] for q in per_question]), 3)

    aggregate = {
        "n_questions": len(per_question),
        "drilldown": {
            "hit_rate_5":  _agg("drilldown", "hit_rate_5"),
            "hit_rate_10": _agg("drilldown", "hit_rate_10"),
            "hit_rate_20": _agg("drilldown", "hit_rate_20"),
            "mrr":         _agg("drilldown", "mrr"),
        },
        "flat_baseline": {
            "hit_rate_5":  _agg("flat_baseline", "hit_rate_5"),
            "hit_rate_10": _agg("flat_baseline", "hit_rate_10"),
            "hit_rate_20": _agg("flat_baseline", "hit_rate_20"),
            "mrr":         _agg("flat_baseline", "mrr"),
        },
    }

    return {"per_question": per_question, "aggregate": aggregate}
