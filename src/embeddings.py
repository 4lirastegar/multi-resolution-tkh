"""Node text embeddings — computed once and cached on disk.

Two models are intentionally separated:
- CLUSTERING_EMBEDDING_MODEL  : drives the hierarchy (semantic signal for P3)
- EVALUATION_EMBEDDING_MODEL  : used ONLY in evaluation/coherence.py to avoid
  the circularity hazard (T6): measuring cluster quality with the same ruler
  that built the clusters would guarantee high scores by construction.
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from src.config import (
    CACHE_DIR,
    CLUSTERING_EMBEDDING_MODEL,
    EVALUATION_EMBEDDING_MODEL,
)


def _cache_path(model_name: str, node_ids: list[str]) -> Path:
    key = hashlib.md5((model_name + "".join(sorted(node_ids))).encode()).hexdigest()[:12]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"emb_{key}.pkl"


def embed_nodes(
    nodes_by_id: dict,
    model_name: str = CLUSTERING_EMBEDDING_MODEL,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Return {node_id: embedding_vector} for every node in nodes_by_id.

    Results are cached to disk; subsequent calls with the same node set and
    model are instant.
    """
    node_ids = sorted(nodes_by_id.keys())
    cache_file = _cache_path(model_name, node_ids)

    if cache_file.exists() and not force:
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    model = SentenceTransformer(model_name)
    texts = [nodes_by_id[nid].get("surface_form", "") for nid in node_ids]

    batch = 256
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc=f"Embedding [{model_name}]"):
        vecs.append(model.encode(texts[i : i + batch], show_progress_bar=False))
    matrix = np.vstack(vecs)

    result = {nid: matrix[i] for i, nid in enumerate(node_ids)}
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


def get_clustering_embeddings(nodes_by_id: dict) -> dict[str, np.ndarray]:
    return embed_nodes(nodes_by_id, CLUSTERING_EMBEDDING_MODEL)


def get_evaluation_embeddings(nodes_by_id: dict) -> dict[str, np.ndarray]:
    """Independent embeddings for evaluation only — never used in clustering."""
    return embed_nodes(nodes_by_id, EVALUATION_EMBEDDING_MODEL)
