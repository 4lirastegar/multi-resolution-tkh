"""T6 — Stability evaluation.

(a) Perturbation stability: remove 10% of hyper-edges, rebuild hierarchy,
    measure ARI (Adjusted Rand Index) between perturbed and original partition.
    Repeated for each seed in config.SEEDS. Report mean ± 95% CI.

(b) Cross-snapshot stability: ARI between consecutive snapshot hierarchies
    restricted to the nodes that appear in both snapshots.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy import stats as scipy_stats

from src.config import SEEDS
from src.data_loading import slice_snapshot
from src.embeddings import get_clustering_embeddings
from src.hierarchy import build_hierarchy


def _partition_vector(hierarchy: dict, node_ids: list[str], level: int) -> np.ndarray:
    """Return integer label array for nodes at given level."""
    n2sn = hierarchy["node_to_level_supernode"].get(str(level), {})
    unique_sns = sorted(set(n2sn.get(nid, "__missing__") for nid in node_ids))
    sn2int = {s: i for i, s in enumerate(unique_sns)}
    return np.array([sn2int.get(n2sn.get(nid, "__missing__"), 0) for nid in node_ids])


def perturbation_stability(
    nodes_by_id: dict,
    hyperedges: list[dict],
    embeddings: dict[str, np.ndarray],
    snapshot_year: int,
    level: int = 0,
    remove_frac: float = 0.10,
    seeds: list[int] = SEEDS,
) -> dict:
    """ARI between full and 10%-edge-removed hierarchies, over multiple seeds."""
    # Build reference hierarchy on full edge set
    ref_h = build_hierarchy(nodes_by_id, hyperedges, embeddings, snapshot_year)
    node_ids = ref_h["node_ids"]
    ref_labels = _partition_vector(ref_h, node_ids, level)

    aris = []
    n_remove = max(1, int(len(hyperedges) * remove_frac))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(hyperedges), size=len(hyperedges) - n_remove, replace=False)
        perturbed_edges = [hyperedges[i] for i in keep_idx]

        pert_h = build_hierarchy(nodes_by_id, perturbed_edges, embeddings, snapshot_year)
        pert_labels = _partition_vector(pert_h, node_ids, level)
        aris.append(adjusted_rand_score(ref_labels, pert_labels))

    mean_ari = float(np.mean(aris))
    std_ari = float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0
    # 95% CI via t-distribution
    if len(aris) > 1:
        ci = scipy_stats.t.interval(0.95, df=len(aris) - 1,
                                    loc=mean_ari, scale=scipy_stats.sem(aris))
        ci95 = (round(ci[0], 4), round(ci[1], 4))
    else:
        ci95 = (round(mean_ari, 4), round(mean_ari, 4))

    return {
        "level": level,
        "snapshot_year": snapshot_year,
        "remove_frac": remove_frac,
        "seeds": seeds,
        "ari_per_seed": [round(a, 4) for a in aris],
        "ari_mean": round(mean_ari, 4),
        "ari_std": round(std_ari, 4),
        "ari_95ci": ci95,
    }


def cross_snapshot_stability(
    hierarchies: dict[int, dict],
    years: list[int],
    level: int = 0,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> list[dict]:
    """ARI between consecutive snapshots on their shared node set.

    Bootstrap resampling over shared nodes gives 95% CIs that match the
    reporting style of perturbation_stability above.
    """
    results = []
    rng = np.random.default_rng(seed)

    for i in range(1, len(years)):
        y_from, y_to = years[i - 1], years[i]
        h_from = hierarchies.get(y_from)
        h_to = hierarchies.get(y_to)
        if h_from is None or h_to is None:
            continue

        shared_nodes = sorted(
            set(h_from["node_ids"]) & set(h_to["node_ids"])
        )
        if len(shared_nodes) < 2:
            continue

        labels_from = _partition_vector(h_from, shared_nodes, level)
        labels_to = _partition_vector(h_to, shared_nodes, level)
        point_ari = float(adjusted_rand_score(labels_from, labels_to))

        # Bootstrap CI: resample shared nodes with replacement
        n = len(shared_nodes)
        boot_aris: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_aris.append(
                float(adjusted_rand_score(labels_from[idx], labels_to[idx]))
            )
        ci95 = (
            round(float(np.percentile(boot_aris, 2.5)), 4),
            round(float(np.percentile(boot_aris, 97.5)), 4),
        )

        results.append({
            "year_from": y_from,
            "year_to": y_to,
            "level": level,
            "n_shared_nodes": len(shared_nodes),
            "ari": round(point_ari, 4),
            "ari_95ci_bootstrap": ci95,
        })

    return results


def run_stability_evaluation(
    hierarchies: dict[int, dict],
    nodes_by_year: dict[int, dict],
    edges_by_year: dict[int, list],
    embeddings_by_year: dict[int, dict],
    years: list[int],
    levels: list[int] = None,
) -> dict:
    if levels is None:
        levels = [0, 1]

    results = {"perturbation": {}, "cross_snapshot": {}}

    for year in years:
        nodes = nodes_by_year[year]
        edges = edges_by_year[year]
        embs = embeddings_by_year[year]
        results["perturbation"][year] = {}
        for level in levels:
            print(f"  Perturbation stability: year={year}, level={level} …")
            results["perturbation"][year][level] = perturbation_stability(
                nodes, edges, embs, year, level=level
            )

    for level in levels:
        results["cross_snapshot"][level] = cross_snapshot_stability(
            hierarchies, years, level=level
        )

    return results
