"""T6 — Run all evaluations and write outputs/metrics.json."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR, SNAPSHOT_DIR, SNAPSHOT_YEARS
from src.data_loading import load_tkh, slice_snapshot
from src.embeddings import get_clustering_embeddings, get_evaluation_embeddings
from src.evaluation.coherence import run_coherence_evaluation
from src.evaluation.stability import run_stability_evaluation
from src.evaluation.faithfulness import overclaim_rate
from src.evaluation.extrinsic import run_extrinsic_evaluation


def load_hierarchies():
    hierarchies = {}
    for year in SNAPSHOT_YEARS:
        path = SNAPSHOT_DIR / f"hierarchy_{year}.json"
        if path.exists():
            with open(path) as f:
                hierarchies[year] = json.load(f)
        else:
            print(f"  WARNING: {path} not found — run 02_build_hierarchies.py first")
    return hierarchies


if __name__ == "__main__":
    print("Loading data …")
    meta, nodes_by_id, hyperedges = load_tkh()

    print("Loading hierarchies …")
    hierarchies = load_hierarchies()
    if not hierarchies:
        raise SystemExit("No hierarchy files found. Run scripts/02_build_hierarchies.py first.")

    # Build per-snapshot data structures
    nodes_by_year, edges_by_year = {}, {}
    clustering_emb_by_year, eval_emb_by_year = {}, {}

    for year in SNAPSHOT_YEARS:
        print(f"  Preparing snapshot {year} …")
        nodes, edges = slice_snapshot(nodes_by_id, hyperedges, year)
        nodes_by_year[year] = nodes
        edges_by_year[year] = edges
        clustering_emb_by_year[year] = get_clustering_embeddings(nodes)
        print(f"    Computing evaluation embeddings (independent, mpnet) …")
        eval_emb_by_year[year] = get_evaluation_embeddings(nodes)

    metrics = {}

    # ── Coherence ─────────────────────────────────────────────────────────
    print("\n=== Coherence evaluation ===")
    metrics["coherence"] = run_coherence_evaluation(
        hierarchies, eval_emb_by_year, levels=[0, 1]
    )
    for year, lvls in metrics["coherence"].items():
        for lvl, scores in lvls.items():
            print(f"  year={year} level={lvl}: separation={scores.get('separation')} "
                  f"z={scores.get('z_above_null')}")

    # ── Stability ─────────────────────────────────────────────────────────
    print("\n=== Stability evaluation ===")
    metrics["stability"] = run_stability_evaluation(
        hierarchies,
        nodes_by_year,
        edges_by_year,
        clustering_emb_by_year,
        SNAPSHOT_YEARS,
        levels=[0, 1],
    )

    # ── Label faithfulness ────────────────────────────────────────────────
    print("\n=== Label faithfulness ===")
    metrics["faithfulness"] = {}
    for year, h in hierarchies.items():
        print(f"  year={year} …")
        result = overclaim_rate(h, nodes_by_id, snapshot_year=year)
        metrics["faithfulness"][year] = result
        if "overclaim_rate" in result:
            print(f"    overclaim_rate={result['overclaim_rate']} "
                  f"supported={result['supported_rate']} n={result['total_evaluated']}")

    # ── Extrinsic utility ─────────────────────────────────────────────────
    print("\n=== Extrinsic utility (17 questions) ===")
    h2026 = hierarchies.get(2026)
    if h2026:
        extrinsic = run_extrinsic_evaluation(
            h2026, nodes_by_id, clustering_emb_by_year[2026]
        )
        metrics["extrinsic"] = extrinsic
        agg = extrinsic["aggregate"]
        print(f"  Drill-down   HR@10={agg['drilldown']['hit_rate_10']}  MRR={agg['drilldown']['mrr']}")
        print(f"  Flat baseline HR@10={agg['flat_baseline']['hit_rate_10']}  MRR={agg['flat_baseline']['mrr']}")
    else:
        print("  WARNING: no 2026 hierarchy found")

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {metrics_path}")
    print("Done.")
