# Multi-Resolution Semantic Abstraction over an Evolving Knowledge Hypergraph

PhD assessment submission for Constructor Knowledge Labs.

Builds a temporally stable, multi-resolution hierarchy over a Temporal Knowledge
Hypergraph (TKH) of materials-science / ML-interatomic-potentials literature —
52 papers, 5,798 nodes, 1,429 hyper-edges — at four time snapshots (≤2020, 2022,
2024, 2026).

---

## Repository layout

```
src/
  config.py             # paths, snapshot years, level budgets, seeds, model names
  data_loading.py       # T1: snapshot slicing by edge year, quality checks
  embeddings.py         # sentence embeddings — two models (MiniLM to cluster, mpnet to evaluate)
  hyperedge_collapse.py # T4: 3-case hyper-edge coarsening rule
  hierarchy.py          # T2: combined-affinity Ward clustering, dendrogram cuts
  temporal.py           # T3: Jaccard supernode matching + event log
  labeling.py           # T5: GPT-4o-mini labels with temporal-honesty filter
  evaluation/
    coherence.py        # T6: intra/inter cosine vs null model (independent mpnet signal)
    stability.py        # T6: perturbation ARI (5 seeds, 10% edge removal) + cross-snapshot ARI
    faithfulness.py     # T6: GPT blind judge → over-claim rate
    extrinsic.py        # T6: drill-down retrieval vs flat baseline on 17-question benchmark

scripts/
  01_describe_data.py   # run T1
  02_build_hierarchies.py # run T2–T4 for all snapshots
  03_label.py           # run T5 (can re-run without rebuilding hierarchies)
  04_evaluate.py        # run T6 → outputs/metrics.json
  run_all.py            # run everything end to end

outputs/
  snapshots/            # hierarchy_2020.json … hierarchy_2026.json
  temporal_events.json  # 327 lifecycle events across all transitions
  metrics.json          # all T6 numbers

report/
  report.pdf            # 5-page write-up
  report.tex            # LaTeX source

data/                   # TKH export — NOT included (provided by CKL, see data/README.md)
```

---

## Reproduction

> **Note:** Place the TKH data files in `data/` before running (see `data/README.md`).
> An OpenAI API key is required for labelling (T5) and faithfulness evaluation (T6).
> Copy `.env.example` to `.env` and add your key.

```bash
python3 -m venv .venv
source .venv/bin/activate       # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline (T1 → T2 → T3 → T4 → T5 → T6)
python scripts/run_all.py
```

Run individual steps:

```bash
python scripts/01_describe_data.py      # snapshot statistics
python scripts/02_build_hierarchies.py  # hierarchies + event log
python scripts/03_label.py              # GPT labels (requires API key)
python scripts/04_evaluate.py           # metrics.json
```

All randomness is seeded via `src/config.py` (`SEEDS = [0, 1, 2, 3, 4]`).

---

## Key outputs

| File | Description |
|---|---|
| `outputs/snapshots/hierarchy_<year>.json` | Super-node id, level, parent id, member ids, label, gloss |
| `outputs/temporal_events.json` | Birth / growth / shrink / death events per transition |
| `outputs/metrics.json` | Coherence z-scores, perturbation ARI with 95% CIs, over-claim rate, HR@10/MRR |
| `report/report.pdf` | 5-page write-up covering formal statement, method, evaluation, limitations |

---

## Method in one paragraph

Nodes are embedded with `all-MiniLM-L6-v2`. A combined affinity matrix
(0.6 × cosine similarity + 0.4 × arity-weighted co-occurrence) is built per
snapshot. Ward-linkage agglomerative clustering produces a single dendrogram
that is cut at three heights to yield 12 / 60 / 300 / leaf super-nodes — a
laminar hierarchy by construction. Hyper-edges are coarsened natively at every
level (never projected to pairs). Temporal stability is maintained by reusing
cached embeddings for existing nodes. Coherence is measured with an independent
model (mpnet) to avoid the circularity hazard.
