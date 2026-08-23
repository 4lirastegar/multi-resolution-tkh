# Multi-Resolution Semantic Abstraction over an Evolving Knowledge Hypergraph

Solution for the Constructor Knowledge Labs PhD assessment task: a multi-resolution,
temporally stable hierarchy of super-nodes over a Temporal Knowledge Hypergraph
(52 materials-science / ML-interatomic-potentials papers; 5,798 nodes, 1,429 hyper-edges).

## Repository layout

```
data/                  # TKH export as shipped (tkh_collection10.json, questions, ground truth)
src/                   # library code
  config.py            # snapshot years, level budgets, seeds, model names
  data_loading.py      # T1: load JSON, slice temporal snapshots
  embeddings.py        # node text embeddings (computed once, cached)
  hyperedge_collapse.py# T4: coarsening rule for hyper-edges (used by hierarchy.py)
  hierarchy.py         # T2: multi-resolution laminar hierarchy construction
  temporal.py          # T3: cross-snapshot coupling + birth/growth/merge/split/death log
  labeling.py          # T5: LLM labels + glosses (temporally honest)
  evaluation/          # T6
    coherence.py       #   coherence with an independent signal + null model
    stability.py       #   perturbation & cross-snapshot stability (ARI, CIs)
    faithfulness.py    #   label over-claim rate
    extrinsic.py       #   17-question drill-down retrieval vs flat baseline
scripts/               # runnable pipeline steps (01..04) + run_all.py
outputs/               # generated artifacts: hierarchies, event log, metrics, figures
report/                # report.md (3–5 pages)
```

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all.py        # regenerates every artifact in outputs/
```

Individual steps:

```bash
python scripts/01_describe_data.py      # T1: per-snapshot statistics
python scripts/02_build_hierarchies.py  # T2–T4: hierarchies per snapshot + event log
python scripts/03_label.py              # T5: labels + glosses
python scripts/04_evaluate.py           # T6: metrics.json
```

All randomness is controlled by the seeds in `src/config.py`.

## Outputs

- `outputs/snapshots/hierarchy_<year>.json` — per snapshot: super-node id, level,
  parent id, member ids, label, gloss
- `outputs/temporal_events.json` — birth/growth/merge/split/death log
- `outputs/metrics.json` — coherence vs null model, stability with CIs,
  over-claim rate, extrinsic-task scores
