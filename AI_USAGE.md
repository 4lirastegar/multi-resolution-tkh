# AI Usage Log

Per §9 of the task specification. This document records every place an AI tool was
used, what I asked it to do, what I accepted or changed, and how I verified the result.

---

## My role vs. the AI's role

I designed the overall approach myself: the decision to use a combined
semantic+structural affinity (not pure embedding), the choice to use two
*different* embedding models to avoid the circularity hazard in coherence
evaluation, the hyper-edge collapse rule, and the Jaccard-based temporal matching.
These were deliberate design choices I made after reading the task carefully —
the AI helped implement them, not invent them.

Where I caught errors and corrected them is noted explicitly below.

---

## Tool usage table

| Module | Tool | What I asked for | What I changed / caught | How I verified |
|---|---|---|---|---|
| Repo scaffolding | Cursor agent | Create directory structure, stub files, README | Minor path corrections | Manual inspection of all stubs |
| `src/config.py` | Cursor agent | Central config with paths, years, seeds, model names | Added `ALPHA=0.6` and `JACCARD_MATCH_THRESHOLD=0.3` myself after deciding the values | Read the file and confirmed all constants |
| `src/data_loading.py` | Cursor agent | Snapshot slicing by edge year, quality checks | Verified the slicing rule is edge-year based (not node year) — this distinction was my call | Confirmed output counts match task spec exactly: 405/589/1063/1429 |
| `src/embeddings.py` | Cursor agent | Two-model embedding with disk cache | I specified the anti-circularity requirement (two different model families) — the agent implemented it | Tested cache hit on second run; confirmed mpnet is never used during clustering |
| `src/hyperedge_collapse.py` | Cursor agent | Implement the 3-case coarsening rule | I defined the 3 cases myself (k=1, 1<k<n, k≥3) before asking for implementation | Manually traced 5 sample edges through the rule; checked Case C never splits to pairs |
| `src/hierarchy.py` | Cursor agent | Combined-affinity Ward clustering + dendrogram cuts | **Caught and fixed a bug**: `max(parent_votes, key=parent_votes.__get__)` → `__getitem__`. Also confirmed 0.6/0.4 split was correctly applied | Verified `n_per_level` output matches budgets 12/60/300 at every snapshot |
| `src/temporal.py` | Cursor agent | Jaccard matching + event classification | Reviewed the greedy matching logic carefully; confirmed threshold 0.30 is applied correctly | Counted 327 events across 3 transitions; spot-checked 4 individual events |
| `src/labeling.py` | Cursor agent | GPT-4o-mini labelling with temporal-honesty filter | Checked that the filter correctly excludes nodes with `first_seen_year > t` from labeller input. Tested the PCA/1901 edge case explicitly | Confirmed PCA absent from 2020 labeller input |
| `src/evaluation/coherence.py` | Cursor agent | Intra/inter cosine with mpnet + null model | Reviewed the null model shuffle — confirmed it shuffles node→cluster assignment while keeping cluster sizes fixed | Z-scores of 60–243 are plausible and consistent across snapshots |
| `src/evaluation/stability.py` | Cursor agent | ARI perturbation (5 seeds, 10% removal) + cross-snapshot ARI | Confirmed CI calculation uses correct t-distribution multiplier for n=5 | Wide CIs at L0 (k=12) are expected and correctly reported |
| `src/evaluation/faithfulness.py` | Cursor agent | GPT-4o-mini blind judge for over-claim rate | Verified the judge prompt does NOT include the labeller's output — just the member texts and the label to evaluate | Manually checked 5 judge verdicts against the label text |
| `src/evaluation/extrinsic.py` | Cursor agent | Drill-down retrieval vs flat baseline | Reviewed the drill-down logic step by step; confirmed it navigates L0→L1→members, not L0 directly to leaves | Cross-checked HR@10 arithmetic on a subset of questions |
| `report/report.tex` | Cursor agent | LaTeX draft with all sections | **Rewrote the framing of several sections** myself — the formal problem statement structure, the α trade-off justification, and the evaluation circularity paragraph were all revised to reflect my actual reasoning | Read every sentence; compiled with tectonic |
| Figure generation | Cursor agent | matplotlib figures for T1 and dendrogram | Checked axis labels and caption accuracy | Visual inspection |

---

## GPT-4o-mini prompts (labelling, run via `src/labeling.py`)

```
You are a scientific knowledge organiser. Below are the text labels of items
grouped together in a cluster from a knowledge graph of materials science and
machine learning literature. The snapshot year is {year}, so only items first
seen by that year are included.

Give this cluster:
1. A short name (5 words or fewer)
2. A one-sentence description of what connects these items

Items:
{member_texts}

Respond in JSON: {"label": "...", "gloss": "..."}
```

**What I checked:** I read all 12 Level-0 labels for every snapshot. Several
came back as ``Machine Learning in Materials Science'' for large heterogeneous
clusters — I kept these rather than asking GPT to invent a false specificity.
That is an honest reflection of the narrow domain.

---

## GPT-4o-mini prompts (faithfulness judge, run via `src/evaluation/faithfulness.py`)

```
You are evaluating whether a cluster label is faithful to its members.

Label: "{label}"
Gloss: "{gloss}"
Member texts (sample): {member_sample}

Rate this label as one of:
- "supported": the label and gloss are accurate and specific to these members
- "vague": the label is technically correct but too generic to be useful
- "overclaim": the label asserts something not supported by the member texts

Respond with just the single word.
```

**Key design choice I made:** The judge receives only the member texts and the
label — it does NOT see the labeller's reasoning or any prior output. This keeps
the evaluation independent. I verified this by inspecting the prompt construction
in `faithfulness.py` before running.

---

## Things I verified manually (not by AI)

- The snapshot slicing rule (edge year, not node year) — I thought this through
  myself after reading the P6 temporal honesty requirement
- The anti-circularity setup (two different embedding families) — I identified
  this as the key methodological hazard before any code was written
- The hyper-edge collapse case definitions — I wrote out the three cases on paper
  before asking the agent to implement them
- The 0.6/0.4 semantic/structural split — I reasoned about this (pure structural
  conflates authors with papers; pure semantic ignores relational context) and
  chose 0.6 as a prior. I did not tune it on outputs.
- The Jaccard threshold 0.30 — chosen to be generous enough to track real
  identity chains even after the 90% growth burst in 2022→2024
- All numbers in the report were cross-checked against `outputs/metrics.json`
  before writing

---

## What I would do differently with more time

Run a human blind rating of 20 glosses to reduce reliance on the GPT judge.
Tune α via cross-validated retrieval score rather than using a prior.
Verify the extraction quality of `surface_form` texts in the raw TKH data.
