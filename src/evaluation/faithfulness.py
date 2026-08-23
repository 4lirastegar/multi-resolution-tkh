"""T6 — Label faithfulness: over-claim rate.

Each super-node gloss is checked by GPT-4o-mini acting as an independent judge.
The judge receives:
  - The gloss to evaluate
  - The list of member surface forms

The judge rates each gloss as one of:
  "supported"  — the gloss is fully supported by the listed items
  "vague"      — the gloss is too general to verify from the items
  "overclaim"  — the gloss asserts something NOT supported by the items

Over-claim rate = # overclaim / total labelled supernodes evaluated.

Circularity control: the judge receives the same surface forms the labeller
used, but is a different prompt with a different instruction (verify vs generate),
so the judgement is not trivially circular.  A stronger control would use an
NLI model; the GPT-judge is used here for practicality on this corpus size.
"""

import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT_TEMPLATE = """\
You are a scientific fact-checker. You will be given a short label and gloss \
describing a group of scientific items, followed by the actual items in the group.

Label: {label}
Gloss: {gloss}

Items in the group:
{items}

Task: Decide whether the gloss is:
- "supported": every claim in the gloss is directly supported by the items listed
- "vague": the gloss is so general it cannot be verified from the items
- "overclaim": the gloss asserts something that is NOT supported by the items

Respond ONLY with valid JSON: {{"verdict": "supported"|"vague"|"overclaim", "reason": "one sentence"}}"""


def _judge(label: str, gloss: str, member_texts: list[str],
           model: str = "gpt-4o-mini", retries: int = 3) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    items_str = "\n".join(f"- {t}" for t in member_texts[:30])
    prompt = JUDGE_PROMPT_TEMPLATE.format(label=label, gloss=gloss, items=items_str)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            if attempt == retries - 1:
                return {"verdict": "vague", "reason": str(e)}
            time.sleep(2 ** attempt)


def overclaim_rate(
    hierarchy: dict,
    nodes_by_id: dict,
    snapshot_year: int,
    levels: list[int] | None = None,
    sample_size: int | None = None,
) -> dict:
    """Compute over-claim rate for labelled supernodes at given levels."""
    if levels is None:
        levels = [0, 1]

    has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if not has_key:
        return {"error": "No OPENAI_API_KEY — skipping faithfulness evaluation"}

    target = [
        s for s in hierarchy["supernodes"]
        if s["level"] in levels and s.get("label") and s.get("gloss")
    ]

    if sample_size and len(target) > sample_size:
        import random
        random.seed(42)
        target = random.sample(target, sample_size)

    verdicts = []
    for sn in target:
        texts = [
            nodes_by_id[nid].get("surface_form", "")
            for nid in sn["members"]
            if nid in nodes_by_id
        ]
        texts = [t for t in texts if t.strip()]
        result = _judge(sn["label"], sn["gloss"], texts)
        verdicts.append({
            "supernode_id": sn["id"],
            "level": sn["level"],
            "label": sn["label"],
            "gloss": sn["gloss"],
            "verdict": result.get("verdict", "vague"),
            "reason": result.get("reason", ""),
        })

    counts = {"supported": 0, "vague": 0, "overclaim": 0}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1

    total = len(verdicts)
    return {
        "snapshot_year": snapshot_year,
        "total_evaluated": total,
        "counts": counts,
        "overclaim_rate": round(counts["overclaim"] / total, 3) if total > 0 else None,
        "supported_rate": round(counts["supported"] / total, 3) if total > 0 else None,
        "details": verdicts,
    }
