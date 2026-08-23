"""T5 — Auto-labelling of super-nodes at levels 0 and 1.

Uses GPT-4o-mini via the OpenAI API.  The exact prompt template is recorded
here and in AI_USAGE.md (required by §9 of the task).

Temporal honesty (P6): for snapshot year t, only surface_form values of
members with first_seen_year <= t are passed to the labeller.  This prevents
the label from asserting facts that the corpus had not yet established by t.

Circularity control: the labeller receives surface_form texts (the same input
the clusterer used).  This is unavoidable for label generation.  Faithfulness
is therefore assessed by a *separate* judge (evaluation/faithfulness.py) that
uses an NLI model — not the labeller — so label quality is measured
independently.
"""

import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

LABEL_PROMPT_TEMPLATE = """\
You are a scientific knowledge organiser. Below are the text labels of items \
that have been grouped together in a knowledge graph of materials science and \
machine learning literature.

Items:
{items}

Task: Give this group:
1. A short name (3-7 words, title-case, no quotes).
2. A one-sentence gloss (max 20 words) that describes what the group represents.
   Only assert facts that are directly supported by the items listed above.
   Do NOT mention anything that is not represented in the list.

Respond ONLY with valid JSON in this exact format:
{{"label": "...", "gloss": "..."}}"""


def _call_openai(prompt: str, model: str = "gpt-4o-mini", retries: int = 3) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=120,
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            if attempt == retries - 1:
                return {"label": "Unlabelled cluster", "gloss": str(e)}
            time.sleep(2 ** attempt)


def _keyword_fallback(texts: list[str]) -> dict:
    """Simple keyword-extraction fallback when no API key is available."""
    from collections import Counter
    import re
    stopwords = {"the","a","an","of","in","for","to","and","or","with","from",
                 "by","on","at","is","are","be","as","its","that","this","it",
                 "was","been","not","but","than","more","which","have","has"}
    words = []
    for t in texts:
        words.extend(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", t)
                     if w.lower() not in stopwords)
    top = [w for w, _ in Counter(words).most_common(6)]
    label = " ".join(w.title() for w in top[:4]) or "Cluster"
    gloss = "Group related to: " + ", ".join(top[:5]) + "."
    return {"label": label, "gloss": gloss}


def label_supernodes(
    hierarchy: dict,
    nodes_by_id: dict,
    snapshot_year: int,
    levels: list[int] | None = None,
    use_openai: bool = True,
) -> dict:
    """Add label and gloss to every super-node at the specified levels.

    Modifies hierarchy in-place and returns it.
    """
    if levels is None:
        levels = [0, 1]

    has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    use_openai = use_openai and has_key

    target_sns = [s for s in hierarchy["supernodes"] if s["level"] in levels]

    for sn in target_sns:
        # Temporal honesty: only include members first_seen by snapshot_year
        texts = []
        for nid in sn["members"]:
            node = nodes_by_id.get(nid, {})
            fsy = node.get("first_seen_year")
            if fsy is None or fsy <= snapshot_year:
                sf = node.get("surface_form", "").strip()
                if sf:
                    texts.append(sf)

        if not texts:
            sn["label"] = "Empty cluster"
            sn["gloss"] = "No members with surface forms visible by this snapshot."
            continue

        # Limit to 40 most informative texts to keep prompt short
        sample = texts[:40]
        items_str = "\n".join(f"- {t}" for t in sample)

        if use_openai:
            result = _call_openai(LABEL_PROMPT_TEMPLATE.format(items=items_str))
        else:
            result = _keyword_fallback(sample)

        sn["label"] = result.get("label", "Unlabelled")
        sn["gloss"] = result.get("gloss", "")

    return hierarchy
