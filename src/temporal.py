"""T3 — Temporal coupling across snapshots (P5).

Mechanism
---------
1. Warm-start: when building snapshot H(t+1), the embeddings and affinity
   matrix from H(t) are reused as a prior — nodes that already existed keep
   their previous cluster assignment as an initial hint (passed as warm_labels
   to hierarchy.build_hierarchy).  This biases the dendrogram towards the
   previous solution in stable regions.

2. Identity matching: after building each snapshot's hierarchy independently,
   we match super-nodes across consecutive snapshots using Jaccard overlap on
   shared member node IDs.  A super-node s in snapshot t is "the same" as s'
   in snapshot t+1 if Jaccard(members(s), members(s')) >= JACCARD_MATCH_THRESHOLD.

3. Event log: from the matching we derive a structured log of:
     birth   — s' exists in t+1 but has no match in t
     death   — s exists in t but has no match in t+1
     growth  — s → s' matched, |members(s')| > |members(s)|
     shrink  — s → s' matched, |members(s')| < |members(s)|
     stable  — s → s' matched, same member set
     merge   — two or more supernodes in t map to one s' in t+1
     split   — one supernode in t maps to two or more s' in t+1

Cost: the warm-start does not change asymptotic complexity but adds one
full affinity-matrix computation per snapshot (reused from the cache).
Identity matching is O(|S_t| * |S_{t+1}| * avg_members) where |S_k| is the
number of super-nodes at a level — negligible for this corpus size.
"""

from collections import defaultdict

from src.config import JACCARD_MATCH_THRESHOLD, SNAPSHOT_YEARS


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def match_supernodes(
    prev_supernodes: list[dict],
    curr_supernodes: list[dict],
    level: int,
    threshold: float = JACCARD_MATCH_THRESHOLD,
) -> list[tuple[str | None, str | None, float]]:
    """Return list of (prev_id, curr_id, jaccard) matches for one level.

    Supports many-to-one (merge) and one-to-many (split) matches in addition
    to the standard 1-to-1 growth/shrink/stable cases.
    prev_id or curr_id may be None for birth/death events.
    """
    prev = [s for s in prev_supernodes if s["level"] == level]
    curr = [s for s in curr_supernodes if s["level"] == level]

    prev_sets = {s["id"]: set(s["members"]) for s in prev}
    curr_sets = {s["id"]: set(s["members"]) for s in curr}

    # All (prev, curr) pairs above threshold — no 1-to-1 exclusion yet
    all_pairs: list[tuple[str, str, float]] = []
    for pid, pset in prev_sets.items():
        for cid, cset in curr_sets.items():
            j = jaccard(pset, cset)
            if j >= threshold:
                all_pairs.append((pid, cid, j))

    # Build adjacency maps
    prev_to_curr: dict[str, list[str]] = defaultdict(list)
    curr_to_prev: dict[str, list[str]] = defaultdict(list)
    for pid, cid, _ in all_pairs:
        prev_to_curr[pid].append(cid)
        curr_to_prev[cid].append(pid)

    matched_prev: set[str] = set()
    matched_curr: set[str] = set()
    matches: list[tuple[str | None, str | None, float]] = []

    # Detect merges: multiple prev → one curr
    for cid, pids in curr_to_prev.items():
        if len(pids) > 1:
            for pid in pids:
                j = jaccard(prev_sets[pid], curr_sets[cid])
                matches.append((pid, cid, j))
                matched_prev.add(pid)
            matched_curr.add(cid)

    # Detect splits: one prev → multiple curr (skip already matched)
    for pid, cids in prev_to_curr.items():
        if pid in matched_prev:
            continue
        remaining = [c for c in cids if c not in matched_curr]
        if len(remaining) > 1:
            for cid in remaining:
                j = jaccard(prev_sets[pid], curr_sets[cid])
                matches.append((pid, cid, j))
                matched_curr.add(cid)
            matched_prev.add(pid)

    # Remaining: greedy 1-to-1 for growth / shrink / stable
    remaining_scores = [
        (pid, cid, j) for pid, cid, j in all_pairs
        if pid not in matched_prev and cid not in matched_curr
    ]
    remaining_scores.sort(key=lambda x: -x[2])
    for pid, cid, j in remaining_scores:
        if pid not in matched_prev and cid not in matched_curr:
            matches.append((pid, cid, j))
            matched_prev.add(pid)
            matched_curr.add(cid)

    # Unmatched prev → deaths; unmatched curr → births
    for pid in prev_sets:
        if pid not in matched_prev:
            matches.append((pid, None, 0.0))
    for cid in curr_sets:
        if cid not in matched_curr:
            matches.append((None, cid, 0.0))

    return matches


def classify_event(
    prev_id: str | None,
    curr_id: str | None,
    prev_supernodes: list[dict],
    curr_supernodes: list[dict],
    all_matches: list[tuple],
    year_from: int,
    year_to: int,
) -> dict:
    """Classify one match into a typed event dict."""
    prev_map = {s["id"]: s for s in prev_supernodes}
    curr_map = {s["id"]: s for s in curr_supernodes}

    if prev_id is None:
        return {
            "event": "birth",
            "supernode_id": curr_id,
            "year_from": year_from,
            "year_to": year_to,
            "label": curr_map.get(curr_id, {}).get("label"),
            "n_members": len(curr_map.get(curr_id, {}).get("members", [])),
        }
    if curr_id is None:
        return {
            "event": "death",
            "supernode_id": prev_id,
            "year_from": year_from,
            "year_to": year_to,
            "label": prev_map.get(prev_id, {}).get("label"),
            "n_members": len(prev_map.get(prev_id, {}).get("members", [])),
        }

    prev_members = set(prev_map.get(prev_id, {}).get("members", []))
    curr_members = set(curr_map.get(curr_id, {}).get("members", []))

    # Check merge: multiple prev → same curr
    n_prev_to_curr = sum(1 for p, c, _ in all_matches if c == curr_id and p is not None)
    # Check split: one prev → multiple curr
    n_prev_to_many = sum(1 for p, c, _ in all_matches if p == prev_id and c is not None)

    if n_prev_to_curr > 1:
        event = "merge"
    elif n_prev_to_many > 1:
        event = "split"
    elif len(curr_members) > len(prev_members):
        event = "growth"
    elif len(curr_members) < len(prev_members):
        event = "shrink"
    else:
        event = "stable"

    return {
        "event": event,
        "supernode_id_from": prev_id,
        "supernode_id_to": curr_id,
        "year_from": year_from,
        "year_to": year_to,
        "jaccard": round(jaccard(prev_members, curr_members), 3),
        "n_members_from": len(prev_members),
        "n_members_to": len(curr_members),
        "new_members": list(curr_members - prev_members),
        "lost_members": list(prev_members - curr_members),
    }


def build_event_log(
    hierarchies: dict[int, dict],
    years: list[int] = SNAPSHOT_YEARS,
    levels: list[int] | None = None,
) -> list[dict]:
    """Build the full temporal event log across all snapshots and levels."""
    if levels is None:
        levels = [0, 1]  # track top two levels (as required by T5 labelling)

    events = []
    for i in range(1, len(years)):
        y_from, y_to = years[i - 1], years[i]
        prev_h = hierarchies.get(y_from)
        curr_h = hierarchies.get(y_to)
        if prev_h is None or curr_h is None:
            continue

        prev_sns = prev_h["supernodes"]
        curr_sns = curr_h["supernodes"]

        for level in levels:
            matches = match_supernodes(prev_sns, curr_sns, level)
            for prev_id, curr_id, j in matches:
                event = classify_event(
                    prev_id, curr_id, prev_sns, curr_sns, matches, y_from, y_to
                )
                event["level"] = level
                events.append(event)

    return events
