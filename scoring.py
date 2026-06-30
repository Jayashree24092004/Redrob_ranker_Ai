"""Stage 7 — Hybrid Ranking Engine. Weighted composite over Stage 3-6 outputs.

Weights are a judgment call, documented here rather than buried in the code:
semantic match and required-skill coverage matter most (JD is explicit about
needing real retrieval/ranking production experience); behaviour and location
are weighted enough to matter (the JD explicitly says to down-weight
unavailable candidates) without dominating skill/experience fit; risk is a
penalty, not a filter, in keeping with the JD's "we will probably not move
forward" (not "automatic disqualify") language. Honeypots are filtered
upstream in risk.py, not penalized here.
"""

WEIGHTS = {
    "semantic": 0.27,
    "skills": 0.20,
    "experience": 0.09,
    "career_growth": 0.07,
    "education": 0.03,
    "assessment": 0.04,
    "behaviour": 0.09,
    "recruiter_interest": 0.04,
    "github": 0.03,
    "location": 0.05,
    "platform_trust": 0.05,
    "availability_fit": 0.04,
}
RISK_PENALTY_WEIGHT = 0.20


def composite_score(features: dict, risk_score: float) -> float:
    base = sum(WEIGHTS[k] * features[k] for k in WEIGHTS)
    return max(0.0, base - RISK_PENALTY_WEIGHT * risk_score)


def rank_candidates(rows, top_n=100):
    """rows: list of dicts each containing at least 'candidate_id' and 'score'.
    Sorts by score desc, then candidate_id asc for deterministic tie-breaks
    (per submission_spec.md Section 3), and assigns rank 1..top_n."""
    rows_sorted = sorted(rows, key=lambda r: (-r["score"], r["candidate_id"]))
    top = rows_sorted[:top_n]
    for i, r in enumerate(top):
        r["rank"] = i + 1
    return top
