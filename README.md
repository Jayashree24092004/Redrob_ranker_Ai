# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A lean, fully-offline ranking pipeline for Redrob AI's "Senior AI Engineer —
Founding Team" job description, built for the Intelligent Candidate Discovery
& Ranking Challenge.

## Why this architecture (and not a bigger one)

The challenge scores exactly three things: a top-100 CSV, a reproducible
GitHub repo, and (for top performers) a sandbox + interview defending the
work. Nothing about FastAPI services, Postgres, Redis, or a Next.js frontend
improves any of those. The compute constraint is explicit and strict:
**CPU-only, no GPU, no network, ≤5 minutes, 16GB RAM, for 100,000
candidates.** This repo is a single Python pipeline, no servers, no DB,
no model downloads at ranking time — by design, not by omission.

## Setup

```bash
pip install -r requirements.txt
```

No other setup. No network access is needed to run `rank.py`.

## Reproduce the submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime on the full 100,000-candidate pool: **~50 seconds**, peak RAM well
under 1GB, single CPU core. (Tested on the actual `candidates.jsonl` from
the hackathon bundle.)

Validate before submitting:

```bash
python validate_submission.py submission.csv
```

## Pipeline (8 stages, `src/`)

| Stage | File | What it does |
|---|---|---|
| 1. JD Intelligence | `src/jd_blueprint.py` | Hand-extracted structured blueprint from the JD (required/preferred skill terms, explicit disqualifiers, location/notice preferences). Extracted once, by reading the JD closely — not worth an LLM agent for a single static document. |
| 2/3. Candidate Intelligence + Feature Engineering | `src/features.py` | Per-candidate scores: skill-term match, experience-band fit, career growth/trajectory, education relevance+tier, assessment average, GitHub activity, location. Pure arithmetic, no ML — needs to run 100K times inside the time budget. |
| 4. Semantic Matching | `src/semantic.py` | TF-IDF (unigrams, capped vocab) + cosine similarity between the JD and each candidate's headline/summary/career-history text. See module docstring for why TF-IDF over sentence-transformers here specifically (no-network constraint). |
| 5. Behaviour Intelligence | `src/features.py::behaviour_score`, `platform_trust_score`, `availability_fit_score` | Uses **22 of the 23** `redrob_signals` fields (the only intentional omission is `expected_salary_range_inr_lpa` — the JD gives no comp band to compare against, so scoring it would be a baseless guess, not a signal). Split into three components: recent *activity/responsiveness* (recency, recruiter response rate, interview completion, notice period, verification), profile *credibility* (completeness, signup tenure, connections, endorsements, offer-acceptance history), and *availability fit* for this specific role (preferred work mode vs. the JD's hybrid Pune/Noida need, willingness to relocate, application-volume sanity check against spray-and-pray). Directly implements the JD's instruction to down-weight unavailable candidates. |
| 6. Risk Detection | `src/risk.py` | Two distinct things: (a) honeypot detection — hard-excluded before ranking, so honeypot rate in the output is 0% by construction; (b) JD-explicit soft disqualifiers (pure-research-only, shallow recent LangChain-only AI experience, architect/manager with no recent coding, consulting-only career, CV/speech background with no NLP overlap, title-chasing) as a continuous risk penalty, matching the JD's "we will probably not move forward" (not automatic-reject) language. |
| 7. Hybrid Ranking | `src/scoring.py` | Weighted composite of all the above, minus the risk penalty. Weights and rationale are documented in the module docstring. Deterministic tie-break by `candidate_id` ascending, per the spec. |
| 8. Explanation | `src/explain.py` | Template-based reasoning built only from fields present on that specific candidate — can't hallucinate a skill they don't have, can't produce identical strings across candidates, since every sentence is data-driven. |

## Honeypot handling

`risk.py::detect_honeypot` flags candidates with (a) 3+ skills claimed at
"expert" proficiency with ~0 months of actual use (the pattern called out in
the hackathon README), or (b) hard timeline impossibilities (overlapping
"current" roles, career-history duration wildly inconsistent with claimed
years of experience). On the real 100K-candidate pool this flags **56
candidates (~0.06% of the pool)** — in the right order of magnitude for the
"~80 honeypots" the spec mentions, after calibrating away an earlier,
looser check that produced false positives (documented in the module
docstring). Honeypots are excluded from the candidate pool *before* ranking,
so the submission's honeypot rate is 0% by construction rather than hoped-for
via a scoring penalty.

## Known limitations / what I'd do with more time

- TF-IDF is a pragmatic stand-in for sentence-transformer embeddings under
  the no-network-at-ranking-time constraint. A vendored, repo-committed
  `all-MiniLM-L6-v2` checkpoint (~90MB) loaded from disk would likely
  improve semantic recall — `src/semantic.py`'s docstring documents the
  swap-in point.
- The "pure research-only" and "consulting-only career" risk checks are
  based on `industry`/`company` string matching against a fixed list; they
  won't catch every real-world case, just the patterns the JD calls out
  explicitly.
- No learned model anywhere — every score is a hand-tuned heuristic. With
  ground-truth relevance labels for even a small labeled subset, the weights
  in `src/scoring.py` could be fit (e.g. logistic regression or LightGBM
  over the same feature set) instead of hand-set.

## AI tools used

Declared honestly in `submission_metadata.yaml`.
