#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Discovery & Ranking
Single reproducible command:

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline (Stages 1-8): see src/ for each module's docstring.
CPU-only, no network, no GPU during ranking. Designed to comfortably clear
the 5-minute / 16GB budget on 100K candidates (TF-IDF + arithmetic feature
scoring only — no model downloads, no LLM calls, in the ranking step).
"""
import argparse
import csv
import time
import sys

from src.candidate_loader import iter_candidates
from src.jd_blueprint import JD_FULL_TEXT
from src.features import (
    build_semantic_text, skill_match_score, experience_score, career_growth_score,
    education_score, assessment_score, behaviour_score, recruiter_interest_score,
    github_score, location_bonus, platform_trust_score, availability_fit_score,
)
from src.risk import detect_honeypot, assess_jd_risk
from src.semantic import compute_semantic_scores
from src.scoring import composite_score, rank_candidates
from src.explain import generate_reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or .jsonl.gz)")
    ap.add_argument("--out", required=True, help="Output submission CSV path")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()

    # ---- Stage 2/3/5/6 (per-candidate, streamed): feature + risk extraction ----
    records = []        # candidates that survive honeypot filtering
    texts = []           # semantic text, aligned with `records`
    excluded_honeypots = []

    n_seen = 0
    for cand in iter_candidates(args.candidates):
        n_seen += 1
        is_hp, hp_reasons = detect_honeypot(cand)
        if is_hp:
            excluded_honeypots.append((cand["candidate_id"], hp_reasons))
            continue

        text_lower = build_semantic_text(cand)
        risk_score, risk_flags = assess_jd_risk(cand, text_lower)

        feats = {
            "skills": skill_match_score(cand, text_lower),
            "experience": experience_score(cand),
            "career_growth": career_growth_score(cand),
            "education": education_score(cand),
            "assessment": assessment_score(cand),
            "behaviour": behaviour_score(cand),
            "recruiter_interest": recruiter_interest_score(cand),
            "github": github_score(cand),
            "location": location_bonus(cand),
            "platform_trust": platform_trust_score(cand),
            "availability_fit": availability_fit_score(cand),
        }
        records.append({
            "candidate_id": cand["candidate_id"],
            "cand": cand,
            "text_lower": text_lower,
            "feats": feats,
            "risk_score": risk_score,
            "risk_flags": risk_flags,
        })
        texts.append(text_lower)

    print(f"[{time.time()-t0:.1f}s] loaded {n_seen} candidates, "
          f"{len(excluded_honeypots)} excluded as honeypots, {len(records)} scored",
          file=sys.stderr)

    # ---- Stage 4: semantic matching (vectorized over full surviving pool) ----
    sem_scores = compute_semantic_scores(JD_FULL_TEXT, texts)
    for r, s in zip(records, sem_scores):
        r["feats"]["semantic"] = s

    print(f"[{time.time()-t0:.1f}s] semantic scoring done", file=sys.stderr)

    # ---- Stage 7: hybrid ranking ----
    for r in records:
        r["score"] = round(composite_score(r["feats"], r["risk_score"]), 4)

    top = rank_candidates(records, top_n=args.top_n)

    print(f"[{time.time()-t0:.1f}s] ranked, writing top {len(top)}", file=sys.stderr)

    # ---- Stage 8: explanations + CSV write ----
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top:
            reasoning = generate_reasoning(
                r["cand"], r["text_lower"], r["feats"], r["risk_flags"],
            )
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", reasoning])

    print(f"[{time.time()-t0:.1f}s] done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
