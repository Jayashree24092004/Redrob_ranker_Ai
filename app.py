"""
HuggingFace Spaces entrypoint (Gradio). Satisfies submission_spec.md Section
10.5: a hosted sandbox that accepts a small candidate sample (<=100) and runs
the same ranking pipeline end-to-end, producing a ranked CSV.

This file does NOT reimplement any ranking logic — it calls the exact same
src/ modules used by rank.py, so what you see here running on a small sample
is provably the same code path the organizers reproduce at full scale in
Stage 3.
"""
import json
import tempfile
import time
import gradio as gr

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


def _load_candidates(file_path):
    candidates = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Accept either a JSON array (sample_candidates.json style) or JSONL.
    if text.startswith("["):
        candidates = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def run_ranking(file, top_n):
    if file is None:
        return "Upload a candidates file first (.json array or .jsonl).", None

    t0 = time.time()
    candidates = _load_candidates(file.name)
    if len(candidates) > 200:
        return f"This sandbox is for small samples only ({len(candidates)} given; please keep it ≤200).", None

    records, texts, excluded = [], [], []
    for cand in candidates:
        is_hp, hp_reasons = detect_honeypot(cand)
        if is_hp:
            excluded.append((cand["candidate_id"], hp_reasons))
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
            "candidate_id": cand["candidate_id"], "cand": cand,
            "text_lower": text_lower, "feats": feats,
            "risk_score": risk_score, "risk_flags": risk_flags,
        })
        texts.append(text_lower)

    if not records:
        return f"All {len(candidates)} candidates were excluded as honeypots.", None

    sem_scores = compute_semantic_scores(JD_FULL_TEXT, texts)
    for r, s in zip(records, sem_scores):
        r["feats"]["semantic"] = s
        r["score"] = round(composite_score(r["feats"], r["risk_score"]), 4)

    n = min(int(top_n), len(records))
    top = rank_candidates(records, top_n=n)

    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    import csv
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top:
            reasoning = generate_reasoning(r["cand"], r["text_lower"], r["feats"], r["risk_flags"])
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", reasoning])

    elapsed = time.time() - t0
    summary = (
        f"Ranked {len(records)}/{len(candidates)} candidates "
        f"({len(excluded)} excluded as honeypots) in {elapsed:.2f}s. "
        f"Top {n} written below."
    )
    return summary, out_path


with gr.Blocks(title="Redrob Ranker Sandbox") as demo:
    gr.Markdown(
        "# Redrob Hackathon — Ranker Sandbox\n"
        "Upload a small candidate sample (JSON array like `sample_candidates.json`, "
        "or `.jsonl`, ≤200 rows) to verify the ranking pipeline runs end-to-end. "
        "This calls the exact same `src/` code as `rank.py` in the repo — "
        "no reimplementation, no LLM calls, CPU-only."
    )
    with gr.Row():
        file_input = gr.File(label="Candidate sample (.json or .jsonl)")
        top_n_input = gr.Number(label="Top N", value=20, precision=0)
    run_btn = gr.Button("Run ranking", variant="primary")
    status_out = gr.Textbox(label="Status", interactive=False)
    file_out = gr.File(label="Ranked CSV output")

    run_btn.click(run_ranking, inputs=[file_input, top_n_input], outputs=[status_out, file_out])

if __name__ == "__main__":
    demo.launch()
