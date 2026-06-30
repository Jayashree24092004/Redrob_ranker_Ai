"""
Stage 6 — Risk Detection.

Two distinct things happen here, and they're kept separate on purpose:

1. HONEYPOT detection — profiles that are internally *impossible*
   (skills used longer than the candidate's total experience, "expert" in a
   skill they've used for 0 months, timeline math that doesn't add up). These
   are hard-excluded from the candidate pool before ranking even starts, so
   the submission's honeypot rate is 0% by construction rather than something
   we hope a scoring penalty pushes low enough.

2. JD-explicit DISQUALIFIERS / soft risk — things the JD says it will
   probably or definitely reject (pure-research-only background, recent
   LangChain-wrapper-only "AI experience", architects who haven't coded in
   18 months, career spent entirely at consulting firms, CV/speech/robotics
   background with no NLP/IR overlap, title-chasers). These produce a
   continuous risk_score in [0, 1] (higher = riskier) that feeds the hybrid
   ranking engine as a negative-weighted term, rather than a hard cut —
   matching the JD's own language ("we will probably not move forward",
   not "automatic rejection").
"""
from datetime import date
from .jd_blueprint import (
    CONSULTING_FIRMS, CV_SPEECH_ROBOTICS_TERMS, NLP_IR_TERMS,
    LANGCHAIN_SHALLOW_TERMS, PRE_LLM_ML_TERMS, NON_CODING_TITLE_TERMS,
)


def _months_between(start, end):
    if not start:
        return 0
    try:
        sy, sm = int(start[:4]), int(start[5:7])
    except (ValueError, TypeError):
        return 0
    if end:
        try:
            ey, em = int(end[:4]), int(end[5:7])
        except (ValueError, TypeError):
            ey, em = date.today().year, date.today().month
    else:
        ey, em = date.today().year, date.today().month
    return max(0, (ey - sy) * 12 + (em - sm))


def detect_honeypot(cand):
    """Returns (is_honeypot: bool, reasons: list[str]).

    Calibrated against the actual dataset (checked against a 20K-row sample):
    skill_duration > years_of_experience turns out to be common in
    *legitimate* profiles here (skills picked up before the current job /
    in school), so it is NOT used as a signal on its own — only the
    redrob_signals_doc-described pattern of *multiple* "expert" skills with
    ~zero months of use (textbook keyword-stuffing honeypot) is treated as
    disqualifying, along with hard timeline impossibilities.
    """
    reasons = []
    yoe_months = float(cand["profile"].get("years_of_experience", 0)) * 12

    # 1. Multiple "expert" skills claimed with ~0 months of actual use —
    #    the specific pattern called out in README.docx.
    zero_dur_experts = [
        sk.get("name") for sk in cand.get("skills", [])
        if sk.get("proficiency") == "expert" and (sk.get("duration_months") or 0) <= 1
    ]
    if len(zero_dur_experts) >= 3:
        reasons.append(
            f"{len(zero_dur_experts)} skills claimed at 'expert' proficiency with ~0 months of use "
            f"(e.g. {', '.join(zero_dur_experts[:3])})"
        )

    # 2. Career history timeline sanity: explicit duration vs start/end dates,
    #    and overlapping-current-role checks. Generous tolerance (12mo) since
    #    month-level rounding noise is common in legitimate synthetic data.
    current_roles = 0
    for job in cand.get("career_history", []):
        computed = _months_between(job.get("start_date"), job.get("end_date"))
        claimed = job.get("duration_months")
        if claimed is not None and abs(claimed - computed) > 12:
            reasons.append(
                f"'{job.get('title')}' at '{job.get('company')}' claims {claimed}mo "
                f"but dates imply ~{computed}mo"
            )
        if job.get("is_current"):
            current_roles += 1
    if current_roles > 1:
        reasons.append(f"{current_roles} roles simultaneously marked is_current")

    # 3. Total career-history months wildly exceeds claimed years_of_experience
    #    (large, generous margin — this is a sanity backstop, not the primary signal).
    total_hist_months = sum(j.get("duration_months", 0) or 0 for j in cand.get("career_history", []))
    if total_hist_months > yoe_months + 36:
        reasons.append(
            f"career history sums to ~{total_hist_months}mo but profile claims "
            f"{cand['profile'].get('years_of_experience')} years"
        )

    return (len(reasons) > 0, reasons)


def assess_jd_risk(cand, all_text_lower):
    """Returns risk_score in [0,1] (higher = worse fit per JD's explicit
    disqualifiers) and a short list of human-readable flags for the
    reasoning generator."""
    flags = []
    risk = 0.0

    career = cand.get("career_history", [])
    companies = [j.get("company", "") for j in career]
    industries_lower = [str(j.get("industry", "")).lower() for j in career] + \
        [str(cand["profile"].get("current_industry", "")).lower()]

    # Pure research-only background.
    research_terms = ["academia", "research", "university", "research lab"]
    if career and all(any(t in ind for t in research_terms) for ind in industries_lower):
        risk += 0.30
        flags.append("career entirely in academia/research, no production deployment signal")

    # Career spent entirely at consulting/services firms.
    def _is_consulting(name):
        n = (name or "").lower()
        return any(f in n for f in CONSULTING_FIRMS)
    if companies and all(_is_consulting(c) for c in companies):
        risk += 0.20
        flags.append("entire career at consulting/services firms, no product-company experience")

    # Recent LangChain/OpenAI-wrapper-only AI experience, no pre-LLM ML depth.
    skills = cand.get("skills", [])
    shallow_ai = [s for s in skills if any(t in s.get("name", "").lower() for t in LANGCHAIN_SHALLOW_TERMS)]
    deep_ml = [s for s in skills if any(t in s.get("name", "").lower() for t in PRE_LLM_ML_TERMS)
               and (s.get("duration_months") or 0) >= 24]
    if shallow_ai and not deep_ml and all((s.get("duration_months") or 0) < 12 for s in shallow_ai):
        risk += 0.15
        flags.append("AI experience looks limited to recent LangChain/API-wrapper work, no pre-LLM ML depth")

    # Architect/manager title, hasn't likely written production code recently.
    cur_title = cand["profile"].get("current_title", "").lower()
    cur_role = next((j for j in career if j.get("is_current")), None)
    cur_dur = (cur_role or {}).get("duration_months", 0) or 0
    if any(t in cur_title for t in NON_CODING_TITLE_TERMS) and cur_dur >= 18:
        risk += 0.15
        flags.append(f"current title '{cur_title}' for {cur_dur}mo suggests limited recent hands-on coding")

    # CV/speech/robotics background without NLP/IR overlap.
    text_has_cv = any(t in all_text_lower for t in CV_SPEECH_ROBOTICS_TERMS)
    text_has_nlp = any(t in all_text_lower for t in NLP_IR_TERMS)
    if text_has_cv and not text_has_nlp:
        risk += 0.15
        flags.append("background appears CV/speech/robotics-focused with no NLP/IR/search overlap")

    # Title-chasing: 3+ short stints (<18mo) with escalating seniority titles.
    short_escalating = 0
    for j in career:
        dur = j.get("duration_months", 0) or 0
        title = (j.get("title") or "").lower()
        if dur < 18 and any(t in title for t in ["senior", "staff", "principal", "lead"]):
            short_escalating += 1
    if short_escalating >= 3:
        risk += 0.15
        flags.append("career pattern shows frequent short stints with escalating titles (title-chasing)")

    return min(risk, 1.0), flags
