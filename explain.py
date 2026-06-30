"""
Stage 8 — Explanation Generator. No LLM (matches compute constraints and the
spec's explicit penalty for hallucinated/templated reasoning). Every sentence
below is built only from fields actually present on the candidate, so the
reasoning can't mention a skill the candidate doesn't have.
"""
from .jd_blueprint import REQUIRED_SKILL_TERMS, TIER1_INDIAN_LOCATIONS


def _matched_required_terms(text_lower, limit=3):
    hits = [t for t in REQUIRED_SKILL_TERMS if t in text_lower]
    return hits[:limit]


def generate_reasoning(cand, text_lower, features, risk_flags, honeypot_excluded=False):
    p = cand["profile"]
    sig = cand.get("redrob_signals", {})
    name_bits = []

    yoe = p.get("years_of_experience")
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    loc = p.get("location", "")

    name_bits.append(f"{yoe:.1f}y experience, currently {title} at {company} ({loc}).")

    matched = _matched_required_terms(text_lower)
    if matched:
        name_bits.append(f"Profile shows direct overlap on: {', '.join(matched)}.")
    elif features["skills"] < 0.15:
        name_bits.append("Limited direct overlap with the core retrieval/ranking skill set in the JD.")

    if features["career_growth"] >= 0.7:
        name_bits.append("Career trajectory shows steady upward progression.")
    elif features["career_growth"] <= 0.35:
        name_bits.append("Career trajectory is flat or shows frequent short stints.")

    notice = sig.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            name_bits.append(f"Notice period of {notice} days is within preferred range.")
        else:
            name_bits.append(f"Notice period of {notice} days exceeds the JD's 30-day preference.")

    if sig.get("recruiter_response_rate") is not None:
        rr = sig["recruiter_response_rate"]
        if rr < 0.2:
            name_bits.append(f"Low recruiter response rate ({rr:.0%}) suggests limited current engagement.")
        elif rr > 0.6:
            name_bits.append(f"Strong recruiter response rate ({rr:.0%}).")

    if any(c in loc.lower() for c in TIER1_INDIAN_LOCATIONS):
        name_bits.append("Located in a JD-preferred city.")
    elif sig.get("willing_to_relocate"):
        name_bits.append("Not currently in a preferred city but open to relocation.")
    elif (sig.get("preferred_work_mode") or "").lower() == "remote":
        name_bits.append("Prefers fully remote, which is a weaker fit for this hybrid Pune/Noida role.")

    if risk_flags:
        name_bits.append("Caution: " + "; ".join(risk_flags[:2]) + ".")

    if honeypot_excluded:
        return "EXCLUDED: " + " ".join(risk_flags)

    return " ".join(name_bits)
