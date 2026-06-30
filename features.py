"""
Stages 2-3-5 — Candidate Intelligence, Feature Engineering, Behaviour Intelligence.

All of this is plain arithmetic over the candidate JSON — deliberately no
LLM calls anywhere in this file, because this runs once per candidate x
100,000 candidates inside the 5-minute CPU budget.
"""
from datetime import date, datetime
from .jd_blueprint import (
    REQUIRED_SKILL_TERMS, PREFERRED_SKILL_TERMS, SENIOR_TITLE_RANK,
    TIER1_INDIAN_LOCATIONS, EDU_RELEVANT_FIELDS, TIER_SCORE, MAX_NOTICE_PREFERRED_DAYS,
)

TODAY = date(2026, 7, 1)


def _clip01(x):
    return max(0.0, min(1.0, x))


def build_semantic_text(cand):
    """Concatenate the fields that actually carry signal about fit, weighting
    headline/summary/career descriptions over raw skill names so a candidate
    can't game this purely by listing keywords (skills are scored separately
    and explicitly down-weighted relative to narrative text)."""
    p = cand["profile"]
    parts = [
        p.get("headline", ""), p.get("headline", ""),  # headline weighted x2
        p.get("summary", ""), p.get("summary", ""),
        p.get("current_title", ""),
    ]
    for job in cand.get("career_history", []):
        parts.append(f"{job.get('title','')} {job.get('description','')}")
    skill_names = [s.get("name", "") for s in cand.get("skills", [])]
    parts.append(" ".join(skill_names))
    return " ".join(parts).lower()


def skill_match_score(cand, text_lower):
    """Required/preferred JD-term coverage, discounted by a trust multiplier
    built from each skill's endorsements + duration_months. A candidate
    whose skill list is mostly long lists of terms with 0 endorsements and
    near-zero duration is exhibiting the exact keyword-stuffing pattern the
    JD calls out as a trap ("a candidate who has all the AI keywords listed
    as skills but whose title is 'Marketing Manager' is not a fit") — this
    doesn't replace the honeypot/risk checks, it just stops keyword-only
    profiles from getting full credit on the skills axis."""
    def hit_fraction(terms):
        if not terms:
            return 0.0
        hits = sum(1 for t in terms if t in text_lower)
        return hits / len(terms)
    req = hit_fraction(REQUIRED_SKILL_TERMS)
    pref = hit_fraction(PREFERRED_SKILL_TERMS)
    raw = _clip01(0.75 * req + 0.25 * pref)

    skills = cand.get("skills", [])
    if not skills:
        return raw * 0.7
    trust_per_skill = []
    for s in skills:
        dur = s.get("duration_months", 0) or 0
        endorsed = s.get("endorsements", 0) or 0
        dur_component = _clip01(dur / 24)        # 2 years of use = fully trusted
        endorse_component = _clip01(endorsed / 10)  # 10 endorsements = fully trusted
        trust_per_skill.append(0.6 * dur_component + 0.4 * endorse_component)
    trust_ratio = sum(trust_per_skill) / len(trust_per_skill)
    # Floor at 0.55 so a real specialist with few-but-deep skills isn't
    # crushed; the multiplier exists to catch breadth-without-depth stuffing.
    multiplier = 0.55 + 0.45 * trust_ratio
    return _clip01(raw * multiplier)


def experience_score(cand):
    yoe = cand["profile"].get("years_of_experience", 0) or 0
    # JD's stated band is 5-9, soft (triangular falloff outside it).
    if 5 <= yoe <= 9:
        band = 1.0
    elif yoe < 5:
        band = _clip01(1 - (5 - yoe) / 5)
    else:
        band = _clip01(1 - (yoe - 9) / 12)
    return band


def career_growth_score(cand):
    career = cand.get("career_history", [])
    if not career:
        return 0.3
    # chronological order: career_history isn't guaranteed sorted, sort by start_date.
    def rank_of(title):
        t = (title or "").lower()
        best = 2  # default to "engineer" level
        for k, v in SENIOR_TITLE_RANK.items():
            if k in t:
                best = max(best, v)
        return best
    sorted_jobs = sorted(career, key=lambda j: j.get("start_date") or "0000")
    ranks = [rank_of(j.get("title")) for j in sorted_jobs]
    non_decreasing = sum(1 for i in range(len(ranks) - 1) if ranks[i + 1] >= ranks[i])
    monotonic_frac = non_decreasing / max(1, len(ranks) - 1) if len(ranks) > 1 else 1.0
    tenure_avg = sum(j.get("duration_months", 0) or 0 for j in career) / len(career)
    tenure_score = _clip01(tenure_avg / 30)  # ~2.5yr average tenure scores 1.0
    return _clip01(0.6 * monotonic_frac + 0.4 * tenure_score)


def education_score(cand):
    edu = cand.get("education", [])
    if not edu:
        return 0.4
    best = 0.0
    for e in edu:
        field = (e.get("field_of_study") or "").lower()
        relevant = any(f in field for f in EDU_RELEVANT_FIELDS)
        tier = TIER_SCORE.get(e.get("tier", "unknown"), 0.5)
        score = tier * (1.0 if relevant else 0.6)
        best = max(best, score)
    return best


def assessment_score(cand):
    scores = cand.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    if not scores:
        return 0.4
    return _clip01(sum(scores.values()) / len(scores) / 100.0)


def _days_since(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 9999
    return (TODAY - d).days


def behaviour_score(cand):
    sig = cand.get("redrob_signals", {})
    recency_days = _days_since(sig.get("last_active_date"))
    if recency_days <= 14:
        recency = 1.0
    elif recency_days <= 30:
        recency = 0.85
    elif recency_days <= 90:
        recency = 0.6
    elif recency_days <= 180:
        recency = 0.35
    elif recency_days <= 365:
        recency = 0.15
    else:
        recency = 0.0

    response_rate = sig.get("recruiter_response_rate", 0) or 0
    interview_completion = sig.get("interview_completion_rate", 0) or 0
    open_to_work = 1.0 if sig.get("open_to_work_flag") else 0.4

    notice = sig.get("notice_period_days", 90) or 90
    if notice <= MAX_NOTICE_PREFERRED_DAYS:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.6
    elif notice <= 90:
        notice_score = 0.3
    else:
        notice_score = 0.1

    resp_time = sig.get("avg_response_time_hours", 48) or 48
    resp_time_score = _clip01(1 - resp_time / 96)

    verified = sum([
        bool(sig.get("verified_email")), bool(sig.get("verified_phone")),
        bool(sig.get("linkedin_connected")),
    ]) / 3.0

    return _clip01(
        0.30 * recency + 0.20 * response_rate + 0.15 * interview_completion
        + 0.15 * notice_score + 0.10 * open_to_work + 0.05 * resp_time_score
        + 0.05 * verified
    )


def recruiter_interest_score(cand):
    sig = cand.get("redrob_signals", {})
    views = sig.get("profile_views_received_30d", 0) or 0
    appearances = sig.get("search_appearance_30d", 0) or 0
    saved = sig.get("saved_by_recruiters_30d", 0) or 0
    # Log-ish normalization against generous ceilings; this is a 30-day
    # activity signal, not a hard cap on candidate quality.
    v = _clip01(views / 50)
    a = _clip01(appearances / 100)
    s = _clip01(saved / 20)
    return _clip01(0.3 * v + 0.3 * a + 0.4 * s)


def github_score(cand):
    g = cand.get("redrob_signals", {}).get("github_activity_score", -1)
    if g is None or g < 0:
        return 0.35  # no GitHub linked: neutral, not penalized hard (JD treats it as a "nice to have")
    return _clip01(g / 100.0)


def location_bonus(cand):
    loc = (cand["profile"].get("location") or "").lower()
    country = (cand["profile"].get("country") or "").lower()
    if any(c in loc for c in TIER1_INDIAN_LOCATIONS):
        return 1.0
    if country == "india":
        return 0.7
    return 0.25  # outside India: JD says case-by-case, no visa sponsorship


def platform_trust_score(cand):
    """Profile-quality / credibility signal, separate from recent-activity
    behaviour_score: how complete and substantiated the profile itself is,
    independent of whether the candidate happened to log in this week."""
    sig = cand.get("redrob_signals", {})

    completeness = _clip01((sig.get("profile_completeness_score", 50) or 50) / 100.0)

    signup_days = _days_since(sig.get("signup_date"))
    # Established profiles (signed up >6mo ago) read as more credible than
    # brand-new ones; cap the benefit at ~2 years tenure.
    tenure = _clip01(signup_days / 730) if signup_days != 9999 else 0.3

    connections = _clip01((sig.get("connection_count", 0) or 0) / 500)
    endorsements = _clip01((sig.get("endorsements_received", 0) or 0) / 50)

    oar = sig.get("offer_acceptance_rate", -1)
    offer_accept = 0.5 if oar is None or oar < 0 else _clip01(oar)  # -1 = no history: neutral

    return _clip01(
        0.30 * completeness + 0.15 * tenure + 0.15 * connections
        + 0.20 * endorsements + 0.20 * offer_accept
    )


def availability_fit_score(cand):
    """How well the candidate's stated availability/work-mode preferences
    line up with what this specific JD needs: Pune/Noida hybrid with
    quarterly in-person offsites, i.e. NOT remote-only, and genuine current
    job-search intent without spray-and-pray over-applying."""
    sig = cand.get("redrob_signals", {})

    mode = (sig.get("preferred_work_mode") or "flexible").lower()
    mode_score = {"hybrid": 1.0, "onsite": 0.9, "flexible": 0.8, "remote": 0.35}.get(mode, 0.6)

    relocate_bonus = 1.0 if sig.get("willing_to_relocate") else 0.5

    # Applying actively signals real job-search intent, but a very high
    # 30-day application count reads as spray-and-pray rather than a
    # targeted search — mild inverse-U, not a straight linear reward.
    apps = sig.get("applications_submitted_30d", 0) or 0
    if apps == 0:
        apps_score = 0.4
    elif apps <= 15:
        apps_score = 0.5 + 0.5 * (apps / 15)
    else:
        apps_score = _clip01(1.0 - (apps - 15) / 60)

    return _clip01(0.45 * mode_score + 0.30 * relocate_bonus + 0.25 * apps_score)
