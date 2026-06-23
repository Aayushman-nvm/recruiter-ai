"""
scoring.py — Pure scoring functions. Single source of truth.

Constants (keywords, companies, cities, seniority, salary) now live in
config/ — import them from there. This file only contains:
  - REFERENCE_DATE
  - compute_*() scoring functions
  - generate_reasoning()
  - compute_structural_score()
  - Pipeline blend weights (PRELIM_*, FINAL_*, FUSION_*)
  - weighted_score_fusion() / rrf_fusion()  (re-exported from pipeline/fusion)

Backward-compatible re-exports keep existing `from scoring import X` calls
working without changes in rank.py, 01_extract_features.py, and 04_eval.py.
"""

from datetime import date

import numpy as np

# ── Config re-exports (backward compat) ──────────────────────────────────────
from config.locations  import TARGET_CITIES, PRIMARY_CITIES
from config.companies  import IT_SERVICES_COMPANIES
from config.keywords   import (
    JD_RELEVANT_SKILLS,
    RESEARCH_ONLY_COMPANY_INDICATORS,
    RESEARCH_ONLY_TITLE_INDICATORS,
    INDUSTRY_RELEVANT_KEYWORDS,
    ML_KEYWORDS,
    ML_KEYWORD_PATTERN,
    has_ml_keyword,
)
from config.seniority  import (
    SENIORITY_HIGH, SENIORITY_SENIOR, SENIORITY_MID, SENIORITY_JUNIOR,
    get_title_seniority,
)
from config.salary     import SALARY_TARGET_MIN, SALARY_TARGET_MAX

# ── Pipeline fusion re-exports (backward compat) ─────────────────────────────
from pipeline.fusion import (
    FUSION_BM25_WEIGHT,
    FUSION_DENSE_WEIGHT,
    weighted_score_fusion,
    rrf_fusion,
)

# ── Reference date ────────────────────────────────────────────────────────────
# Fixed reference point for all date-relative computations.
# Must be >= the latest last_active_date in the dataset.
REFERENCE_DATE = date(2026, 5, 28)


# ── Pipeline blend weights ────────────────────────────────────────────────────
# prelim_score = fusion + structural + availability (pre-cross-encoder)
PRELIM_FUSION_WEIGHT       = 0.55
PRELIM_STRUCTURAL_WEIGHT   = 0.25
PRELIM_AVAILABILITY_WEIGHT = 0.20

# final_score = prelim_score + cross-encoder
# CE gets more weight — it's the only stage with full cross-attention over
# JD + candidate text vs prelim which is still partly keyword/rule-driven.
FINAL_PRELIM_WEIGHT = 0.30
FINAL_CE_WEIGHT     = 0.70

assert abs(PRELIM_FUSION_WEIGHT + PRELIM_STRUCTURAL_WEIGHT + PRELIM_AVAILABILITY_WEIGHT - 1.0) < 1e-9
assert abs(FINAL_PRELIM_WEIGHT + FINAL_CE_WEIGHT - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_experience_fit(years: float) -> float:
    """
    Score experience match. Sweet spot: 5–9 years.

    Curve changes (vs original):
      4–5 yrs: 0.75 (was 0.82) — just below band, small but real gap
      3–4 yrs: 0.55 (was 0.65) — meaningful under-experience
      >15 yrs: 0.40 (was 0.55) — high probability of no longer writing
                                   production code (JD explicit concern)
    """
    if 5 <= years <= 9:    return 1.0
    elif 9 < years <= 12:  return 0.82
    elif 4 <= years < 5:   return 0.75
    elif 12 < years <= 15: return 0.65
    elif 3 <= years < 4:   return 0.55
    elif years > 15:       return 0.40
    else:                  return 0.25   # < 3 years


def compute_location_fit(
    is_india_based: bool,
    is_target_city: bool,
    willing_to_relocate: bool,
    is_primary_city: bool = False,
) -> float:
    """
    Score location fit.

    Tiers:
      primary city (Pune/Noida): 1.0
      JD-welcomed India city:    0.88
      India, willing to relocate: 0.78
      India, other city:         0.52
      abroad, willing to self-relocate: 0.55  (was 0.45 — case-by-case per JD)
      abroad, not willing:       0.15
    """
    if not is_india_based:
        return 0.55 if willing_to_relocate else 0.15
    if is_primary_city:  return 1.0
    if is_target_city:   return 0.88
    if willing_to_relocate: return 0.78
    return 0.52


def compute_company_fit(
    entire_career_it_services: bool,
    has_product_company_exp: bool,
    has_ml_production_experience: bool,
    years_since_last_ml_role: float,
    entire_career_research_only: bool = False,
    shallow_recent_ml_only: bool = False,
) -> float:
    """
    Score company + ML production background.

    Hard disqualifiers (return 0.0):
      - entire career in IT services (jd.txt explicit)
      - entire career in pure research with no production deployment (jd.txt explicit)

    shallow_recent_ml_only: recent LangChain-only experience without prior
    depth — treated as if no ML experience for scoring purposes.

    ML recency decay ensures stale ML always beats no-ML-ever (floor at 0.65).
    """
    if entire_career_it_services or entire_career_research_only:
        return 0.0

    effective_has_ml = has_ml_production_experience and not shallow_recent_ml_only

    if years_since_last_ml_role <= 0:   ml_recency = 1.0
    elif years_since_last_ml_role <= 1: ml_recency = 0.85
    elif years_since_last_ml_role <= 2: ml_recency = 0.65
    elif years_since_last_ml_role <= 4: ml_recency = 0.40
    else:                               ml_recency = 0.20

    if has_product_company_exp and effective_has_ml:
        if years_since_last_ml_role >= 99.0:
            return 0.65
        return max(0.65, 1.0 * ml_recency)
    elif has_product_company_exp:
        return 0.65
    else:
        return 0.38


def compute_salary_fit(expected_min: float, expected_max: float) -> float:
    """Score salary alignment with estimated role budget (20–65 LPA). Soft signal."""
    if expected_min <= SALARY_TARGET_MAX and expected_max >= SALARY_TARGET_MIN:
        return 1.0
    if expected_max < SALARY_TARGET_MIN * 0.7:
        return 0.55
    if expected_min > SALARY_TARGET_MAX * 1.5:
        return 0.35
    return 0.72


def compute_skill_assessment_bonus(skill_assessment_scores: dict) -> float:
    """
    Bonus for platform-verified skills in JD-relevant areas.
    Sparse signal (~20% of candidates). Max bonus: 0.08.
    """
    if not skill_assessment_scores:
        return 0.0
    relevant = [
        float(score) / 100.0
        for skill_name, score in skill_assessment_scores.items()
        if any(jd_skill in skill_name.lower() for jd_skill in JD_RELEVANT_SKILLS)
    ]
    if not relevant:
        return 0.0
    return min(0.08, sum(relevant) / len(relevant) * 0.08)


def compute_education_bonus(edu_tier: str, field_of_study: str) -> float:
    """Small education tiebreaker. Range: -0.02 to +0.05."""
    RELEVANT_FIELDS = {
        "computer science", "cs", "information technology",
        "machine learning", "data science", "statistics",
        "mathematics", "electrical engineering", "electronics",
    }
    field_relevant = any(f in (field_of_study or "").lower() for f in RELEVANT_FIELDS)
    if edu_tier == "tier_1":   return 0.05 if field_relevant else 0.03
    elif edu_tier == "tier_2": return 0.03 if field_relevant else 0.01
    elif edu_tier == "tier_3": return 0.0
    elif edu_tier == "tier_4": return -0.02
    return 0.0


def compute_industry_bonus(current_industry: str) -> float:
    """Small tiebreaker for HR-tech / marketplace background (jd.txt nice-to-have)."""
    industry = (current_industry or "").lower()
    return 0.02 if any(kw in industry for kw in INDUSTRY_RELEVANT_KEYWORDS) else 0.0


def compute_github_bonus(github_activity_score) -> float:
    """Small tiebreaker for open-source activity (jd.txt nice-to-have)."""
    score = float(github_activity_score) if github_activity_score is not None else -1.0
    if score < 0:
        return 0.0
    return round(0.03 * max(0.0, min(1.0, score)), 4)


def compute_availability_score(
    open_to_work: bool,
    last_active_days_ago: int,
    recruiter_response_rate: float,
    avg_response_time_hours: float,
    notice_period_days: int,
    saved_by_recruiters_30d: int,
    verified_email: bool,
    verified_phone: bool,
    interview_completion_rate: float = -1.0,
    offer_acceptance_rate: float = -1.0,
) -> float:
    """Composite availability signal. All sub-scores in [0, 1]."""
    # Recency
    if last_active_days_ago <= 14:   recency = 1.0
    elif last_active_days_ago <= 30: recency = 0.85
    elif last_active_days_ago <= 90: recency = 0.55
    elif last_active_days_ago <= 180: recency = 0.25
    else:                            recency = 0.08

    # Notice period
    if notice_period_days <= 15:   notice_score = 1.0
    elif notice_period_days <= 30: notice_score = 0.85
    elif notice_period_days <= 60: notice_score = 0.60
    elif notice_period_days <= 90: notice_score = 0.35
    else:                          notice_score = 0.15

    # Response time (-1 = no history → neutral 0.5)
    if avg_response_time_hours < 0:    response_time_score = 0.5
    elif avg_response_time_hours <= 4:  response_time_score = 1.0
    elif avg_response_time_hours <= 24: response_time_score = 0.75
    elif avg_response_time_hours <= 72: response_time_score = 0.40
    else:                               response_time_score = 0.10

    # Interview / offer history (-1 = no history → neutral 0.5)
    interview_score = 0.5 if interview_completion_rate < 0 else float(interview_completion_rate)
    offer_score     = 0.5 if offer_acceptance_rate     < 0 else float(offer_acceptance_rate)

    social_proof = min(saved_by_recruiters_30d, 20) / 20.0
    reachability = 0.5 * float(verified_email) + 0.5 * float(verified_phone)

    availability = (
        0.22 * float(open_to_work) +
        0.20 * recency +
        0.16 * float(recruiter_response_rate) +
        0.13 * notice_score +
        0.08 * response_time_score +
        0.07 * interview_score +
        0.05 * offer_score +
        0.04 * social_proof +
        0.05 * reachability
    )
    return min(1.0, max(0.0, availability))


def compute_structural_score(
    experience_fit: float,
    location_fit: float,
    company_fit: float,
    trajectory_score: float,
    salary_fit: float,
    skill_assessment_bonus: float,
    edu_bonus: float,
    industry_bonus: float = 0.0,
    github_bonus: float = 0.0,
    n_it_services_roles: int = 0,
    job_hop_score: float = 0.0,
) -> float:
    """
    Weighted structural score. Weights:
      company/ML (0.32) > location (0.25) > experience (0.20)
      > trajectory (0.15) > salary (0.02) + small additive bonuses

    Penalties:
      n_it_services_roles: -0.02/-0.04/-0.06 graduated (not a hard disqualifier)
      job_hop_score (yoe/n_roles): -0.05 if <1.5yr/company, -0.02 if <2.5yr
    """
    raw = (
        0.32 * company_fit +
        0.25 * location_fit +
        0.20 * experience_fit +
        0.15 * trajectory_score +
        0.02 * salary_fit
    )

    if n_it_services_roles >= 5:   it_penalty = -0.06
    elif n_it_services_roles >= 3: it_penalty = -0.04
    elif n_it_services_roles >= 1: it_penalty = -0.02
    else:                          it_penalty = 0.0

    if job_hop_score <= 0:        hop_penalty = 0.0
    elif job_hop_score < 1.5:     hop_penalty = -0.05
    elif job_hop_score < 2.5:     hop_penalty = -0.02
    else:                         hop_penalty = 0.0

    bonuses = skill_assessment_bonus + edu_bonus + industry_bonus + github_bonus
    return min(1.0, max(0.0, raw + bonuses + it_penalty + hop_penalty))


def generate_reasoning(row: dict) -> str:
    """
    Generate 2-sentence reasoning from feature columns — no LLM required.

    Sentence 1: dominant signal (disqualifiers first, then strongest positive).
    Sentence 2: concrete secondary fact that qualifies or reinforces sentence 1.
    """
    title        = str(row.get("current_title", "candidate") or "candidate").strip()
    yoe          = float(row.get("years_of_experience", 0) or 0)
    company      = str(row.get("current_company", "") or "").strip()
    location     = str(row.get("location", "") or "").strip()
    country      = str(row.get("country", "") or "").strip()

    is_india         = bool(row.get("is_india_based", False))
    is_target_city   = bool(row.get("is_target_city", False))
    willing_relocate = bool(row.get("willing_to_relocate", False))
    entire_it        = bool(row.get("entire_career_it_services", False))
    entire_research  = bool(row.get("entire_career_research_only", False))
    shallow_ml_only  = bool(row.get("shallow_recent_ml_only", False))
    has_product      = bool(row.get("has_product_company_exp", False))
    has_ml           = bool(row.get("has_ml_production_experience", False))
    yrs_since_ml     = float(row.get("years_since_last_ml_role") if row.get("years_since_last_ml_role") is not None else 99)
    traj_score       = float(row.get("trajectory_score") if row.get("trajectory_score") is not None else 0)
    avg_tenure       = float(row.get("avg_tenure_months") if row.get("avg_tenure_months") is not None else 0)
    trajectory_up    = bool(row.get("trajectory_upward", False))

    sal_min      = float(row.get("salary_min_lpa") if row.get("salary_min_lpa") is not None else 0)
    sal_max      = float(row.get("salary_max_lpa") if row.get("salary_max_lpa") is not None else 999)
    open_to_work = bool(row.get("open_to_work_flag", False))
    days_ago     = int(row.get("last_active_days_ago") if row.get("last_active_days_ago") is not None else 999)
    response_rate = float(row.get("recruiter_response_rate") if row.get("recruiter_response_rate") is not None else 0)
    notice_days  = int(row.get("notice_period_days") if row.get("notice_period_days") is not None else 90)
    resp_time_hrs = float(row.get("avg_response_time_hours") if row.get("avg_response_time_hours") is not None else -1)
    saves        = int(row.get("saved_by_recruiters_30d") if row.get("saved_by_recruiters_30d") is not None else 0)
    skill_bonus  = float(row.get("skill_assessment_bonus") if row.get("skill_assessment_bonus") is not None else 0)
    edu_tier     = str(row.get("edu_tier", "unknown") or "unknown")
    ce_score     = float(row.get("ce_score") if row.get("ce_score") is not None else 0)
    rank         = int(row.get("rank") if row.get("rank") is not None else 0)
    n_it         = int(row.get("n_it_services_roles", 0) or 0)

    yoe_str     = f"{yoe:.1f}"
    loc_str     = f"{location}, {country}" if location and country else (location or country or "unknown location")
    company_str = company if company else "their current employer"

    # ── Sentence 1: Lead signal ───────────────────────────────────────────────
    if entire_it:
        s1 = (f"{title} with {yoe_str} years of experience whose entire career "
              f"has been in IT services consulting, which the JD explicitly disqualifies.")
    elif entire_research:
        s1 = (f"{title} with {yoe_str} years of experience whose career has been entirely "
              f"in research/academic roles with no detected production deployment — the JD "
              f"is explicit that this is not a fit.")
    elif shallow_ml_only:
        s1 = (f"{title} with {yoe_str} years of experience whose only detected AI/ML signal "
              f"is a single recent role under a year — the JD specifically flags recent "
              f"LangChain/API-only experience without substantial prior ML production work.")
    elif not has_ml and not has_product:
        s1 = (f"{title} with {yoe_str} years of experience at {company_str}, "
              f"with no detected production ML experience and no product-company background.")
    elif has_ml and yrs_since_ml <= 1:
        if yoe > 15:
            s1 = (f"{title} with {yoe_str} years of experience currently working in "
                  f"production ML at {company_str} — strong ML signal, but at {yoe_str} years "
                  f"the JD flags a risk of having moved away from hands-on production coding.")
        else:
            s1 = (f"{title} with {yoe_str} years of experience currently working in "
                  f"production ML at {company_str}, directly matching the role's core requirement.")
    elif has_ml and 1 < yrs_since_ml <= 3:
        s1 = (f"{title} with {yoe_str} years of experience who last worked in a production ML role "
              f"{yrs_since_ml:.1f} years ago at {company_str}, bringing relevant but dated hands-on experience.")
    elif has_ml and yrs_since_ml > 3:
        if yoe > 15:
            s1 = (f"{title} with {yoe_str} years of experience whose most recent production ML role "
                  f"was {yrs_since_ml:.1f} years ago — combined with {yoe_str} years total experience, "
                  f"this raises concern that they've moved into architecture rather than hands-on coding.")
        else:
            s1 = (f"{title} with {yoe_str} years of experience whose most recent production ML role "
                  f"was {yrs_since_ml:.1f} years ago — the role requires active hands-on ML work, not historical.")
    elif has_product and not has_ml:
        s1 = (f"{title} with {yoe_str} years at product companies including {company_str}, "
              f"but no explicit production ML or retrieval/ranking experience detected in their career history.")
    elif ce_score >= 0.7:
        s1 = (f"{title} with {yoe_str} years of experience whose profile content closely "
              f"matches the technical requirements of this role (semantic match: {ce_score:.2f}).")
    else:
        s1 = (f"{title} with {yoe_str} years of experience at {company_str}, "
              f"ranked {rank} based on a combination of semantic fit and structured signals.")

    # ── Sentence 2: Supporting / qualifying signal ────────────────────────────
    if entire_it or entire_research or shallow_ml_only:
        if open_to_work and days_ago <= 30:
            s2 = (f"Despite this, they are actively available "
                  f"(open to work, last active {days_ago} days ago, response rate {response_rate:.0%}).")
        else:
            s2 = (f"Their recruiter response rate is {response_rate:.0%} "
                  f"and they were last active {days_ago} days ago.")

    elif not is_india:
        # Fix F: distinguish self-funded relocation from refusal
        if willing_relocate:
            s2 = (f"Based in {loc_str} and willing to self-fund relocation to India — "
                  f"the JD handles these cases individually (no visa sponsorship).")
        else:
            s2 = (f"Based in {loc_str} and not willing to relocate — "
                  f"significant location mismatch for a Noida/Pune-based role.")

    elif notice_days > 90:
        # Fix E: long notice is a concrete hiring blocker — surface it first
        it_note = ""
        if n_it >= 1:
            it_note = (f" Note: {n_it} IT-services role(s) detected "
                       f"(including {company_str}) — a graduated penalty applies per the JD.")
        s2 = (f"Notice period of {notice_days} days exceeds the JD's preferred 30-day "
              f"buyout window — a practical hiring delay.{it_note}")

    elif n_it >= 1 and not entire_it:
        # Fix E: partial IT-services career — explain the penalty
        severity = "minor" if n_it <= 2 else ("moderate" if n_it <= 4 else "strong")
        s2 = (f"{n_it} IT-services role(s) in career history "
              f"(including {company_str}) — {severity} JD-concern penalty applied; "
              f"not a hard disqualifier given prior product-company experience.")

    elif is_india and not is_target_city and not willing_relocate:
        s2 = (f"Currently in {loc_str}, not in a JD-preferred city and not open to relocation "
              f"— partial location fit only.")

    elif skill_bonus >= 0.04:
        s2 = (f"Platform-verified skill assessments (bonus: {skill_bonus:.3f}) provide "
              f"independently confirmed technical ability, a signal absent in most candidates.")

    elif traj_score >= 0.8 and trajectory_up:
        tenure_str = f"{avg_tenure:.0f}" if avg_tenure > 0 else "unknown"
        s2 = (f"Their career shows upward title progression with an average tenure of "
              f"{tenure_str} months per role, indicating stability and growth rather than title-chasing.")

    elif avg_tenure < 15 and avg_tenure > 0:
        s2 = (f"Average tenure of {avg_tenure:.0f} months per role suggests a title-chasing "
              f"pattern — a stated JD concern — which the trajectory score ({traj_score:.2f}) reflects.")

    elif open_to_work and days_ago <= 14 and response_rate >= 0.7:
        s2 = (f"Actively job-seeking: open to work, last active {days_ago} days ago, "
              f"and {response_rate:.0%} recruiter response rate — highly reachable.")

    elif open_to_work and notice_days <= 30:
        notice_str = "immediately" if notice_days == 0 else f"within {notice_days} days"
        s2 = (f"Open to work and can start {notice_str}, "
              f"last active {days_ago} days ago with a {response_rate:.0%} response rate.")

    elif days_ago > 180:
        s2 = (f"Last active {days_ago} days ago with a {response_rate:.0%} response rate — "
              f"low platform engagement raises reachability concerns despite their technical profile.")

    elif resp_time_hrs >= 0 and resp_time_hrs <= 4:
        s2 = (f"Responds to recruiter messages within {resp_time_hrs:.0f} hours on average "
              f"({response_rate:.0%} response rate), indicating strong engagement.")

    elif saves >= 5:
        s2 = (f"Saved by {saves} other recruiters in the last 30 days — "
              f"crowd-validated interest from the recruiting community.")

    elif sal_min > SALARY_TARGET_MAX * 1.3:
        s2 = (f"Expected salary (min {sal_min:.0f} LPA) likely exceeds the role's budget "
              f"for a Series A startup — potential offer negotiation risk.")

    elif sal_max < SALARY_TARGET_MIN * 0.7:
        s2 = (f"Expected salary range ({sal_min:.0f}–{sal_max:.0f} LPA) is below the "
              f"estimated market rate for this seniority, possibly indicating a junior profile.")

    elif edu_tier == "tier_1":
        s2 = (f"Tier-1 institution background adds a marginal quality signal "
              f"as a tiebreaker ({response_rate:.0%} response rate, {days_ago} days since last active).")

    else:
        active_str = "recently" if days_ago <= 30 else f"{days_ago} days ago"
        s2 = (f"Last active {active_str} with a {response_rate:.0%} recruiter response rate "
              f"and {notice_days}-day notice period.")

    return f"{s1} {s2}"
