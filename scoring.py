"""
scoring.py — Single source of truth for all scoring logic.

Imported by: rank.py, sandbox/app.py, scripts/05_eval.py

All scoring constants and compute_*() functions live here.
No other file should redefine these formulas.
"""

from datetime import date

# ── Reference date ──────────────────────────────────────────────────────────
# Fixed reference point for all date-relative computations (last_active_days_ago, etc.)
REFERENCE_DATE = date(2025, 6, 1)

# ── Target cities ───────────────────────────────────────────────────────────
# All India cities explicitly mentioned in the JD or implied by proximity to Noida.
# Single authoritative source — Script 01 uses this to compute is_target_city boolean.
TARGET_CITIES = {
    "pune", "noida", "delhi", "ncr", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "gurgaon", "gurugram", "faridabad",
}

# ── IT services companies ────────────────────────────────────────────────────
# Explicit disqualifier in JD: entire career in IT services consulting.
IT_SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "mphasis", "hexaware",
    "ltimindtree", "tech mahindra", "hcl technologies", "hcltech",
    "ibm global", "atos", "dxc technology", "epam", "niit technologies",
}

# ── JD-relevant skills for skill_assessment_scores bonus ─────────────────────
JD_RELEVANT_SKILLS = {
    "python", "nlp", "machine learning", "deep learning",
    "information retrieval", "ranking", "search", "embeddings",
    "recommendation systems", "data science", "pytorch", "tensorflow",
}

# ── Estimated salary target range (INR LPA) ──────────────────────────────────
# Senior AI Engineer, Series A India startup, 5–9 YoE, AI specialisation.
# Wide range deliberately chosen to reduce false negatives from estimation error.
SALARY_TARGET_MIN = 20.0
SALARY_TARGET_MAX = 65.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_experience_fit(years: float) -> float:
    """
    Score how well years of experience matches JD (sweet spot: 5–9 years).
    JD explicitly states experience band is not a hard requirement.
    Curve plateaus at 10–15 years, tapers slowly after — overqualified ≠ unqualified.
    """
    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5 or 9 < years <= 12:
        return 0.82
    elif 3 <= years < 4 or 12 < years <= 15:
        return 0.65
    elif years > 15:
        return 0.55   # overqualified but not disqualified
    else:
        return 0.25   # < 3 years: genuinely under-experienced


def compute_location_fit(
    is_india_based: bool,
    is_target_city: bool,
    willing_to_relocate: bool,
) -> float:
    """
    Score location fit. JD office is Noida; India-based strongly preferred.
    Uses pre-computed boolean columns from features.parquet (Script 01).
    """
    if not is_india_based:
        return 0.15 if not willing_to_relocate else 0.45
    if is_target_city:
        return 1.0
    elif willing_to_relocate:
        return 0.78
    else:
        return 0.52


def compute_company_fit(
    entire_career_it_services: bool,
    has_product_company_exp: bool,
    has_ml_production_experience: bool,
    years_since_last_ml_role: float,
) -> float:
    """
    Score company + ML production background.

    Hard disqualifier: entire career in IT services (explicit in JD).

    Bug 2 fix: stale ML always scores >= no ML ever.
    When ml_recency decays below 0.65, we floor at 0.65 (= product-no-ML baseline).
    This ensures "product + ML from 4 years ago" >= "product + zero ML experience".
    """
    if entire_career_it_services:
        return 0.0   # hard disqualifier

    # ML recency decay
    if years_since_last_ml_role <= 0:
        ml_recency = 1.0
    elif years_since_last_ml_role <= 1:
        ml_recency = 0.85
    elif years_since_last_ml_role <= 2:
        ml_recency = 0.65
    elif years_since_last_ml_role <= 4:
        ml_recency = 0.40
    else:
        ml_recency = 0.20

    if has_product_company_exp and has_ml_production_experience:
        # Floor at 0.65 — stale ML always beats no-ML-at-all (Bug 2 fix)
        return max(0.65, 1.0 * ml_recency)
    elif has_product_company_exp:
        return 0.65
    else:
        return 0.38


def compute_salary_fit(expected_min: float, expected_max: float) -> float:
    """
    Score salary alignment with estimated role budget (20–65 LPA).
    Soft signal — wide range reduces false negatives from estimation error.
    """
    # Full overlap with target range
    if expected_min <= SALARY_TARGET_MAX and expected_max >= SALARY_TARGET_MIN:
        return 1.0
    # Candidate expects significantly less (may indicate junior level)
    if expected_max < SALARY_TARGET_MIN * 0.7:
        return 0.55
    # Candidate expects significantly more (budget mismatch)
    if expected_min > SALARY_TARGET_MAX * 1.5:
        return 0.35
    # Mild mismatch — partial overlap or close to range
    return 0.72


def compute_skill_assessment_bonus(skill_assessment_scores: dict) -> float:
    """
    Bonus for platform-verified skills in JD-relevant areas.
    Sparse signal (~20% of candidates). When present: high-signal verified ability.
    Max bonus: 0.08 (meaningful tiebreaker, not dominant).
    """
    if not skill_assessment_scores:
        return 0.0
    relevant_scores = []
    for skill_name, score in skill_assessment_scores.items():
        if any(jd_skill in skill_name.lower() for jd_skill in JD_RELEVANT_SKILLS):
            relevant_scores.append(float(score) / 100.0)
    if not relevant_scores:
        return 0.0
    return min(0.08, sum(relevant_scores) / len(relevant_scores) * 0.08)


def compute_education_bonus(edu_tier: str, field_of_study: str) -> float:
    """
    Small education tiebreaker. Not a dominant signal — JD doesn't list education
    as a hard requirement. Used only to differentiate near-equal candidates.
    Range: -0.02 to +0.05.
    """
    RELEVANT_FIELDS = {
        "computer science", "cs", "information technology",
        "machine learning", "data science", "statistics",
        "mathematics", "electrical engineering", "electronics",
    }
    field_lower = (field_of_study or "").lower()
    field_relevant = any(f in field_lower for f in RELEVANT_FIELDS)

    if edu_tier == "tier_1":
        return 0.05 if field_relevant else 0.03
    elif edu_tier == "tier_2":
        return 0.03 if field_relevant else 0.01
    elif edu_tier == "tier_3":
        return 0.0
    elif edu_tier == "tier_4":
        return -0.02
    return 0.0   # unknown tier → neutral


def compute_availability_score(
    open_to_work: bool,
    last_active_days_ago: int,
    recruiter_response_rate: float,
    avg_response_time_hours: float,
    notice_period_days: int,
    saved_by_recruiters_30d: int,
    verified_email: bool,
    verified_phone: bool,
) -> float:
    """
    Availability signal. Five primary sub-signals + social proof + reachability.

    Note: interview_completion_rate intentionally NOT a parameter (I1 fix).
    It was accepted by the function before but never used in the formula body.
    Removed to avoid dead parameter confusion at Stage 4.

    avg_response_time_hours: -1 means no history → treated as neutral (0.5).
    saved_by_recruiters_30d: capped at 20 to avoid over-rewarding in-demand candidates.
    """
    # Recency
    if last_active_days_ago <= 14:
        recency = 1.0
    elif last_active_days_ago <= 30:
        recency = 0.85
    elif last_active_days_ago <= 90:
        recency = 0.55
    elif last_active_days_ago <= 180:
        recency = 0.25
    else:
        recency = 0.08

    # Notice period
    if notice_period_days <= 15:
        notice_score = 1.0
    elif notice_period_days <= 30:
        notice_score = 0.85
    elif notice_period_days <= 60:
        notice_score = 0.60
    elif notice_period_days <= 90:
        notice_score = 0.35
    else:
        notice_score = 0.15

    # Response time (-1 = no history → neutral)
    if avg_response_time_hours < 0:
        response_time_score = 0.5
    elif avg_response_time_hours <= 4:
        response_time_score = 1.0
    elif avg_response_time_hours <= 24:
        response_time_score = 0.75
    elif avg_response_time_hours <= 72:
        response_time_score = 0.40
    else:
        response_time_score = 0.10

    # Social proof — capped at 20
    social_proof = min(saved_by_recruiters_30d, 20) / 20.0

    # Reachability composite
    reachability = 0.5 * float(verified_email) + 0.5 * float(verified_phone)

    availability = (
        0.25 * float(open_to_work) +
        0.22 * recency +
        0.18 * float(recruiter_response_rate) +
        0.15 * notice_score +
        0.10 * response_time_score +
        0.05 * social_proof +
        0.05 * reachability
    )
    return min(1.0, availability)


def compute_structural_score(
    experience_fit: float,
    location_fit: float,
    company_fit: float,
    trajectory_score: float,
    salary_fit: float,
    skill_assessment_bonus: float,
    edu_bonus: float,
) -> float:
    """
    Weighted structural score combining all hard-requirement signals.

    Weights reflect JD priorities:
      company background + ML experience (0.30) > location (0.25) > experience band (0.20)
      > trajectory (0.12) > salary (0.08) > bonuses (additive, max ±0.13)

    Bug 4 fix: return min(1.0, ...) — bonuses can push the weighted sum past 1.0
    without the cap. Capped to keep all scores in [0, 1].
    """
    raw = (
        0.30 * company_fit +
        0.25 * location_fit +
        0.20 * experience_fit +
        0.12 * trajectory_score +
        0.08 * salary_fit
    )
    return min(1.0, raw + skill_assessment_bonus + edu_bonus)
