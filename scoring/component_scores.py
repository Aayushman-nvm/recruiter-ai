"""
scoring/component_scores.py — Individual scoring functions.

Each function scores one dimension of a candidate against the JD.
All are pure functions with no side effects.
"""

from datetime import date

from config.locations import TARGET_CITIES, PRIMARY_CITIES
from config.keywords  import SKILL_ASSESSMENT_JD_RELEVANT, INDUSTRY_RELEVANT_KEYWORDS
from config.salary    import SALARY_TARGET_MIN, SALARY_TARGET_MAX
from config.weights   import (
    NOTICE_SCORE_0_15, NOTICE_SCORE_16_30, NOTICE_SCORE_31_60,
    NOTICE_SCORE_61_90, NOTICE_SCORE_91_120, NOTICE_SCORE_120P,
)


def compute_experience_fit(years: float) -> float:
    """
    Score experience match. Sweet spot: 5–9 years (JD explicit band).

      5–9 yrs:   1.00 — ideal band
      9–12 yrs:  0.82 — slightly over, but JD says "still consider if signals strong"
      4–5 yrs:   0.75 — just below band, small but real gap
      12–15 yrs: 0.65
      3–4 yrs:   0.55 — meaningful under-experience
      >15 yrs:   0.40 — JD flags hands-on coding risk at this tenure
      <3 yrs:    0.25
    """
    if 5 <= years <= 9:    return 1.0
    elif 9 < years <= 12:  return 0.82
    elif 4 <= years < 5:   return 0.75
    elif 12 < years <= 15: return 0.65
    elif 3 <= years < 4:   return 0.55
    elif years > 15:       return 0.40
    else:                  return 0.25


def compute_narrative_score(narrative_embedding_score: float) -> float:
    """
    Piecewise map from narrative_embedding_score ∈ {0.0, 0.333, 0.667, 1.0}.

    The score reflects how many of the three JD-required signal categories are
    evidenced in the candidate's career narrative (not their skills list):
      (a) embedding/vector/vectorDB keywords in narrative
      (b) NDCG/MRR/MAP/eval-framework keywords in narrative
      (c) has_ml_production_experience AND years_since_last_ml_role ≤ 1.0

    Mapping:
      0.0   → 0.00  (no JD signal categories evidenced)
      0.333 → 0.35  (one of three)
      0.667 → 0.70  (two of three)
      1.0   → 1.00  (all three)
    """
    if narrative_embedding_score <= 0.0:   return 0.0
    elif narrative_embedding_score <= 1/3: return 0.35
    elif narrative_embedding_score <= 2/3: return 0.70
    else:                                  return 1.0


def compute_location_fit(
    is_india_based: bool,
    is_target_city: bool,
    willing_to_relocate: bool,
    is_primary_city: bool = False,
) -> float:
    """
    Score location fit against Noida/Pune-preferred JD.

      Primary city (Pune/Noida): 1.00
      JD-welcomed India city:    0.88
      India, willing to relocate: 0.78
      India, other city:         0.52
      Abroad, willing to relocate: 0.55
      Abroad, not willing:       0.15
    """
    if not is_india_based:
        return 0.55 if willing_to_relocate else 0.15
    if is_primary_city:      return 1.0
    if is_target_city:       return 0.88
    if willing_to_relocate:  return 0.78
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
    Score company background and ML production experience.

    Hard disqualifiers (return 0.0):
      - Entire career in IT services (JD explicit)
      - Entire career in pure research with no production deployment

    shallow_recent_ml_only: recent LangChain-only experience without prior
    depth — treated as no effective ML experience.

    ML recency decay: stale ML always beats no-ML-ever (floor 0.65).
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
    """Score salary alignment with role budget (20–65 LPA). Soft signal only."""
    if expected_min <= SALARY_TARGET_MAX and expected_max >= SALARY_TARGET_MIN:
        return 1.0
    if expected_max < SALARY_TARGET_MIN * 0.7:
        return 0.55
    if expected_min > SALARY_TARGET_MAX * 1.5:
        return 0.35
    return 0.72


def compute_skill_assessment_bonus(skill_assessment_scores: dict) -> float:
    """
    Bonus for platform-verified skills in JD-relevant areas only.
    Sparse signal (~20% of candidates). Max bonus: 0.08.
    """
    if not skill_assessment_scores:
        return 0.0
    relevant = [
        float(score) / 100.0
        for skill_name, score in skill_assessment_scores.items()
        if any(jd_skill in skill_name.lower() for jd_skill in SKILL_ASSESSMENT_JD_RELEVANT)
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
    """Small tiebreaker for HR-tech / marketplace background (JD nice-to-have)."""
    industry = (current_industry or "").lower()
    return 0.02 if any(kw in industry for kw in INDUSTRY_RELEVANT_KEYWORDS) else 0.0


def compute_github_bonus(github_activity_score) -> float:
    """Small tiebreaker for open-source activity (JD nice-to-have)."""
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
    """
    Composite availability signal. All sub-scores in [0, 1].

    Notice weight raised to 0.18 (from 0.13) to better reflect the JD's
    explicit preference for sub-30-day notice and tightened tier scores.
    """
    # Recency
    if last_active_days_ago <= 14:    recency = 1.0
    elif last_active_days_ago <= 30:  recency = 0.85
    elif last_active_days_ago <= 90:  recency = 0.55
    elif last_active_days_ago <= 180: recency = 0.25
    else:                             recency = 0.08

    # Notice period
    if notice_period_days <= 15:    notice_score = NOTICE_SCORE_0_15
    elif notice_period_days <= 30:  notice_score = NOTICE_SCORE_16_30
    elif notice_period_days <= 60:  notice_score = NOTICE_SCORE_31_60
    elif notice_period_days <= 90:  notice_score = NOTICE_SCORE_61_90
    elif notice_period_days <= 120: notice_score = NOTICE_SCORE_91_120
    else:                           notice_score = NOTICE_SCORE_120P

    # Response time (-1 = no history → neutral 0.5)
    if avg_response_time_hours < 0:     response_time_score = 0.5
    elif avg_response_time_hours <= 4:  response_time_score = 1.0
    elif avg_response_time_hours <= 24: response_time_score = 0.75
    elif avg_response_time_hours <= 72: response_time_score = 0.40
    else:                               response_time_score = 0.10

    # Interview / offer history (-1 = no history → neutral 0.5)
    interview_score = 0.5 if interview_completion_rate < 0 else float(interview_completion_rate)
    offer_score     = 0.5 if offer_acceptance_rate < 0 else float(offer_acceptance_rate)

    social_proof = min(saved_by_recruiters_30d, 20) / 20.0
    reachability = 0.5 * float(verified_email) + 0.5 * float(verified_phone)

    availability = (
        0.22 * float(open_to_work) +
        0.18 * recency +
        0.14 * float(recruiter_response_rate) +
        0.18 * notice_score +
        0.08 * response_time_score +
        0.07 * interview_score +
        0.05 * offer_score +
        0.04 * social_proof +
        0.04 * reachability
    )
    return min(1.0, max(0.0, availability))
