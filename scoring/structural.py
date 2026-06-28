from config.weights import (
    NARRATIVE_SCORE_WEIGHT,
    COMPANY_FIT_WEIGHT,
    LOCATION_FIT_WEIGHT,
    EXPERIENCE_FIT_WEIGHT,
    TRAJECTORY_SCORE_WEIGHT,
    SALARY_FIT_WEIGHT,
    FINAL_PRELIM_WEIGHT,
    FINAL_CE_WEIGHT,
    SCORE_PENALTY_MULTIPLIER,
    SCORE_CAP_MAX,
    SCORE_CAP_EXP_FIT_FLOOR,
    SCORE_CAP_ML_RECENCY_YEARS,
)
from scoring.component_scores import compute_narrative_score


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
    narrative_embedding_score: float = 0.0,
    has_disqualifying_language: bool = False,
    is_ghost_skill_candidate: bool = False,
    is_cv_speech_no_nlp: bool = False,
    notice_period_days: int = 0,
    recruiter_response_rate: float = 0.5,
    is_junior_stagnant: bool = False,
) -> float:
    narrative_score = compute_narrative_score(narrative_embedding_score)
    raw = (
        NARRATIVE_SCORE_WEIGHT * narrative_score +
        COMPANY_FIT_WEIGHT * company_fit +
        LOCATION_FIT_WEIGHT * location_fit +
        EXPERIENCE_FIT_WEIGHT * experience_fit +
        TRAJECTORY_SCORE_WEIGHT * trajectory_score +
        SALARY_FIT_WEIGHT * salary_fit
    )

    # Graduated IT-services penalty — having some IT-services roles is common
    # in India; what matters is whether the whole career is services.
    if n_it_services_roles >= 5:   it_penalty = -0.07
    elif n_it_services_roles >= 3: it_penalty = -0.05
    elif n_it_services_roles >= 1: it_penalty = -0.02
    else:                          it_penalty = 0.0

    # Job-hop penalty — JD explicitly flags title-chasing (avg <1.5yr/role)
    if job_hop_score <= 0:      hop_penalty = 0.0
    elif job_hop_score < 1.0:   hop_penalty = -0.08   # very aggressive hopping
    elif job_hop_score < 1.5:   hop_penalty = -0.05
    elif job_hop_score < 2.0:   hop_penalty = -0.02
    else:                       hop_penalty = 0.0

    penalty_multiplier = 1.0

    # Zero-narrative means no JD-relevant terms appear in actual role descriptions.
    # This is the primary signal of keyword stuffing / irrelevant background.
    if narrative_embedding_score == 0.0:
        penalty_multiplier *= 0.65   # tightened from 0.85 — no narrative = very low confidence

    # Disqualifying language in narrative (explicit non-ownership of deployment)
    if has_disqualifying_language:
        penalty_multiplier *= 0.25   # tightened from 0.30

    # Ghost skills: tech keywords in skills list that appear nowhere in narrative.
    # The JD explicitly warns: "right answer is NOT the candidate whose skills
    # section contains the most AI keywords."
    if is_ghost_skill_candidate:
        penalty_multiplier *= 0.38   # tightened from 0.45

    # CV/speech primary without NLP crossover — JD: "re-learning fundamentals"
    if is_cv_speech_no_nlp:
        penalty_multiplier *= 0.45   # tightened from 0.55

    # Junior stagnation — founding team needs senior judgment
    if is_junior_stagnant:
        penalty_multiplier *= 0.60   # tightened from 0.65

    # Low recruiter response rate — JD: "low response rate is a strong negative signal"
    if recruiter_response_rate < 0.20:
        penalty_multiplier *= 0.62   # tightened from 0.70
    elif recruiter_response_rate < 0.40:
        penalty_multiplier *= 0.80   # tightened from 0.85
    elif recruiter_response_rate < 0.60:
        penalty_multiplier *= 0.92

    # Long notice period — JD: "30+ day notice, the bar gets higher"
    if notice_period_days > 120:
        penalty_multiplier *= 0.80   # tightened from 0.86
    elif notice_period_days > 90:
        penalty_multiplier *= 0.86   # tightened from 0.91
    elif notice_period_days > 60:
        penalty_multiplier *= 0.92   # tightened from 0.95

    bonuses = skill_assessment_bonus + edu_bonus + industry_bonus + github_bonus

    base = min(1.0, max(0.0,
        raw + bonuses + it_penalty + hop_penalty
    ))

    # Apply multiplier penalties
    score = min(1.0, max(0.0, base * penalty_multiplier))

    # Hard ceiling: ghost-skill candidate with no narrative evidence whatsoever
    # These are keyword-stuffers — cap them below any legitimate candidate.
    if is_ghost_skill_candidate and narrative_embedding_score == 0.0:
        score = min(score, 0.15)   # tightened from 0.20

    # Hard ceiling: zero-narrative candidates are capped regardless of skills list.
    # This directly prevents the HR Manager / Mechanical Engineer type from scoring high.
    if narrative_embedding_score == 0.0:
        score = min(score, 0.28)   # NEW: explicit cap for zero-narrative candidates

    return score


def compute_final_score_with_cap(
    prelim_score: float,
    ce_score: float,
    experience_fit: float,
    years_since_last_ml_role: float,
) -> float:
    """
    Blend prelim + cross-encoder score with a penalty+cap for structurally weak candidates.

    The penalty fires when:
      - experience_fit < SCORE_CAP_EXP_FIT_FLOOR (outside 5–9yr ideal band by enough)
      - years_since_last_ml_role > SCORE_CAP_ML_RECENCY_YEARS (stale ML)

    Both conditions reflect the JD's core disqualifiers and should prevent
    weak candidates from being elevated to top slots by a strong CE score alone.
    """
    raw = FINAL_PRELIM_WEIGHT * prelim_score + FINAL_CE_WEIGHT * ce_score
    needs_penalty = (
        experience_fit < SCORE_CAP_EXP_FIT_FLOOR
        or years_since_last_ml_role > SCORE_CAP_ML_RECENCY_YEARS
    )
    if needs_penalty:
        return min(raw * SCORE_PENALTY_MULTIPLIER, SCORE_CAP_MAX)
    return raw