"""
scoring/structural.py — Structural score and final score blending.

compute_structural_score(): combines all component scores + penalties into
  a single structural signal.

compute_final_score_with_cap(): blends prelim + CE scores, applies penalty
  and ceiling for over-experience / stale ML candidates.
"""

from config.weights import (
    NARRATIVE_SCORE_WEIGHT,
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
) -> float:
    """
    Weighted structural score. Clamped to [0.0, 1.0].

    Base weights (sum = 1.09 — intentionally slightly over 1.0 to allow
    bonuses to push strong candidates above 1.0 before clamping):
      narrative_score  0.35  (NARRATIVE_SCORE_WEIGHT)
      location_fit     0.25
      experience_fit   0.20
      company_fit      0.22
      trajectory_score 0.05
      salary_fit       0.02

    Penalties applied after weighted sum:
      narrative_zero   −0.06  (narrative_embedding_score == 0.0)
      disqualifying    −0.12  (has_disqualifying_language)
      ghost_skill      −0.25  (is_ghost_skill_candidate)
      cv_speech        −0.20  (is_cv_speech_no_nlp)
      it_services      −0.02 / −0.04 / −0.06  (graduated by count)
      job_hop          −0.05 / −0.02           (graduated by years/role)

    Hard ceiling: when is_ghost_skill_candidate AND narrative_embedding_score == 0.0,
      score is capped at 0.30 after all penalties.
    """
    narrative_score = compute_narrative_score(narrative_embedding_score)

    raw = (
        NARRATIVE_SCORE_WEIGHT * narrative_score +
        0.22 * company_fit +
        0.25 * location_fit +
        0.20 * experience_fit +
        0.05 * trajectory_score +
        0.02 * salary_fit
    )

    # Graduated IT-services penalty
    if n_it_services_roles >= 5:   it_penalty = -0.06
    elif n_it_services_roles >= 3: it_penalty = -0.04
    elif n_it_services_roles >= 1: it_penalty = -0.02
    else:                          it_penalty = 0.0

    # Job-hop penalty
    if job_hop_score <= 0:    hop_penalty = 0.0
    elif job_hop_score < 1.5: hop_penalty = -0.05
    elif job_hop_score < 2.5: hop_penalty = -0.02
    else:                     hop_penalty = 0.0

    # Narrative / disqualifying penalties
    # narrative_zero at -0.06: avoids double-dipping with disqualifying_penalty;
    # the 0.35-weight narrative_score = 0.0 already captures the bulk of the signal.
    narrative_zero_penalty = -0.06 if narrative_embedding_score == 0.0 else 0.0
    # disqualifying at -0.12: reduced from -0.18 to prevent collapsing all flagged
    # candidates to the same floor, which was destroying differentiation from retrieval.
    disqualifying_penalty  = -0.12 if has_disqualifying_language else 0.0
    ghost_penalty          = -0.25 if is_ghost_skill_candidate else 0.0
    cv_speech_penalty      = -0.20 if is_cv_speech_no_nlp else 0.0

    bonuses = skill_assessment_bonus + edu_bonus + industry_bonus + github_bonus

    score = min(1.0, max(0.0,
        raw + bonuses + it_penalty + hop_penalty
        + narrative_zero_penalty + disqualifying_penalty
        + ghost_penalty + cv_speech_penalty
    ))

    # Hard ceiling: ghost-skill candidate with no narrative evidence
    if is_ghost_skill_candidate and narrative_embedding_score == 0.0:
        score = min(score, 0.30)

    return score


def compute_final_score_with_cap(
    prelim_score: float,
    ce_score: float,
    experience_fit: float,
    years_since_last_ml_role: float,
) -> float:
    """
    Blend prelim + CE scores, then apply penalty + ceiling for disqualified cases.

    WHY PENALTY THEN CAP (not just cap):
      A pure cap collapses all breaching candidates to the same ceiling,
      losing relative ordering within the penalised group. Multiplying first
      preserves separation, then the cap enforces the structural ceiling.

    Triggers (either is sufficient):
      1. experience_fit < SCORE_CAP_EXP_FIT_FLOOR (> 15 yrs total)
      2. years_since_last_ml_role > SCORE_CAP_ML_RECENCY_YEARS (stale ML)
    """
    raw = FINAL_PRELIM_WEIGHT * prelim_score + FINAL_CE_WEIGHT * ce_score
    needs_penalty = (
        experience_fit < SCORE_CAP_EXP_FIT_FLOOR
        or years_since_last_ml_role > SCORE_CAP_ML_RECENCY_YEARS
    )
    if needs_penalty:
        return min(raw * SCORE_PENALTY_MULTIPLIER, SCORE_CAP_MAX)
    return raw
