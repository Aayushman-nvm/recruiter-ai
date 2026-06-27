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
    notice_period_days: int = 0,
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

    Multiplier penalties applied after weighted sum (stack multiplicatively):
      narrative_zero      ×0.85  (narrative_embedding_score == 0.0)
      disqualifying       ×0.30  (has_disqualifying_language)
      ghost_skill         ×0.45  (is_ghost_skill_candidate)
      cv_speech           ×0.55  (is_cv_speech_no_nlp)
      notice 61–90d       ×0.98  — light tiebreaker (was ×0.88; softened, see inline note)
      notice 91–120d      ×0.95  — light tiebreaker (was ×0.75; softened, see inline note)
      notice >120d        ×0.90  — light tiebreaker (was ×0.60; softened, see inline note)

    Additive penalties after weighted sum (before multipliers):
      it_services      −0.02 / −0.04 / −0.06  (graduated by count)
      job_hop          −0.05 / −0.02           (graduated by years/role)

    Hard ceiling: when is_ghost_skill_candidate AND narrative_embedding_score == 0.0,
      score is capped at 0.20 after all penalties.
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

    # Narrative / disqualifying penalties — load-bearing multipliers, not cosmetic deductions.
    # Flat deductions collapsed everyone to the same floor; multipliers preserve relative
    # ordering within the penalised group while making the penalty actually matter.
    #
    # narrative_zero: ×0.85 — zero narrative evidence is already captured by
    #   narrative_score = 0.0 (contributing 0 to the 0.35-weighted term). This
    #   small additional multiplier discourages near-zero narrative further.
    # disqualifying: ×0.30 — explicit non-ownership admission is near-disqualifying
    #   per the JD. Was -0.12 flat, then ×0.50. At ×0.50 + the OLD prelim fusion
    #   weight (0.53), a keyword/semantic-strong candidate with this flag could
    #   still rank #1 (observed in submission_v15: CAND_0004402, CAND_0043860).
    #   Tightened to ×0.30 so the penalty holds even after PRELIM_STRUCTURAL_WEIGHT
    #   was raised — this is now the primary defense against keyword-stuffed
    #   profiles with a non-ownership admission; the rank.py hard-zero gate
    #   (narrative_embedding_score < 0.667) is the backstop for clear-cut cases.
    # ghost_skill: ×0.45 — 3+ JD skills with no narrative support is keyword stuffing.
    # cv_speech: ×0.55 — CV/speech-without-NLP is an explicit JD "do NOT want".

    penalty_multiplier = 1.0
    if narrative_embedding_score == 0.0:
        penalty_multiplier *= 0.85
    if has_disqualifying_language:
        penalty_multiplier *= 0.30
    if is_ghost_skill_candidate:
        penalty_multiplier *= 0.45
    if is_cv_speech_no_nlp:
        penalty_multiplier *= 0.55

    # Notice period structural penalty — JD: "bar gets higher" beyond 30 days.
    # SOFTENED — this was double-penalising notice period: it's already a
    # 0.18-weighted sub-score inside compute_availability_score, and this
    # multiplier stacked a SECOND penalty on top of the entire structural score
    # (which carries narrative/company/experience fit — the signals that should
    # dominate per "skills/fit > notice > location"). Empirically, notice period
    # barely separated good candidates in this pool (most strong matches sit in
    # the 60-120 day range simply because few people have <30-day notice), so a
    # ×0.60–0.75 multiplier here was punishing fit-strength candidates for a
    # weak, low-information signal. Reduced to a light tiebreaker rather than a
    # second hard penalty; availability's notice sub-score still carries the
    # bulk of notice's influence.
    if notice_period_days > 120:
        penalty_multiplier *= 0.90
    elif notice_period_days > 90:
        penalty_multiplier *= 0.95
    elif notice_period_days > 60:
        penalty_multiplier *= 0.98

    bonuses = skill_assessment_bonus + edu_bonus + industry_bonus + github_bonus

    base = min(1.0, max(0.0,
        raw + bonuses + it_penalty + hop_penalty
    ))

    # Apply multiplier penalties — load-bearing, not cosmetic
    score = min(1.0, max(0.0, base * penalty_multiplier))

    # Hard ceiling: ghost-skill candidate with no narrative evidence
    if is_ghost_skill_candidate and narrative_embedding_score == 0.0:
        score = min(score, 0.20)   # tightened from 0.30

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