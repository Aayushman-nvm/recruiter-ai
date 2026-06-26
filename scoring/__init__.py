"""
scoring/__init__.py — Re-exports everything for backward compatibility.

All scoring logic lives in:
  scoring/component_scores.py  — individual compute_*() functions
  scoring/structural.py        — compute_structural_score(), compute_final_score_with_cap()
  scoring/reasoning.py         — generate_reasoning()

Any existing `from scoring import X` calls keep working unchanged.
"""

from datetime import date

# ── Reference date ────────────────────────────────────────────────────────────
REFERENCE_DATE = date(2026, 5, 28)

# ── Config re-exports (backward compat) ──────────────────────────────────────
from config.locations import TARGET_CITIES, PRIMARY_CITIES
from config.companies import IT_SERVICES_COMPANIES
from config.keywords  import (
    SKILL_ASSESSMENT_JD_RELEVANT,
    RESEARCH_ONLY_COMPANY_INDICATORS,
    RESEARCH_ONLY_TITLE_INDICATORS,
    INDUSTRY_RELEVANT_KEYWORDS,
    ML_KEYWORDS,
    ML_KEYWORD_PATTERN,
    has_ml_keyword,
)
from config.seniority import (
    SENIORITY_HIGH, SENIORITY_SENIOR, SENIORITY_MID, SENIORITY_JUNIOR,
    get_title_seniority,
)
from config.salary import SALARY_TARGET_MIN, SALARY_TARGET_MAX

# ── Weight re-exports (backward compat) ──────────────────────────────────────
from config.weights import (
    NARRATIVE_SCORE_WEIGHT,
    FUSION_BM25_WEIGHT,
    FUSION_DENSE_WEIGHT,
    PRELIM_FUSION_WEIGHT,
    PRELIM_STRUCTURAL_WEIGHT,
    PRELIM_AVAILABILITY_WEIGHT,
    FINAL_PRELIM_WEIGHT,
    FINAL_CE_WEIGHT,
    SCORE_PENALTY_MULTIPLIER,
    SCORE_CAP_MAX,
    SCORE_CAP_EXP_FIT_FLOOR,
    SCORE_CAP_ML_RECENCY_YEARS,
    NOTICE_SCORE_0_15,
    NOTICE_SCORE_16_30,
    NOTICE_SCORE_31_60,
    NOTICE_SCORE_61_90,
    NOTICE_SCORE_91_120,
    NOTICE_SCORE_120P,
)

# ── Pipeline fusion re-exports (backward compat) ─────────────────────────────
from pipeline.fusion import weighted_score_fusion, rrf_fusion

# ── All scoring functions ─────────────────────────────────────────────────────
from scoring.component_scores import (
    compute_experience_fit,
    compute_narrative_score,
    compute_location_fit,
    compute_company_fit,
    compute_salary_fit,
    compute_skill_assessment_bonus,
    compute_education_bonus,
    compute_industry_bonus,
    compute_github_bonus,
    compute_availability_score,
)
from scoring.structural import (
    compute_structural_score,
    compute_final_score_with_cap,
)
from scoring.reasoning import generate_reasoning
