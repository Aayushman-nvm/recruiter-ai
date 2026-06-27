"""
config/__init__.py — Re-export everything so existing
`from scoring import TARGET_CITIES, ...` style imports keep working
while new code can do `from config import TARGET_CITIES`.

weights.py is the canonical home for all pipeline blend weights and score caps.
Any module that needs them should import from config.weights directly to avoid
circular imports (scoring.py → pipeline/fusion.py → scoring.py was the old cycle).
"""
from config.locations import TARGET_CITIES, PRIMARY_CITIES, TIER_1, TIER_2, TIER_3
from config.companies import IT_SERVICES_COMPANIES
from config.keywords import (
    SKILL_ASSESSMENT_JD_RELEVANT,
    RESEARCH_ONLY_COMPANY_INDICATORS,
    RESEARCH_ONLY_TITLE_INDICATORS,
    INDUSTRY_RELEVANT_KEYWORDS,
    ML_KEYWORDS,
)
from config.seniority import (
    SENIORITY_HIGH,
    SENIORITY_SENIOR,
    SENIORITY_MID,
    SENIORITY_JUNIOR,
    get_title_seniority,
)
from config.salary import SALARY_TARGET_MIN, SALARY_TARGET_MAX
from config.weights import (
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

__all__ = [
    # locations
    "TARGET_CITIES", "PRIMARY_CITIES", "TIER_1", "TIER_2", "TIER_3",
    # companies
    "IT_SERVICES_COMPANIES",
    # keywords
    "SKILL_ASSESSMENT_JD_RELEVANT",
    "RESEARCH_ONLY_COMPANY_INDICATORS",
    "RESEARCH_ONLY_TITLE_INDICATORS",
    "INDUSTRY_RELEVANT_KEYWORDS",
    "ML_KEYWORDS",
    # seniority
    "SENIORITY_HIGH", "SENIORITY_SENIOR", "SENIORITY_MID", "SENIORITY_JUNIOR",
    "get_title_seniority",
    # salary
    "SALARY_TARGET_MIN", "SALARY_TARGET_MAX",
    # weights & caps
    "FUSION_BM25_WEIGHT", "FUSION_DENSE_WEIGHT",
    "PRELIM_FUSION_WEIGHT", "PRELIM_STRUCTURAL_WEIGHT", "PRELIM_AVAILABILITY_WEIGHT",
    "FINAL_PRELIM_WEIGHT", "FINAL_CE_WEIGHT",
    "SCORE_PENALTY_MULTIPLIER", "SCORE_CAP_MAX", "SCORE_CAP_EXP_FIT_FLOOR", "SCORE_CAP_ML_RECENCY_YEARS",
    "NOTICE_SCORE_0_15", "NOTICE_SCORE_16_30", "NOTICE_SCORE_31_60",
    "NOTICE_SCORE_61_90", "NOTICE_SCORE_91_120", "NOTICE_SCORE_120P",
]
