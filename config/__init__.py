"""
config/__init__.py — Re-export everything so existing
`from scoring import TARGET_CITIES, ...` style imports keep working
while new code can do `from config import TARGET_CITIES`.
"""
from config.locations import TARGET_CITIES, PRIMARY_CITIES, TIER_1, TIER_2, TIER_3
from config.companies import IT_SERVICES_COMPANIES
from config.keywords import (
    JD_RELEVANT_SKILLS,
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

__all__ = [
    "TARGET_CITIES", "PRIMARY_CITIES", "TIER_1", "TIER_2", "TIER_3",
    "IT_SERVICES_COMPANIES",
    "JD_RELEVANT_SKILLS",
    "RESEARCH_ONLY_COMPANY_INDICATORS",
    "RESEARCH_ONLY_TITLE_INDICATORS",
    "INDUSTRY_RELEVANT_KEYWORDS",
    "ML_KEYWORDS",
    "SENIORITY_HIGH", "SENIORITY_SENIOR", "SENIORITY_MID", "SENIORITY_JUNIOR",
    "get_title_seniority",
    "SALARY_TARGET_MIN", "SALARY_TARGET_MAX",
]
