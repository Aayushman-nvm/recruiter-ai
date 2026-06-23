"""
config/seniority.py — Title seniority constants and lookup.

Used by: scoring.py, pipeline/feature_extraction.py
Single source of truth — previously duplicated between scoring.py and
04_eval.py where the local copy silently dropped SENIORITY_MID.

Levels:
  5: principal / staff / head / vp / director
  4: senior / lead / tech lead
  3: default (no keyword matched)
  2: mid-level (engineer II, SDE II)
  1: junior / associate / intern
"""

SENIORITY_HIGH = {
    "principal", "staff engineer", "head of", "vp", "vice president",
    "director", "distinguished", "fellow", "chief",
}
SENIORITY_SENIOR = {"senior", "lead", "tech lead", "sr.", "sr ", "staff"}
SENIORITY_MID    = {"mid-level", "engineer ii", "sde ii", "swe ii"}
SENIORITY_JUNIOR = {"junior", "associate", "entry", "fresher", "intern", "trainee"}


def get_title_seniority(title: str) -> int:
    """Return an integer seniority level (1–5) for the given job title."""
    t = (title or "").lower()
    if any(k in t for k in SENIORITY_HIGH):   return 5
    if any(k in t for k in SENIORITY_SENIOR): return 4
    if any(k in t for k in SENIORITY_JUNIOR): return 1
    if any(k in t for k in SENIORITY_MID):    return 2
    return 3
