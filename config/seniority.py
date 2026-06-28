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
