"""
Script 01 (v2): Full feature extraction from candidates.jsonl.gz
Output: precomputed/features.parquet

Adds over v1:
  - salary_min_lpa / salary_max_lpa  (from expected_salary_range_inr_lpa)
  - trajectory_score  (avg tenure, title seniority arc, ML recency)
  - skill_assessment_bonus  (verified platform scores vs JD-relevant skills)
  - edu_tier / edu_field / edu_bonus

Imports TARGET_CITIES, IT_SERVICES_COMPANIES, compute_skill_assessment_bonus,
compute_education_bonus from scoring.py — single authoritative source.
"""

import gzip
import json
import os
import re
from datetime import date

import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring import (
    TARGET_CITIES, IT_SERVICES_COMPANIES,
    compute_skill_assessment_bonus, compute_education_bonus,
    REFERENCE_DATE
)

# ── Title seniority mapping ─────────────────────────────────────────────────
# 5: principal/staff/head/vp/director  4: senior/lead  3: default  2: mid  1: junior
SENIORITY_HIGH   = {"principal", "staff engineer", "head of", "vp", "vice president",
                    "director", "distinguished", "fellow", "chief"}
SENIORITY_SENIOR = {"senior", "lead", "tech lead", "sr.", "sr ", "staff"}
SENIORITY_MID    = {"mid-level", "engineer ii", "sde ii", "swe ii"}
SENIORITY_JUNIOR = {"junior", "associate", "entry", "fresher", "intern", "trainee"}


def get_title_seniority(title: str) -> int:
    t = (title or "").lower()
    if any(k in t for k in SENIORITY_HIGH):   return 5
    if any(k in t for k in SENIORITY_SENIOR): return 4
    if any(k in t for k in SENIORITY_JUNIOR): return 1
    if any(k in t for k in SENIORITY_MID):    return 2
    return 3

ML_KEYWORDS = [
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "llm", "fine-tun", "rag", "semantic search", "sentence-transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "bert", "transformer", "nlp", "information retrieval", "learning to rank",
    "xgboost ranking", "neural ranker", "reranker", "dense retrieval",
    "search engine", "knowledge graph", "question answering", "vector database",
    "approximate nearest neighbor", "hnsw", "cosine similarity"
]
# Dropped bare "ann" — confirmed root cause of has_ml_production_experience being
# True on ~52% of completely unrelated titles (HR Manager, Accountant, Civil
# Engineer, ...). "ann" as a naive substring matched inside "channel", "planning",
# "announce", etc. The full phrase "approximate nearest neighbor" above already
# covers the legitimate case; "hnsw" covers the other common spelling.
#
# Matching now requires a leading word boundary (re.search(r"\bKEYWORD", desc))
# instead of plain `kw in desc`. A *leading* boundary (not a trailing one) was
# chosen deliberately: stems like "fine-tun" and "sentence-transformer" are meant
# to also match "fine-tuning"/"sentence-transformers", so they can't require a
# boundary at the end. A leading boundary alone is enough to stop "llm" matching
# inside "fulfillment" (no boundary before "llm" there) and "bert" matching inside
# "Robert"/"Albert" (no boundary before "bert" there), while still matching every
# legitimate case (a keyword preceded by whitespace, punctuation, or string start).
ML_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in ML_KEYWORDS) + r")"
)


def _has_ml_keyword(desc: str) -> bool:
    return ML_KEYWORD_PATTERN.search(desc) is not None

DATA_PATH = "data/candidates.jsonl.gz"
OUT_PATH  = "precomputed/features.parquet"


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def extract_features(c: dict) -> dict:
    p = c["profile"]
    career    = c.get("career_history", [])
    education = c.get("education", [])
    skills = c.get("skills", [])
    sig = c.get("redrob_signals", {})

    yoe = float(p.get("years_of_experience", 0) or 0)
    country = (p.get("country") or "").strip()
    location = (p.get("location") or "").strip()

    # ── Location ────────────────────────────────────────────────────────
    is_india_based = country.lower() == "india"
    is_target_city = any(city in location.lower() for city in TARGET_CITIES)
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))

    # ── Company type ─────────────────────────────────────────────────────
    n_total_roles = len(career)
    n_it_services_roles = sum(
        1 for r in career
        if any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
    )
    entire_career_it_services = (n_total_roles > 0 and n_it_services_roles == n_total_roles)
    has_product_company_exp = any(
        not any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
        for r in career
    )

    # ── ML production experience ─────────────────────────────────────────
    # Scan ALL career history roles (not just recent 6 years).
    # The 6-year cutoff was discarding valid ML roles that started slightly before
    # the window, even if they ran until recently. years_since_last_ml_role already
    # captures recency — we don't need a hard cutoff here.
    has_ml_production_experience = False
    # Sentinel 99.0 = "never did ML" — distinct from yoe and handled explicitly
    # in compute_company_fit (ml_recency branch) and generate_reasoning (has_ml guard)
    years_since_last_ml_role = 99.0

    for role in career:
        desc = (role.get("description") or "").lower()
        if _has_ml_keyword(desc):
            has_ml_production_experience = True
            end_raw = role.get("end_date")
            end = parse_date(end_raw) if end_raw else REFERENCE_DATE
            yrs_ago = max(0.0, (REFERENCE_DATE - (end or REFERENCE_DATE)).days / 365.25)
            years_since_last_ml_role = min(years_since_last_ml_role, yrs_ago)

    # ── Salary features ───────────────────────────────────────────────────
    sal = (sig.get("expected_salary_range_inr_lpa") or {})
    salary_min_lpa = float(sal.get("min", 0) or 0)
    salary_max_lpa = float(sal.get("max", 999) or 999)

    # ── Skill assessment bonus ────────────────────────────────────────────
    skill_assessment_scores = sig.get("skill_assessment_scores") or {}
    skill_assessment_bonus  = compute_skill_assessment_bonus(skill_assessment_scores)

    # ── Education features ────────────────────────────────────────────────
    if education:
        edu_tier  = education[0].get("tier", "unknown") or "unknown"
        edu_field = education[0].get("field_of_study", "") or ""
    else:
        edu_tier  = "unknown"
        edu_field = ""
    edu_bonus = compute_education_bonus(edu_tier, edu_field)

    # ── Career trajectory ─────────────────────────────────────────────────
    durations       = [r.get("duration_months", 0) or 0 for r in career]
    avg_tenure_months = float(sum(durations) / len(durations)) if durations else 0.0

    title_now   = get_title_seniority(p.get("current_title", ""))
    title_first = get_title_seniority(career[-1].get("title", "") if career else "")
    trajectory_upward = title_now > title_first

    tenure_stability = 1.0 if avg_tenure_months > 24 else (0.7 if avg_tenure_months > 12 else 0.3)
    ml_recency_score = (1.0 if years_since_last_ml_role <= 0
                        else (0.5 if years_since_last_ml_role <= 2
                              else (0.2 if years_since_last_ml_role <= 4 else 0.0)))
    trajectory_score = (
        0.4 * tenure_stability +
        0.3 * float(trajectory_upward) +
        0.3 * ml_recency_score
    )

    # ── Availability signals (raw) ────────────────────────────────────────
    open_to_work_flag = bool(sig.get("open_to_work_flag", False))
    recruiter_response_rate = float(sig.get("recruiter_response_rate", 0.0) or 0.0)
    avg_response_time_hours = float(sig.get("avg_response_time_hours") if sig.get("avg_response_time_hours") is not None else -1)
    notice_period_days = int(sig.get("notice_period_days", 90) or 90)
    interview_completion_rate = float(sig.get("interview_completion_rate", 0.0) or 0.0)
    offer_acceptance_rate = float(sig.get("offer_acceptance_rate") if sig.get("offer_acceptance_rate") is not None else -1)
    saved_by_recruiters_30d = int(sig.get("saved_by_recruiters_30d", 0) or 0)
    verified_email = bool(sig.get("verified_email", False))
    verified_phone = bool(sig.get("verified_phone", False))
    github_activity_score = float(sig.get("github_activity_score") if sig.get("github_activity_score") is not None else -1)
    preferred_work_mode = sig.get("preferred_work_mode", "") or ""

    # last_active_days_ago: DERIVED — raw field is last_active_date (date string)
    # Never use sig.get("last_active_days_ago") — that key does not exist in raw data
    try:
        last_active = date.fromisoformat(str(sig.get("last_active_date", ""))[:10])
        last_active_days_ago = (REFERENCE_DATE - last_active).days
    except Exception:
        last_active_days_ago = 999

    # ── Honeypot detection ────────────────────────────────────────────────
    is_honeypot = False

    # Condition 1: expert skill with duration_months == 0
    if not is_honeypot:
        for s in skills:
            if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0:
                is_honeypot = True
                break

    # Condition 2: expert skill total duration > yoe * 12 * 1.6
    if not is_honeypot:
        expert_total = sum(s.get("duration_months", 0) or 0 for s in skills if s.get("proficiency") == "expert")
        if yoe > 0 and expert_total > yoe * 12 * 1.6:
            is_honeypot = True

    # Condition 3: impossible role dates (start > end)
    if not is_honeypot:
        for role in career:
            s = parse_date(role.get("start_date"))
            e = parse_date(role.get("end_date"))
            if s and e and s > e:
                is_honeypot = True
                break

    # Condition 4: career duration impossibility (total months > yoe * 15)
    if not is_honeypot:
        total_months = sum(r.get("duration_months", 0) or 0 for r in career)
        if yoe > 0 and total_months > yoe * 15:
            is_honeypot = True

    # Condition 5: sanity — years_of_experience > 40
    if not is_honeypot and yoe > 40:
        is_honeypot = True

    # Condition 6: synthetic inflation (perfect profile + extreme saves)
    if not is_honeypot:
        completeness = float(sig.get("profile_completeness_score", 0) or 0)
        linkedin = bool(sig.get("linkedin_connected", False))
        rr = float(sig.get("recruiter_response_rate", 0) or 0)
        if (completeness == 100 and verified_email and verified_phone
                and linkedin and saved_by_recruiters_30d > 50 and rr == 1.0):
            is_honeypot = True

    return {
        "candidate_id":               c["candidate_id"],
        "years_of_experience":         yoe,
        "country":                     country,
        "location":                    location,
        "current_title":               p.get("current_title", ""),
        "current_company":             p.get("current_company", ""),
        "current_industry":            p.get("current_industry", ""),
        # Location
        "is_india_based":              is_india_based,
        "is_target_city":              is_target_city,
        "willing_to_relocate":         willing_to_relocate,
        # Company type
        "n_total_roles":               n_total_roles,
        "n_it_services_roles":         n_it_services_roles,
        "entire_career_it_services":   entire_career_it_services,
        "has_product_company_exp":     has_product_company_exp,
        # ML experience
        "has_ml_production_experience": has_ml_production_experience,
        "years_since_last_ml_role":    years_since_last_ml_role,
        # Trajectory (NEW in v2)
        "avg_tenure_months":           avg_tenure_months,
        "trajectory_upward":           trajectory_upward,
        "trajectory_score":            trajectory_score,
        # Salary (NEW in v2)
        "salary_min_lpa":              salary_min_lpa,
        "salary_max_lpa":              salary_max_lpa,
        # Skill assessment (NEW in v2)
        "skill_assessment_bonus":      skill_assessment_bonus,
        # Education (NEW in v2)
        "edu_tier":                    edu_tier,
        "edu_field":                   edu_field,
        "edu_bonus":                   edu_bonus,
        # Availability
        "open_to_work_flag":           open_to_work_flag,
        "last_active_days_ago":        last_active_days_ago,
        "recruiter_response_rate":     recruiter_response_rate,
        "avg_response_time_hours":     avg_response_time_hours,
        "notice_period_days":          notice_period_days,
        "interview_completion_rate":   interview_completion_rate,
        "offer_acceptance_rate":       offer_acceptance_rate,
        "saved_by_recruiters_30d":     saved_by_recruiters_30d,
        "verified_email":              verified_email,
        "verified_phone":              verified_phone,
        "github_activity_score":       github_activity_score,
        "preferred_work_mode":         preferred_work_mode,
        "is_honeypot":                 is_honeypot,
    }


def main():
    os.makedirs("precomputed", exist_ok=True)
    rows = []
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting features"):
            line = line.strip()
            if not line:
                continue
            rows.append(extract_features(json.loads(line)))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUT_PATH}")
    print(f"Honeypots detected: {df['is_honeypot'].sum()}")
    print(f"IT-services-only: {df['entire_career_it_services'].sum()}")


if __name__ == "__main__":
    main()