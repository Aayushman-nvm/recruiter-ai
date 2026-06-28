"""
pipeline/feature_extraction.py — Shared feature extraction logic.

Previously duplicated between scripts/01_extract_features.py (parquet bulk
extraction) and scripts/04_eval.py (inline per-candidate extraction for eval).
Any divergence between the two was silent and corrupted the ablation table.

This module is the single authoritative implementation. Both scripts import
from here — if you change the logic, it propagates to both automatically.

Public API:
  is_research_only_role(role)   — True if a career role is purely academic
  extract_features(c)           — Full feature dict for one candidate
  parse_date(s)                 — ISO date string → date or None

Feature columns produced (in addition to legacy columns):
  narrative_text                (str)   — concatenated career_history[].description
  narrative_embedding_score     (float) — fraction of JD signal categories evidenced {0.0, 0.333, 0.667, 1.0}
  has_disqualifying_language    (bool)  — explicit non-ownership/delegation phrase in narrative
  n_ghost_skills                (int)   — count of GHOST_SKILL_KEYWORDS in skills list absent from narrative
  is_ghost_skill_candidate      (bool)  — n_ghost_skills >= 3 AND narrative_embedding_score < 0.667
  is_cv_speech_primary          (bool)  — current title or >50% career roles match CV/speech domain titles
  is_cv_speech_no_nlp           (bool)  — is_cv_speech_primary AND distinct NLP/IR crossover keywords < 2
"""

from datetime import date
from pathlib import Path
import sys

# Ensure project root is on path when this module is imported directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.keywords import (
    RESEARCH_ONLY_COMPANY_INDICATORS,
    RESEARCH_ONLY_TITLE_INDICATORS,
    has_ml_keyword,
    DISQUALIFYING_PHRASES,
    GHOST_SKILL_KEYWORDS,
    CV_SPEECH_DOMAIN_TITLES,
    NLP_IR_CROSSOVER_KEYWORDS,
)
from config.companies import IT_SERVICES_COMPANIES
from config.locations import TARGET_CITIES, PRIMARY_CITIES, TIER_1
from config.seniority import get_title_seniority
from scoring import (
    REFERENCE_DATE,
    compute_skill_assessment_bonus,
    compute_education_bonus,
)


def parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def is_research_only_role(role: dict) -> bool:
    """
    True if a single career_history role looks like a pure academic /
    research role.

    Deliberately narrow — only unambiguous academic-institution names and
    research-specific titles. Does NOT flag corporate research labs
    ("Google Research", "Microsoft Research") because those sit inside
    real product companies that also ship production systems.
    """
    company = (role.get("company") or "").lower()
    title   = (role.get("title")   or "").lower()
    if any(ind in company for ind in RESEARCH_ONLY_COMPANY_INDICATORS):
        return True
    if any(ind in title for ind in RESEARCH_ONLY_TITLE_INDICATORS):
        return True
    return False


def extract_features(c: dict, detect_honeypot: bool = True) -> dict:
    """
    Extract all ranking features from a single raw candidate dict.

    Parameters
    ----------
    c : dict
        Raw candidate record (from candidates.jsonl.gz or sample_candidates.json).
    detect_honeypot : bool
        Run honeypot detection (default True). Set False for the 50-candidate
        eval set where it isn't needed and adds noise.

    Returns
    -------
    dict with all feature columns expected by scoring.py and rank.py.
    """
    p      = c["profile"]
    career = c.get("career_history", [])
    edu    = c.get("education", [])
    skills = c.get("skills", [])
    sig    = c.get("redrob_signals", {})

    yoe      = float(p.get("years_of_experience", 0) or 0)
    country  = (p.get("country")  or "").strip()
    location = (p.get("location") or "").strip()

    # ── Location ──────────────────────────────────────────────────────────────
    is_india_based      = country.lower() == "india"
    is_target_city      = any(city in location.lower() for city in TARGET_CITIES)
    is_primary_city     = any(city in location.lower() for city in PRIMARY_CITIES)
    is_tier_1_city      = any(city in location.lower() for city in TIER_1)
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))

    # ── Company type ──────────────────────────────────────────────────────────
    n_total_roles = len(career)
    n_it_services_roles = sum(
        1 for r in career
        if any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
    )
    entire_career_it_services = (n_total_roles > 0 and n_it_services_roles == n_total_roles)

    # jd.txt hard disqualifier: pure research/academic career with no production
    # deployment. Mirrors the IT-services check structurally.
    n_research_only_roles   = sum(1 for r in career if is_research_only_role(r))
    entire_career_research_only = (n_total_roles > 0 and n_research_only_roles == n_total_roles)

    # has_product_company_exp excludes BOTH IT-services and research-only roles.
    # Previously it only excluded IT-services, so an all-academia candidate
    # incorrectly satisfied this flag.
    has_product_company_exp = any(
        not any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
        and not is_research_only_role(r)
        for r in career
    )

    # ── ML production experience ──────────────────────────────────────────────
    # Scan ALL career history — no 6-year cutoff. years_since_last_ml_role
    # captures recency; a hard cutoff discarded valid roles just outside the window.
    has_ml_production_experience = False
    years_since_last_ml_role     = 99.0   # sentinel: "never did ML"
    n_ml_roles                   = 0
    total_ml_months               = 0

    for role in career:
        desc = (role.get("description") or "").lower()
        if has_ml_keyword(desc):
            has_ml_production_experience = True
            n_ml_roles   += 1
            total_ml_months += (role.get("duration_months", 0) or 0)
            end_raw = role.get("end_date")
            end     = parse_date(end_raw) if end_raw else REFERENCE_DATE
            yrs_ago = max(0.0, (REFERENCE_DATE - (end or REFERENCE_DATE)).days / 365.25)
            years_since_last_ml_role = min(years_since_last_ml_role, yrs_ago)

    # jd.txt: "If your 'AI experience' consists primarily of recent (under 12
    # months) projects using LangChain to call OpenAI — we will probably not
    # move forward." Proxy: only ONE detected ML role, recent, and short.
    shallow_recent_ml_only = (
        has_ml_production_experience
        and n_ml_roles <= 1
        and years_since_last_ml_role <= 1.0
        and total_ml_months < 12
    )

    # ── Narrative evidence (R1) ───────────────────────────────────────────────
    narrative_text = " ".join(
        (role.get("description") or "")
        for role in career
    ).strip()
    narrative_lower = narrative_text.lower()

    # Category (a): embedding/vector/vectorDB keywords in narrative
    CAT_A_TERMS = {
        "embedding", "vector", "vector database", "faiss", "pinecone",
        "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
        "semantic search", "hybrid search", "hybrid retrieval", "dense retrieval",
    }
    # Category (b): eval framework keywords in narrative
    CAT_B_TERMS = {
        "ndcg", "mrr", "map", "a/b test", "evaluation framework",
        "learning to rank",
    }

    cat_a = any(term in narrative_lower for term in CAT_A_TERMS)
    cat_b = any(term in narrative_lower for term in CAT_B_TERMS)
    cat_c = has_ml_production_experience and years_since_last_ml_role <= 1.0

    narrative_embedding_score = (int(cat_a) + int(cat_b) + int(cat_c)) / 3.0

    # ── Disqualifying language (R2) ───────────────────────────────────────────
    has_disqualifying_language = any(
        phrase in narrative_lower for phrase in DISQUALIFYING_PHRASES
    )

    # ── Ghost skills (R3) ────────────────────────────────────────────────────
    skill_names_lower = {(s.get("name") or "").lower() for s in skills}
    ghost_skills_found = {
        kw for kw in GHOST_SKILL_KEYWORDS
        if kw in skill_names_lower and kw not in narrative_lower
    }
    n_ghost_skills = len(ghost_skills_found)
    is_ghost_skill_candidate = (
        n_ghost_skills >= 3 and narrative_embedding_score < 0.667
    )

    # ── Top JD-relevant skills (for reasoning sentence 1) ────────────────────
    # Pull up to 4 skills that appear in SKILL_ASSESSMENT_JD_RELEVANT,
    # sorted by endorsements descending so the most credible ones surface first.
    from config.keywords import SKILL_ASSESSMENT_JD_RELEVANT as _JD_REL
    skills_sorted = sorted(skills, key=lambda s: s.get("endorsements", 0) or 0, reverse=True)
    top_jd_skills = ", ".join(
        s["name"] for s in skills_sorted
        if any(jd_kw in (s.get("name") or "").lower() for jd_kw in _JD_REL)
    )[:120]  # cap string length

    # ── CV/speech domain (R5) ────────────────────────────────────────────────
    current_title_lower = (p.get("current_title") or "").lower()
    is_cv_speech_primary = any(term in current_title_lower for term in CV_SPEECH_DOMAIN_TITLES)

    if not is_cv_speech_primary and career:
        cv_speech_role_count = sum(
            1 for role in career
            if any(term in (role.get("title") or "").lower() for term in CV_SPEECH_DOMAIN_TITLES)
        )
        is_cv_speech_primary = cv_speech_role_count > len(career) / 2

    if is_cv_speech_primary:
        distinct_crossover = {
            kw for kw in NLP_IR_CROSSOVER_KEYWORDS if kw in narrative_lower
        }
        is_cv_speech_no_nlp = len(distinct_crossover) < 2
    else:
        is_cv_speech_no_nlp = False

    # ── Salary ────────────────────────────────────────────────────────────────
    sal          = (sig.get("expected_salary_range_inr_lpa") or {})
    salary_min_lpa = float(sal.get("min", 0)   or 0)
    salary_max_lpa = float(sal.get("max", 999) or 999)

    # ── Skill assessment bonus ────────────────────────────────────────────────
    skill_assessment_bonus = compute_skill_assessment_bonus(
        sig.get("skill_assessment_scores") or {}
    )

    # ── Education ─────────────────────────────────────────────────────────────
    edu_tier  = edu[0].get("tier",           "unknown") if edu else "unknown"
    edu_field = edu[0].get("field_of_study", "")        if edu else ""
    edu_bonus = compute_education_bonus(edu_tier, edu_field)

    # ── Career trajectory ─────────────────────────────────────────────────────
    durations         = [r.get("duration_months", 0) or 0 for r in career]
    avg_tenure_months = float(sum(durations) / len(durations)) if durations else 0.0

    title_now   = get_title_seniority(p.get("current_title", ""))
    title_first = get_title_seniority(career[-1].get("title", "") if career else "")
    trajectory_upward = title_now > title_first

    tenure_stability = (1.0 if avg_tenure_months > 24
                        else (0.7 if avg_tenure_months > 12 else 0.3))
    ml_recency_score = (1.0 if years_since_last_ml_role <= 0
                        else (0.5 if years_since_last_ml_role <= 2
                              else (0.2 if years_since_last_ml_role <= 4 else 0.0)))

    # Title-chasing penalty — jd.txt explicit "do NOT want" #1:
    # "Optimizing for Senior→Staff→Principal titles by switching every 1.5 years."
    years_per_role   = (yoe / n_total_roles) if n_total_roles > 0 else yoe
    is_title_chasing = trajectory_upward and n_total_roles >= 3 and years_per_role < 1.5
    upward_term      = -0.10 if is_title_chasing else 0.3 * float(trajectory_upward)

    trajectory_score = 0.4 * tenure_stability + upward_term + 0.3 * ml_recency_score
    trajectory_score = max(0.0, min(1.0, trajectory_score))

    # Junior-title stagnation — JD seeks someone who's "hit senior engineer
    # judgment." Someone who has spent the bulk of their career (>=36 months)
    # at a junior seniority level (title_now == 1) with no upward movement is
    # a meaningful mismatch against the "founding team / senior judgment" bar.
    # Only fires when the current title is junior AND they haven't moved up.
    is_junior_stagnant = (
        title_now == 1                   # currently titled junior
        and not trajectory_upward        # no upward movement detected
        and sum(durations) >= 36         # at least 3 years total tenure
    )

    # ── Availability signals ──────────────────────────────────────────────────
    open_to_work_flag        = bool(sig.get("open_to_work_flag", False))
    recruiter_response_rate  = float(sig.get("recruiter_response_rate", 0.0) or 0.0)
    avg_response_time_hours  = float(
        sig.get("avg_response_time_hours")
        if sig.get("avg_response_time_hours") is not None else -1
    )
    notice_period_days       = int(sig.get("notice_period_days", 90) or 90)
    interview_completion_rate = float(
        sig.get("interview_completion_rate")
        if sig.get("interview_completion_rate") is not None else -1
    )
    offer_acceptance_rate    = float(
        sig.get("offer_acceptance_rate")
        if sig.get("offer_acceptance_rate") is not None else -1
    )
    saved_by_recruiters_30d  = int(sig.get("saved_by_recruiters_30d", 0) or 0)
    verified_email           = bool(sig.get("verified_email", False))
    verified_phone           = bool(sig.get("verified_phone", False))
    github_activity_score    = float(
        sig.get("github_activity_score")
        if sig.get("github_activity_score") is not None else -1
    )
    preferred_work_mode      = sig.get("preferred_work_mode", "") or ""

    # last_active_days_ago is DERIVED from last_active_date — the raw key does
    # not exist in the dataset; using it directly would silently return 0.
    try:
        last_active        = date.fromisoformat(str(sig.get("last_active_date", ""))[:10])
        last_active_days_ago = (REFERENCE_DATE - last_active).days
    except Exception:
        last_active_days_ago = 999

    # ── Honeypot detection ────────────────────────────────────────────────────
    is_honeypot = False
    if detect_honeypot:
        # Condition 1: expert skill with duration_months == 0
        for s in skills:
            if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0:
                is_honeypot = True
                break

        # Condition 2: expert skill total duration > yoe * 12 * 1.6
        if not is_honeypot:
            expert_total = sum(
                s.get("duration_months", 0) or 0
                for s in skills if s.get("proficiency") == "expert"
            )
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

        # Condition 5: years_of_experience > 40
        if not is_honeypot and yoe > 40:
            is_honeypot = True

        # Condition 6: synthetic inflation (perfect profile + extreme saves)
        if not is_honeypot:
            completeness = float(sig.get("profile_completeness_score", 0) or 0)
            linkedin     = bool(sig.get("linkedin_connected", False))
            rr           = float(sig.get("recruiter_response_rate", 0) or 0)
            if (completeness == 100 and verified_email and verified_phone
                    and linkedin and saved_by_recruiters_30d > 50 and rr == 1.0):
                is_honeypot = True

    return {
        "candidate_id":                c["candidate_id"],
        "years_of_experience":          yoe,
        "country":                      country,
        "location":                     location,
        "current_title":                p.get("current_title", ""),
        "current_company":              p.get("current_company", ""),
        "current_industry":             p.get("current_industry", ""),
        # Location
        "is_india_based":               is_india_based,
        "is_target_city":               is_target_city,
        "is_primary_city":              is_primary_city,
        "is_tier_1_city":               is_tier_1_city,
        "willing_to_relocate":          willing_to_relocate,
        # Company type
        "n_total_roles":                n_total_roles,
        "n_it_services_roles":          n_it_services_roles,
        "entire_career_it_services":    entire_career_it_services,
        "n_research_only_roles":        n_research_only_roles,
        "entire_career_research_only":  entire_career_research_only,
        "has_product_company_exp":      has_product_company_exp,
        "job_hop_score":                years_per_role,
        # ML experience
        "has_ml_production_experience": has_ml_production_experience,
        "years_since_last_ml_role":     years_since_last_ml_role,
        "n_ml_roles":                   n_ml_roles,
        "total_ml_months":              total_ml_months,
        "shallow_recent_ml_only":       shallow_recent_ml_only,
        # Trajectory
        "avg_tenure_months":            avg_tenure_months,
        "trajectory_upward":            trajectory_upward,
        "trajectory_score":             trajectory_score,
        "is_junior_stagnant":           is_junior_stagnant,
        # Salary
        "salary_min_lpa":               salary_min_lpa,
        "salary_max_lpa":               salary_max_lpa,
        # Skill assessment
        "skill_assessment_bonus":       skill_assessment_bonus,
        # Education
        "edu_tier":                     edu_tier,
        "edu_field":                    edu_field,
        "edu_bonus":                    edu_bonus,
        # Availability
        "open_to_work_flag":            open_to_work_flag,
        "last_active_days_ago":         last_active_days_ago,
        "recruiter_response_rate":      recruiter_response_rate,
        "avg_response_time_hours":      avg_response_time_hours,
        "notice_period_days":           notice_period_days,
        "interview_completion_rate":    interview_completion_rate,
        "offer_acceptance_rate":        offer_acceptance_rate,
        "saved_by_recruiters_30d":      saved_by_recruiters_30d,
        "verified_email":               verified_email,
        "verified_phone":               verified_phone,
        "github_activity_score":        github_activity_score,
        "preferred_work_mode":          preferred_work_mode,
        "is_honeypot":                  is_honeypot,
        # Narrative evidence (R1–R5)
        "narrative_text":               narrative_text,
        "narrative_embedding_score":    narrative_embedding_score,
        "has_disqualifying_language":   has_disqualifying_language,
        "n_ghost_skills":               n_ghost_skills,
        "is_ghost_skill_candidate":     is_ghost_skill_candidate,
        "is_cv_speech_primary":         is_cv_speech_primary,
        "is_cv_speech_no_nlp":          is_cv_speech_no_nlp,
        "top_jd_skills":                top_jd_skills,
    }
