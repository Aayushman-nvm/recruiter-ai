"""
scoring.py — Single source of truth for all scoring logic.

Imported by: rank.py, sandbox/app.py, scripts/05_eval.py

All scoring constants and compute_*() functions live here.
No other file should redefine these formulas.
"""

from datetime import date

# ── Reference date ──────────────────────────────────────────────────────────
# Fixed reference point for all date-relative computations (last_active_days_ago, etc.)
REFERENCE_DATE = date(2025, 6, 1)

# ── Target cities ───────────────────────────────────────────────────────────
# All India cities explicitly mentioned in the JD or implied by proximity to Noida.
# Single authoritative source — Script 01 uses this to compute is_target_city boolean.
TARGET_CITIES = {
    "pune", "noida", "delhi", "ncr", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "gurgaon", "gurugram", "faridabad",
}

# ── IT services companies ────────────────────────────────────────────────────
# Explicit disqualifier in JD: entire career in IT services consulting.
IT_SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "mphasis", "hexaware",
    "ltimindtree", "tech mahindra", "hcl technologies", "hcltech", "hcl",
    "ibm global", "atos", "dxc technology", "epam", "niit technologies",
}

# ── JD-relevant skills for skill_assessment_scores bonus ─────────────────────
JD_RELEVANT_SKILLS = {
    "python", "nlp", "machine learning", "deep learning",
    "information retrieval", "ranking", "search", "embeddings",
    "recommendation systems", "data science", "pytorch", "tensorflow",
}

# ── Estimated salary target range (INR LPA) ──────────────────────────────────
# Senior AI Engineer, Series A India startup, 5–9 YoE, AI specialisation.
# Wide range deliberately chosen to reduce false negatives from estimation error.
SALARY_TARGET_MIN = 20.0
SALARY_TARGET_MAX = 65.0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_experience_fit(years: float) -> float:
    """
    Score how well years of experience matches JD (sweet spot: 5–9 years).
    JD explicitly states experience band is not a hard requirement.
    Curve plateaus at 10–15 years, tapers slowly after — overqualified ≠ unqualified.
    """
    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5 or 9 < years <= 12:
        return 0.82
    elif 3 <= years < 4 or 12 < years <= 15:
        return 0.65
    elif years > 15:
        return 0.55   # overqualified but not disqualified
    else:
        return 0.25   # < 3 years: genuinely under-experienced


def compute_location_fit(
    is_india_based: bool,
    is_target_city: bool,
    willing_to_relocate: bool,
) -> float:
    """
    Score location fit. JD office is Noida; India-based strongly preferred.
    Uses pre-computed boolean columns from features.parquet (Script 01).
    """
    if not is_india_based:
        return 0.15 if not willing_to_relocate else 0.45
    if is_target_city:
        return 1.0
    elif willing_to_relocate:
        return 0.78
    else:
        return 0.52


def compute_company_fit(
    entire_career_it_services: bool,
    has_product_company_exp: bool,
    has_ml_production_experience: bool,
    years_since_last_ml_role: float,
) -> float:
    """
    Score company + ML production background.

    Hard disqualifier: entire career in IT services (explicit in JD).

    Bug 2 fix: stale ML always scores >= no ML ever.
    When ml_recency decays below 0.65, we floor at 0.65 (= product-no-ML baseline).
    This ensures "product + ML from 4 years ago" >= "product + zero ML experience".
    """
    if entire_career_it_services:
        return 0.0   # hard disqualifier

    # ML recency decay
    if years_since_last_ml_role <= 0:
        ml_recency = 1.0
    elif years_since_last_ml_role <= 1:
        ml_recency = 0.85
    elif years_since_last_ml_role <= 2:
        ml_recency = 0.65
    elif years_since_last_ml_role <= 4:
        ml_recency = 0.40
    else:
        ml_recency = 0.20

    if has_product_company_exp and has_ml_production_experience:
        # Guard: sentinel 99.0 means "never did ML" — treat as no ML experience
        if years_since_last_ml_role >= 99.0:
            return 0.65   # product, no ML (sentinel case)
        # Floor at 0.65 — stale ML always beats no-ML-at-all (Bug 2 fix)
        return max(0.65, 1.0 * ml_recency)
    elif has_product_company_exp:
        return 0.65
    else:
        return 0.38


def compute_salary_fit(expected_min: float, expected_max: float) -> float:
    """
    Score salary alignment with estimated role budget (20–65 LPA).
    Soft signal — wide range reduces false negatives from estimation error.
    """
    # Full overlap with target range
    if expected_min <= SALARY_TARGET_MAX and expected_max >= SALARY_TARGET_MIN:
        return 1.0
    # Candidate expects significantly less (may indicate junior level)
    if expected_max < SALARY_TARGET_MIN * 0.7:
        return 0.55
    # Candidate expects significantly more (budget mismatch)
    if expected_min > SALARY_TARGET_MAX * 1.5:
        return 0.35
    # Mild mismatch — partial overlap or close to range
    return 0.72


def compute_skill_assessment_bonus(skill_assessment_scores: dict) -> float:
    """
    Bonus for platform-verified skills in JD-relevant areas.
    Sparse signal (~20% of candidates). When present: high-signal verified ability.
    Max bonus: 0.08 (meaningful tiebreaker, not dominant).
    """
    if not skill_assessment_scores:
        return 0.0
    relevant_scores = []
    for skill_name, score in skill_assessment_scores.items():
        if any(jd_skill in skill_name.lower() for jd_skill in JD_RELEVANT_SKILLS):
            relevant_scores.append(float(score) / 100.0)
    if not relevant_scores:
        return 0.0
    return min(0.08, sum(relevant_scores) / len(relevant_scores) * 0.08)


def compute_education_bonus(edu_tier: str, field_of_study: str) -> float:
    """
    Small education tiebreaker. Not a dominant signal — JD doesn't list education
    as a hard requirement. Used only to differentiate near-equal candidates.
    Range: -0.02 to +0.05.
    """
    RELEVANT_FIELDS = {
        "computer science", "cs", "information technology",
        "machine learning", "data science", "statistics",
        "mathematics", "electrical engineering", "electronics",
    }
    field_lower = (field_of_study or "").lower()
    field_relevant = any(f in field_lower for f in RELEVANT_FIELDS)

    if edu_tier == "tier_1":
        return 0.05 if field_relevant else 0.03
    elif edu_tier == "tier_2":
        return 0.03 if field_relevant else 0.01
    elif edu_tier == "tier_3":
        return 0.0
    elif edu_tier == "tier_4":
        return -0.02
    return 0.0   # unknown tier → neutral


def compute_availability_score(
    open_to_work: bool,
    last_active_days_ago: int,
    recruiter_response_rate: float,
    avg_response_time_hours: float,
    notice_period_days: int,
    saved_by_recruiters_30d: int,
    verified_email: bool,
    verified_phone: bool,
) -> float:
    """
    Availability signal. Five primary sub-signals + social proof + reachability.

    Note: interview_completion_rate intentionally NOT a parameter (I1 fix).
    It was accepted by the function before but never used in the formula body.
    Removed to avoid dead parameter confusion at Stage 4.

    avg_response_time_hours: -1 means no history → treated as neutral (0.5).
    saved_by_recruiters_30d: capped at 20 to avoid over-rewarding in-demand candidates.
    """
    # Recency
    if last_active_days_ago <= 14:
        recency = 1.0
    elif last_active_days_ago <= 30:
        recency = 0.85
    elif last_active_days_ago <= 90:
        recency = 0.55
    elif last_active_days_ago <= 180:
        recency = 0.25
    else:
        recency = 0.08

    # Notice period
    if notice_period_days <= 15:
        notice_score = 1.0
    elif notice_period_days <= 30:
        notice_score = 0.85
    elif notice_period_days <= 60:
        notice_score = 0.60
    elif notice_period_days <= 90:
        notice_score = 0.35
    else:
        notice_score = 0.15

    # Response time (-1 = no history → neutral)
    if avg_response_time_hours < 0:
        response_time_score = 0.5
    elif avg_response_time_hours <= 4:
        response_time_score = 1.0
    elif avg_response_time_hours <= 24:
        response_time_score = 0.75
    elif avg_response_time_hours <= 72:
        response_time_score = 0.40
    else:
        response_time_score = 0.10

    # Social proof — capped at 20
    social_proof = min(saved_by_recruiters_30d, 20) / 20.0

    # Reachability composite
    reachability = 0.5 * float(verified_email) + 0.5 * float(verified_phone)

    availability = (
        0.25 * float(open_to_work) +
        0.22 * recency +
        0.18 * float(recruiter_response_rate) +
        0.15 * notice_score +
        0.10 * response_time_score +
        0.05 * social_proof +
        0.05 * reachability
    )
    return min(1.0, availability)


def generate_reasoning(row: dict) -> str:
    """
    Generate 2-sentence reasoning from feature columns — no LLM required.

    Sentence 1: Lead signal — what most strongly drove this ranking decision.
                Branches on the dominant positive or disqualifying factor.
    Sentence 2: Supporting signal — a concrete secondary fact that reinforces
                or qualifies sentence 1. Always cites a real number.

    Design principles:
    - Every branch cites a specific value from the row (years, days, rate, etc.)
    - Sentence structure rotates across branches so adjacent candidates read differently
    - Reasoning is causally tied to the score: the branch chosen matches the
      signal that actually dominated the ranking formula
    - No generic filler phrases ("strong background", "excellent fit")

    Input: a dict-like row with columns from features.parquet plus computed
           structural_score, availability_score, ce_score, final_score, rank.
    """

    # ── Extract key values ────────────────────────────────────────────────────
    title        = str(row.get("current_title", "candidate") or "candidate").strip()
    yoe          = float(row.get("years_of_experience", 0) or 0)
    company      = str(row.get("current_company", "") or "").strip()
    location     = str(row.get("location", "") or "").strip()
    country      = str(row.get("country", "") or "").strip()

    is_india          = bool(row.get("is_india_based", False))
    is_target_city    = bool(row.get("is_target_city", False))
    willing_relocate  = bool(row.get("willing_to_relocate", False))
    entire_it         = bool(row.get("entire_career_it_services", False))
    has_product       = bool(row.get("has_product_company_exp", False))
    has_ml            = bool(row.get("has_ml_production_experience", False))
    yrs_since_ml      = float(row.get("years_since_last_ml_role", 99) or 99)
    traj_score        = float(row.get("trajectory_score", 0) or 0)
    avg_tenure        = float(row.get("avg_tenure_months", 0) or 0)
    trajectory_up     = bool(row.get("trajectory_upward", False))

    sal_min           = float(row.get("salary_min_lpa", 0) or 0)
    sal_max           = float(row.get("salary_max_lpa", 999) or 999)
    open_to_work      = bool(row.get("open_to_work_flag", False))
    days_ago          = int(row.get("last_active_days_ago", 999) or 999)
    response_rate     = float(row.get("recruiter_response_rate", 0) or 0)
    notice_days       = int(row.get("notice_period_days", 90) or 90)
    resp_time_hrs     = float(row.get("avg_response_time_hours", -1) or -1)
    saves             = int(row.get("saved_by_recruiters_30d", 0) or 0)
    skill_bonus       = float(row.get("skill_assessment_bonus", 0) or 0)
    edu_tier          = str(row.get("edu_tier", "unknown") or "unknown")

    structural   = float(row.get("structural_score", 0) or 0)
    availability = float(row.get("availability_score", 0) or 0)
    ce_score     = float(row.get("ce_score", 0) or 0)
    final_score  = float(row.get("final_score", 0) or 0)
    rank         = int(row.get("rank", 0) or 0)

    # ── Shorthands ────────────────────────────────────────────────────────────
    yoe_str     = f"{yoe:.1f}"
    loc_str     = f"{location}, {country}" if location and country else (location or country or "unknown location")
    company_str = company if company else "their current employer"

    # ── Sentence 1: Lead signal ───────────────────────────────────────────────
    # Priority order: disqualifiers first, then strongest positive signal.

    if entire_it:
        # Hard disqualifier — explain why they're in the list at all (low rank)
        s1 = (f"{title} with {yoe_str} years of experience whose entire career "
              f"has been in IT services consulting, which the JD explicitly disqualifies.")

    elif not has_ml and not has_product:
        # No ML and no product company — weakest structural profile
        s1 = (f"{title} with {yoe_str} years of experience at {company_str}, "
              f"with no detected production ML experience and no product-company background.")

    elif has_ml and yrs_since_ml <= 1:
        # Currently doing ML — strongest signal
        s1 = (f"{title} with {yoe_str} years of experience currently working in "
              f"production ML at {company_str}, directly matching the role's core requirement.")

    elif has_ml and 1 < yrs_since_ml <= 3:
        # ML experience but somewhat stale
        s1 = (f"{title} with {yoe_str} years of experience who last worked in a production ML role "
              f"{yrs_since_ml:.1f} years ago at {company_str}, bringing relevant but dated hands-on experience.")

    elif has_ml and yrs_since_ml > 3:
        # ML experience, notably stale
        s1 = (f"{title} with {yoe_str} years of experience whose most recent production ML role "
              f"was {yrs_since_ml:.1f} years ago — the role requires active hands-on ML work, not historical.")

    elif has_product and not has_ml:
        # Product company but no ML keywords detected
        s1 = (f"{title} with {yoe_str} years at product companies including {company_str}, "
              f"but no explicit production ML or retrieval/ranking experience detected in their career history.")

    elif ce_score >= 0.7:
        # Cross-encoder found strong semantic match despite weaker structural signals
        s1 = (f"{title} with {yoe_str} years of experience whose profile content closely "
              f"matches the technical requirements of this role (semantic match: {ce_score:.2f}).")

    else:
        # Generic fallback — use experience and title
        s1 = (f"{title} with {yoe_str} years of experience at {company_str}, "
              f"ranked {rank} based on a combination of semantic fit and structured signals.")

    # ── Sentence 2: Supporting / qualifying signal ────────────────────────────
    # Pick the most informative secondary fact — rotate structure based on what's notable.

    if entire_it:
        # For disqualified candidates, cite their availability as the reason they appear at all
        if open_to_work and days_ago <= 30:
            s2 = (f"Despite the disqualification, they are actively available "
                  f"(open to work, last active {days_ago} days ago, response rate {response_rate:.0%}).")
        else:
            s2 = (f"Their recruiter response rate is {response_rate:.0%} "
                  f"and they were last active {days_ago} days ago.")

    elif not is_india:
        # International candidate — location is the key concern
        if willing_relocate:
            s2 = (f"Based in {loc_str}, they are willing to relocate to India, "
                  f"though visa sponsorship is not available per the JD.")
        else:
            s2 = (f"Based in {loc_str} and not willing to relocate — "
                  f"significant location mismatch for a Noida-based role.")

    elif is_india and not is_target_city and not willing_relocate:
        # India-based but wrong city and not relocating
        s2 = (f"Currently in {loc_str}, not in a JD-preferred city and not open to relocation "
              f"— partial location fit only.")

    elif skill_bonus >= 0.04:
        # Platform-verified skill scores are rare and high-signal — lead with them
        s2 = (f"Platform-verified skill assessments (bonus: {skill_bonus:.3f}) provide "
              f"independently confirmed technical ability, a signal absent in most candidates.")

    elif traj_score >= 0.8 and trajectory_up:
        # Strong upward trajectory — cite tenure as evidence
        tenure_str = f"{avg_tenure:.0f}" if avg_tenure > 0 else "unknown"
        s2 = (f"Their career shows upward title progression with an average tenure of "
              f"{tenure_str} months per role, indicating stability and growth rather than title-chasing.")

    elif avg_tenure < 15 and avg_tenure > 0:
        # Short tenures — flag title-chasing pattern
        s2 = (f"Average tenure of {avg_tenure:.0f} months per role suggests a title-chasing "
              f"pattern — a stated JD concern — which the trajectory score ({traj_score:.2f}) reflects.")

    elif open_to_work and days_ago <= 14 and response_rate >= 0.7:
        # Excellent availability — all three signals positive
        s2 = (f"Actively job-seeking: open to work, last active {days_ago} days ago, "
              f"and {response_rate:.0%} recruiter response rate — highly reachable.")

    elif open_to_work and notice_days <= 30:
        # Available and fast start
        notice_str = "immediately" if notice_days == 0 else f"within {notice_days} days"
        s2 = (f"Open to work and can start {notice_str}, "
              f"last active {days_ago} days ago with a {response_rate:.0%} response rate.")

    elif days_ago > 180:
        # Inactive for over 6 months — availability concern
        s2 = (f"Last active {days_ago} days ago with a {response_rate:.0%} response rate — "
              f"low platform engagement raises reachability concerns despite their technical profile.")

    elif resp_time_hrs >= 0 and resp_time_hrs <= 4:
        # Responds very quickly — notable positive
        s2 = (f"Responds to recruiter messages within {resp_time_hrs:.0f} hours on average "
              f"({response_rate:.0%} response rate), indicating strong engagement.")

    elif saves >= 5:
        # Multiple recruiters have bookmarked this candidate — social proof
        s2 = (f"Saved by {saves} other recruiters in the last 30 days — "
              f"crowd-validated interest from the recruiting community.")

    elif sal_min > SALARY_TARGET_MAX * 1.3:
        # Significant salary mismatch — flag it
        s2 = (f"Expected salary (min {sal_min:.0f} LPA) likely exceeds the role's budget "
              f"for a Series A startup — potential offer negotiation risk.")

    elif sal_max < SALARY_TARGET_MIN * 0.7:
        # Under the expected range — flag as potential seniority signal
        s2 = (f"Expected salary range ({sal_min:.0f}–{sal_max:.0f} LPA) is below the "
              f"estimated market rate for this seniority, possibly indicating a junior profile.")

    elif edu_tier == "tier_1":
        s2 = (f"Tier-1 institution background adds a marginal quality signal "
              f"as a tiebreaker ({response_rate:.0%} response rate, {days_ago} days since last active).")

    else:
        # Fallback: cite availability numbers since they're always meaningful
        active_str = "recently" if days_ago <= 30 else f"{days_ago} days ago"
        s2 = (f"Last active {active_str} with a {response_rate:.0%} recruiter response rate "
              f"and {notice_days}-day notice period.")

    return f"{s1} {s2}"


def compute_structural_score(
    experience_fit: float,
    location_fit: float,
    company_fit: float,
    trajectory_score: float,
    salary_fit: float,
    skill_assessment_bonus: float,
    edu_bonus: float,
) -> float:
    """
    Weighted structural score combining all hard-requirement signals.

    Weights reflect JD priorities:
      company background + ML experience (0.30) > location (0.25) > experience band (0.20)
      > trajectory (0.12) > salary (0.08) > bonuses (additive, max ±0.13)

    Bug 4 fix: return min(1.0, ...) — bonuses can push the weighted sum past 1.0
    without the cap. Capped to keep all scores in [0, 1].
    """
    raw = (
        0.30 * company_fit +
        0.25 * location_fit +
        0.20 * experience_fit +
        0.12 * trajectory_score +
        0.08 * salary_fit
    )
    return min(1.0, raw + skill_assessment_bonus + edu_bonus)
