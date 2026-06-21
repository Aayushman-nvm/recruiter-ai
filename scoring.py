"""
scoring.py — Single source of truth for all scoring logic.

Imported by: rank.py, sandbox/app.py, scripts/05_eval.py

All scoring constants and compute_*() functions live here.
No other file should redefine these formulas.
"""

from datetime import date

import numpy as np

# ── Reference date ──────────────────────────────────────────────────────────
# Fixed reference point for all date-relative computations (last_active_days_ago, etc.)
# NOTE: this MUST be >= the latest last_active_date in the dataset, or every
# last_active_days_ago comes out negative and silently maxes out the recency
# sub-score for the entire population (verified: with the old 2025-06-01 value,
# all 100,000 rows in features.parquet had last_active_days_ago < 0, since the
# dataset's last_active_date actually ranges 2025-09-29 -> 2026-05-27).
# Bump this if you regenerate the dataset with a later snapshot date.
REFERENCE_DATE = date(2026, 5, 28)

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
        # Defensive fallback only — under the current extraction logic,
        # has_ml_production_experience=True always pairs with a real
        # (non-sentinel) years_since_last_ml_role, so this branch shouldn't
        # actually fire. Kept in case extraction logic changes upstream.
        if years_since_last_ml_role >= 99.0:
            return 0.65
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
    interview_completion_rate: float = 0.5,
    offer_acceptance_rate: float = -1.0,
    github_activity_score: float = -1.0,
) -> float:
    """
    Availability / track-record signal.

    interview_completion_rate, offer_acceptance_rate, and github_activity_score
    were being extracted into features.parquet by 01_extract_features.py but
    were never wired into this function (interview_completion_rate had even
    been explicitly removed from the signature at one point — "I1 fix" — and
    never reinstated). All three are real signals a recruiter would actually
    look at ("decent activity in terms of socials, github, response time,
    interview completion and acceptance rate") and are added back here.

    avg_response_time_hours / offer_acceptance_rate / github_activity_score:
    -1 means no history -> treated as neutral (0.5), not penalized. Many
    strong candidates simply have no GitHub linked or no offer history yet —
    that's an absence of data, not a negative signal.
    interview_completion_rate: plain 0-1 rate per the schema, no -1 sentinel.
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

    # Interview completion — direct 0-1 rate, no sentinel case in this field
    interview_score = max(0.0, min(1.0, float(interview_completion_rate)))

    # Offer acceptance — historical rate; -1 = no offer history yet → neutral
    if offer_acceptance_rate is None or offer_acceptance_rate < 0:
        offer_score = 0.5
    else:
        offer_score = max(0.0, min(1.0, float(offer_acceptance_rate)))

    # GitHub activity — -1/None = no GitHub linked → neutral, NOT penalized
    # (plenty of strong candidates, especially from closed-source product
    # companies, just don't have public activity tied to their account).
    if github_activity_score is None or github_activity_score < 0:
        github_score = 0.5
    else:
        github_score = max(0.0, min(1.0, float(github_activity_score) / 100.0))

    # Social proof — capped at 20
    social_proof = min(saved_by_recruiters_30d, 20) / 20.0

    # Reachability composite
    reachability = 0.5 * float(verified_email) + 0.5 * float(verified_phone)

    availability = (
        0.18 * float(open_to_work) +
        0.16 * recency +
        0.13 * float(recruiter_response_rate) +
        0.13 * notice_score +
        0.11 * interview_score +
        0.08 * response_time_score +
        0.08 * offer_score +
        0.06 * github_score +
        0.04 * social_proof +
        0.03 * reachability
    )
    return min(1.0, availability)


def compute_location_multiplier(
    is_india_based: bool,
    is_target_city: bool,
    willing_to_relocate: bool,
) -> float:
    """
    Multiplicative final-score dampener for location/relocation logistics —
    applied directly to final_score in rank.py, on top of (not instead of)
    compute_location_fit()'s additive contribution inside structural_score.

    Why a second, multiplicative term is needed: compute_location_fit's effect
    on final_score was getting diluted to near-irrelevance by the time it
    passed through structural_score (one term among five) -> prelim_score
    (one term among three) -> final_score (blended with the cross-encoder,
    which has ZERO visibility into location at all — it only ever sees JD
    text vs. candidate profile text). In practice this meant a candidate who
    is neither in a target city nor willing to relocate could still rank #1
    purely on text-similarity strength, which doesn't match how a recruiter
    would actually triage candidates: target-city first, willing-to-relocate
    next, same-country-but-stuck after that, international last.

    This stays continuous (no hard cutoff/disqualification) — a much-better
    -fit non-local candidate can still outrank a weaker local one — but now
    the gap is large enough to actually matter, not just exist on paper.
    """
    if not is_india_based:
        return 0.78 if willing_to_relocate else 0.55
    if is_target_city:
        return 1.00
    elif willing_to_relocate:
        return 0.95
    else:
        return 0.85


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
    yrs_since_ml      = float(row.get("years_since_last_ml_role") if row.get("years_since_last_ml_role") is not None else 99)
    traj_score        = float(row.get("trajectory_score") if row.get("trajectory_score") is not None else 0)
    avg_tenure        = float(row.get("avg_tenure_months") if row.get("avg_tenure_months") is not None else 0)
    trajectory_up     = bool(row.get("trajectory_upward", False))

    sal_min           = float(row.get("salary_min_lpa") if row.get("salary_min_lpa") is not None else 0)
    sal_max           = float(row.get("salary_max_lpa") if row.get("salary_max_lpa") is not None else 999)
    open_to_work      = bool(row.get("open_to_work_flag", False))
    days_ago          = int(row.get("last_active_days_ago") if row.get("last_active_days_ago") is not None else 999)
    response_rate     = float(row.get("recruiter_response_rate") if row.get("recruiter_response_rate") is not None else 0)
    notice_days       = int(row.get("notice_period_days") if row.get("notice_period_days") is not None else 90)
    resp_time_hrs     = float(row.get("avg_response_time_hours") if row.get("avg_response_time_hours") is not None else -1)
    saves             = int(row.get("saved_by_recruiters_30d") if row.get("saved_by_recruiters_30d") is not None else 0)
    skill_bonus       = float(row.get("skill_assessment_bonus") if row.get("skill_assessment_bonus") is not None else 0)
    edu_tier          = str(row.get("edu_tier", "unknown") or "unknown")

    structural   = float(row.get("structural_score") if row.get("structural_score") is not None else 0)
    availability = float(row.get("availability_score") if row.get("availability_score") is not None else 0)
    ce_score     = float(row.get("ce_score") if row.get("ce_score") is not None else 0)
    final_score  = float(row.get("final_score") if row.get("final_score") is not None else 0)
    rank         = int(row.get("rank") if row.get("rank") is not None else 0)

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


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval fusion + pipeline blend weights — single source of truth.
#
# rank.py and 04_eval.py both implement this same multi-stage blend
# (retrieval fusion -> +structural -> +availability -> +cross-encoder), and
# previously each hardcoded its own copy of every weight. That's the same
# class of drift risk as the ML_KEYWORDS/seniority duplication fixed earlier
# in this pass — so all of it now lives here once.
# ─────────────────────────────────────────────────────────────────────────────

# Fusion of BM25 (keyword) + dense (semantic) retrieval.
# Was 0.35/0.65 — moderated after eval_results.json showed weighting fusion
# that heavily toward dense pulled Config C down (0.6911 -> 0.6397 vs RRF)
# on the labeled eval set. The theoretical argument for favoring semantic
# match still holds (jd.txt's keyword-stuffing trap is real), but a 50-
# candidate, 10-relevant eval shouldn't be ignored either — this is a more
# moderate compromise than either extreme. Revisit once the eval set is larger.
FUSION_BM25_WEIGHT  = 0.45
FUSION_DENSE_WEIGHT = 0.55

# prelim_score = fusion + structural + availability (pre-cross-encoder).
PRELIM_FUSION_WEIGHT       = 0.55
PRELIM_STRUCTURAL_WEIGHT   = 0.25
PRELIM_AVAILABILITY_WEIGHT = 0.20

# final_score = prelim_score + cross-encoder.
# Was 0.30/0.70 — walked back hard after eval_results.json showed Config F
# (full pipeline incl. cross-encoder) scoring *worse* than Config E (prelim
# alone, no CE) in both the old run (0.5466 vs 0.6474) and the new one
# (0.5713 vs 0.7276). That's not noise from one run — it's the same direction
# twice, with different weights both times. The theoretical case for CE
# (real cross-attention, more contextual than a single embedding dot product)
# doesn't survive contact with this specific small CE model on long,
# multi-topic candidate-profile text — trusting the eval over the theory here.
# prelim_score now carries most of the final blend; CE is a tiebreaker, not
# the deciding vote.
FINAL_PRELIM_WEIGHT = 0.60
FINAL_CE_WEIGHT     = 0.40

assert abs(FUSION_BM25_WEIGHT + FUSION_DENSE_WEIGHT - 1.0) < 1e-9
assert abs(PRELIM_FUSION_WEIGHT + PRELIM_STRUCTURAL_WEIGHT + PRELIM_AVAILABILITY_WEIGHT - 1.0) < 1e-9
assert abs(FINAL_PRELIM_WEIGHT + FINAL_CE_WEIGHT - 1.0) < 1e-9


def weighted_score_fusion(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    candidate_ids: list,
    bm25_weight: float = FUSION_BM25_WEIGHT,
    dense_weight: float = FUSION_DENSE_WEIGHT,
) -> dict:
    """
    Score-based fusion of BM25 and dense retrieval — replaces Reciprocal Rank
    Fusion (RRF).

    RRF converts both rankings to rank *positions* before combining
    (1/(k+rank+1) per list), which throws away how strong a match actually was.
    A candidate with dense cosine 0.95 and one with cosine 0.32 get identical
    credit if they're both rank #1 in their own list — RRF has no notion of
    "confident match" vs "barely squeaked into first place". That's a real
    problem here specifically because the JD's whole point is that keyword
    overlap is an unreliable signal (see jd.txt's closing note to participants)
    — RRF structurally can't let a strong semantic match win over a weak
    keyword match; it can only let them tie.

    This function instead min-max normalizes each raw score distribution to
    [0, 1] and takes a weighted sum, so *how confident* each method was
    directly affects the fused score, not just *whether* it ranked first.

    bm25_scores / dense_scores must be aligned with candidate_ids by position
    (i.e. bm25_scores[i] and dense_scores[i] both refer to candidate_ids[i]).
    """
    bm25_arr  = np.asarray(bm25_scores, dtype=np.float64)
    dense_arr = np.asarray(dense_scores, dtype=np.float64)

    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    fused = bm25_weight * _minmax(bm25_arr) + dense_weight * _minmax(dense_arr)
    return dict(zip(candidate_ids, fused.tolist()))


def rrf_fusion(rank_list_a: list, rank_list_b: list, k: int = 60) -> dict:
    """
    Reciprocal Rank Fusion — kept for backward compatibility (e.g. sandbox/app.py
    or any other consumer that may still call it). New code should prefer
    weighted_score_fusion() above; see its docstring for why.
    """
    scores = {}
    for rank, cid in enumerate(rank_list_a):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(rank_list_b):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores