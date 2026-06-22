"""
scoring.py — Single source of truth for all scoring logic.

Imported by: rank.py, sandbox/app.py, scripts/05_eval.py

All scoring constants and compute_*() functions live here.
No other file should redefine these formulas.
"""

import re
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

TIER_1 = {
    "ahmedabad",
    "bengaluru",
    "bangalore",
    "chennai",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "kolkata",
    "mumbai",
    "new delhi",
    "delhi",
    "pune",
}

TIER_2 = {
    "agra",
    "ambala",
    "amravati",
    "amritsar",
    "ananthapur",
    "asansol",
    "belagavi",
    "bhavnagar",
    "bhiwandi",
    "bhopal",
    "bhubaneswar",
    "calicut",
    "ch.sambhajinagar",
    "aurangabad",
    "chandigarh",
    "coimbatore",
    "cuttack",
    "davangere",
    "dhanbad",
    "durg",
    "bhilai",
    "faridabad",
    "gandhinagar",
    "ghaziabad",
    "goa",
    "greater noida",
    "guntur",
    "guwahati",
    "gwalior",
    "hisar",
    "howrah",
    "hooghly",
    "huballi",
    "dharwad",
    "indore",
    "jabalpur",
    "jaipur",
    "jalandhar",
    "jalgaon",
    "jamnagar",
    "jamshedpur",
    "jodhpur",
    "kadapa",
    "kakinada",
    "kalyan dombivli",
    "kanpur",
    "karnal",
    "kochi",
    "kolhapur",
    "kota",
    "kurnool",
    "latur",
    "lucknow",
    "ludhiana",
    "madurai",
    "malegaon",
    "mangaluru",
    "mira bhayander",
    "mohali",
    "moradabad",
    "mysuru",
    "nagpur",
    "nanded",
    "nashik",
    "navi mumbai",
    "nellore",
    "nizamabad",
    "noida",
    "panchkula",
    "patna",
    "prayagraj",
    "puducherry",
    "raipur",
    "rajahmundry",
    "rajkot",
    "ranchi",
    "rohtak",
    "sagar",
    "salem",
    "sangli",
    "satara",
    "solapur",
    "sonipat",
    "surat",
    "thane",
    "thrissur",
    "tirupathi",
    "trichy",
    "trivandrum",
    "udaipur",
    "vadodara",
    "varanasi",
    "vijayawada",
    "visakhapatnam",
    "vizag",
    "warangal",
    "yamuna",
}

TIER_3 = {
    "ahmednagar",
    "akola",
    "aligarh",
    "alwar",
    "amalner",
    "ambajogai",
    "amreli",
    "anand",
    "baramati",
    "bardoli",
    "barshi",
    "bathinda",
    "becharaji",
    "beed",
    "begusarai",
    "berhampur",
    "bhadradri kothagudem",
    "bhandara",
    "bharatpur",
    "bharuch",
    "bhimavaram",
    "bhusawal",
    "bidar",
    "bilaspur",
    "buldhana",
    "chalisgaon",
    "chandrapur",
    "chiplun",
    "dahod",
    "daund",
    "dharmapuri",
    "dhule",
    "eluru",
    "erode",
    "gadhinglaj",
    "gadwal",
    "gandhidham",
    "gaya",
    "godhra",
    "gondia",
    "gurdaspur",
    "hingoli",
    "hoshangabad",
    "hosur",
    "ichalkaranji",
    "indapur",
    "islampur",
    "jagityal",
    "jalna",
    "jaysingpur",
    "jhansi",
    "jharsuguda",
    "junagadh",
    "kachchh",
    "kutch",
    "kadi",
    "kagal",
    "kalol",
    "kamareddy",
    "karad",
    "karimnagar",
    "karim nagar",
    "karwar",
    "khamgaon",
    "khammam",
    "kharar",
    "kopargaon",
    "shirdi",
    "machilipatnam",
    "mahabubnagar",
    "mahad",
    "malvan",
    "mancherial",
    "mathura",
    "mehsana",
    "modasa",
    "mundra",
    "muzaffarpur",
    "nadiad",
    "nalgonda",
    "nandyala",
    "narsipatnam",
    "navsari",
    "nilanga",
    "nirmal",
    "north bengal",
    "omerga",
    "ongole",
    "osmanabad",
    "dharashiv",
    "ozar",
    "pachora",
    "palakkad",
    "palakollu",
    "palanpur",
    "palghar",
    "boisar",
    "pandharpur",
    "parbhani",
    "patan",
    "phaltan",
    "pimpalgaon baswant",
    "puri",
    "raigad",
    "raigarh",
    "ratnagiri",
    "ropar",
    "rupnagar",
    "sabarkantha",
    "sakri",
    "sambalpur",
    "sangrur",
    "sawantwadi",
    "shahada",
    "nandurbar",
    "shahapur",
    "murbad",
    "shirpur",
    "siddipet",
    "silvassa",
    "sindhudurg",
    "sinnar",
    "srikakulam",
    "tirunelveli",
    "tuni",
    "udgir",
    "udupi",
    "ujjain",
    "unjha",
    "uran",
    "dronagiri",
    "valsad",
    "vapi",
    "vijapur",
    "vijayapura",
    "visnagar",
    "vizianagaram",
    "vyara",
    "wanaparthy",
    "wani",
    "wardha",
    "washim",
    "yavatmal",
    "zirakpur",
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

# ── Primary office cities ────────────────────────────────────────────────────
# jd.txt's actual offices ("Location: Pune/Noida, India") and the location
# named in its own "ideal candidate" summary ("How to read between the lines":
# "Located in or willing to relocate to Noida or Pune"). Distinct from
# TARGET_CITIES below — the JD ranks Pune/Noida above the other welcomed
# cities (Hyderabad, Mumbai, Delhi NCR, Bengaluru), it just doesn't rank those
# against each other, so compute_location_fit() only needs two tiers.
PRIMARY_CITIES = {"pune", "noida"}

# ── Research-only / academic career detection ────────────────────────────────
# Explicit hard disqualifier in jd.txt: "If you've spent your career in pure
# research environments (academic labs, research-only roles) without any
# production deployment — we will not move forward. We are explicit about
# this." Mirrors IT_SERVICES_COMPANIES' role -- a company/title name heuristic,
# checked per-role in 01_extract_features.py / 04_eval.py the same way IT
# services employers are.
#
# Deliberately narrow: only unambiguous academic-institution / pure-research
# names and titles. Does NOT include corporate research-lab names ("Google
# Research", "Microsoft Research", "X Labs") -- jd.txt's own examples are
# academic ("academic labs"), and those roles sit inside real product
# companies that also ship production systems, so flagging them would create
# false positives well beyond what the JD describes.
RESEARCH_ONLY_COMPANY_INDICATORS = {
    "university", "institute of technology", "iisc",
    "indian institute of science", "research institute",
    "academy of sciences", "college of engineering",
}
RESEARCH_ONLY_TITLE_INDICATORS = {
    "phd", "postdoc", "post-doc", "doctoral researcher", "research fellow",
    "research scientist", "research intern", "graduate researcher",
}

# ── Industry relevance for current_industry bonus ────────────────────────────
# jd.txt nice-to-have: "Prior exposure to HR-tech, recruiting tech, or
# marketplace products." Word/phrase choices avoid bare substrings that
# collide with unrelated industries (e.g. no bare "search", which is a
# substring of "Research").
INDUSTRY_RELEVANT_KEYWORDS = {
    "hr tech", "hrtech", "human resources", "recruit", "talent",
    "staffing", "marketplace", "e-commerce", "ecommerce",
    "search engine", "classifieds", "job board", "jobs platform",
}

# ── ML/IR production-experience keywords ─────────────────────────────────────
# Single authoritative source for has_ml_production_experience detection.
# Previously duplicated verbatim (with a "must stay in sync" comment) across
# 01_extract_features.py and 04_eval.py -- a real drift already happened once
# (04_eval.py's seniority mapping silently fell out of sync, see
# get_title_seniority() below). Centralizing here removes that risk the same
# way TARGET_CITIES/IT_SERVICES_COMPANIES already do for their callers.
#
# Dropped bare "ann" — confirmed root cause of has_ml_production_experience
# being True on ~52% of completely unrelated titles (HR Manager, Accountant,
# Civil Engineer, ...) in the real dataset. "ann" as a naive substring matched
# inside "channel", "planning", "announce", etc. "approximate nearest
# neighbor" and "hnsw" below already cover the legitimate case.
#
# "opensearch", "hybrid search", "hybrid retrieval" added — jd.txt and
# jd_query.txt both name OpenSearch and hybrid retrieval explicitly as
# required infra/technique, but the old list only had "elasticsearch" and
# "dense retrieval", so a candidate whose description used the JD's own
# vocabulary for hybrid search could be missed.
ML_KEYWORDS = [
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "llm", "fine-tun", "rag", "semantic search", "sentence-transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "bert", "transformer", "nlp", "information retrieval",
    "learning to rank", "xgboost ranking", "neural ranker", "reranker",
    "dense retrieval", "hybrid search", "hybrid retrieval", "search engine",
    "knowledge graph", "question answering", "vector database",
    "approximate nearest neighbor", "hnsw", "cosine similarity",
]
# Matching requires a leading word boundary (re.search(r"\bKEYWORD", desc))
# instead of plain `kw in desc`. A *leading* boundary (not a trailing one) was
# chosen deliberately: stems like "fine-tun" and "sentence-transformer" are
# meant to also match "fine-tuning"/"sentence-transformers", so they can't
# require a boundary at the end. A leading boundary alone is enough to stop
# "llm" matching inside "fulfillment" and "bert" matching inside
# "Robert"/"Albert" (no boundary before "bert" there), while still matching
# every legitimate case (a keyword preceded by whitespace, punctuation, or
# string start).
ML_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in ML_KEYWORDS) + r")"
)


def has_ml_keyword(desc: str) -> bool:
    """True if desc (a role description, any case) contains any ML/IR
    production keyword. Used by 01_extract_features.py and 04_eval.py when
    scanning career_history."""
    return ML_KEYWORD_PATTERN.search((desc or "").lower()) is not None


# ── Title seniority mapping ─────────────────────────────────────────────────
# 5: principal/staff/head/vp/director  4: senior/lead  3: default  2: mid
# 1: junior. Single authoritative source — see ML_KEYWORDS above for why
# (04_eval.py's copy of this exact mapping had silently dropped the
# SENIORITY_MID branch despite a "must stay in sync" comment).
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
    is_primary_city: bool = False,
) -> float:
    """
    Score location fit. JD offices are Noida and Pune; India-based strongly
    preferred. Uses pre-computed boolean columns from features.parquet
    (Script 01).

    is_primary_city: Pune/Noida specifically (PRIMARY_CITIES). jd.txt names
    these as its actual offices and as the location in its own "ideal
    candidate" summary ("Located in or willing to relocate to Noida or
    Pune"). The other JD-welcomed cities (is_target_city: Hyderabad, Mumbai,
    Delhi NCR, Bengaluru, etc.) score slightly below that — not because the
    JD ranks them against each other (it doesn't), but because it explicitly
    ranks them below Pune/Noida.
    """
    if not is_india_based:
        return 0.15 if not willing_to_relocate else 0.45
    if is_primary_city:
        return 1.0
    elif is_target_city:
        return 0.88
    elif willing_to_relocate:
        return 0.78
    else:
        return 0.52


def compute_company_fit(
    entire_career_it_services: bool,
    has_product_company_exp: bool,
    has_ml_production_experience: bool,
    years_since_last_ml_role: float,
    entire_career_research_only: bool = False,
    shallow_recent_ml_only: bool = False,
) -> float:
    """
    Score company + ML production background.

    Hard disqualifiers:
      - entire career in IT services (explicit in JD)
      - entire career in pure research/academic roles with no production
        deployment (explicit in JD: "we will not move forward. We are
        explicit about this.") — previously NOT enforced here: prior to this
        fix, has_product_company_exp only excluded IT-services employers, so
        an all-academia candidate satisfied has_product_company_exp=True and
        was scored as if they had real product-company experience. See
        entire_career_research_only's definition in 01_extract_features.py
        (mirrored in 04_eval.py) for the detection heuristic.

    shallow_recent_ml_only: jd.txt — "If your 'AI experience' consists
    primarily of recent (under 12 months) projects using LangChain to call
    OpenAI — we will probably not move forward, unless you can demonstrate
    substantial pre-LLM-era ML production experience." A candidate whose only
    detected ML signal is a single recent (<=1yr) role under 12 months gets
    this flag (see 01_extract_features.py) and is scored here as if they had
    no ML production experience, rather than getting full credit for it.

    Bug 2 fix: stale ML always scores >= no ML ever.
    When ml_recency decays below 0.65, we floor at 0.65 (= product-no-ML baseline).
    This ensures "product + ML from 4 years ago" >= "product + zero ML experience".
    """
    if entire_career_it_services or entire_career_research_only:
        return 0.0   # hard disqualifier

    effective_has_ml = has_ml_production_experience and not shallow_recent_ml_only

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

    if has_product_company_exp and effective_has_ml:
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


def compute_industry_bonus(current_industry: str) -> float:
    """
    Small tiebreaker for industry background relevant to the role.

    jd.txt lists "Prior exposure to HR-tech, recruiting tech, or marketplace
    products" under "Things we'd like you to have but won't reject you for" —
    a nice-to-have, not a gate. Scoring_plan's Tier 2 #10 frames
    current_industry the same way. Kept small and purely additive: absence of
    this background is never penalized, only a match is rewarded.
    """
    industry = (current_industry or "").lower()
    if any(kw in industry for kw in INDUSTRY_RELEVANT_KEYWORDS):
        return 0.02
    return 0.0


def compute_github_bonus(github_activity_score) -> float:
    """
    Small tiebreaker for public/open-source engineering activity.

    jd.txt nice-to-have: "Open-source contributions in the AI/ML space."
    jd.txt negative signal: "People whose work has been entirely on
    closed-source proprietary systems for 5+ years without external
    validation (papers, talks, open-source)." github_activity_score is the
    only proxy available in redrob_signals for either of these — there's no
    papers/talks field to draw on.

    Assumes github_activity_score is pre-normalized to [0, 1], like the other
    redrob_signals rates (recruiter_response_rate, offer_acceptance_rate). If
    the raw field turns out to use a different scale, rescale at the call
    site before passing it in here.

    -1 sentinel (no GitHub data) -> 0.0: most legitimate engineers don't have
    public activity tied to their profile, so absence isn't itself negative —
    only presence is rewarded.
    """
    score = float(github_activity_score) if github_activity_score is not None else -1.0
    if score < 0:
        return 0.0
    return round(0.03 * max(0.0, min(1.0, score)), 4)


def compute_availability_score(
    open_to_work: bool,
    last_active_days_ago: int,
    recruiter_response_rate: float,
    avg_response_time_hours: float,
    notice_period_days: int,
    saved_by_recruiters_30d: int,
    verified_email: bool,
    verified_phone: bool,
    interview_completion_rate: float = -1.0,
    offer_acceptance_rate: float = -1.0,
) -> float:
    """
    Availability signal. Seven sub-signals + social proof + reachability.

    interview_completion_rate / offer_acceptance_rate: re-added (Scoring_plan
    Tier 3, #16-17). Both were already extracted into features.parquet but
    never consumed here — interview_completion_rate was explicitly removed as
    a "dead parameter" in an earlier pass (the I1 fix), and
    offer_acceptance_rate was extracted but never wired in at all. That
    earlier fix addressed the dead-code symptom but not the missing signal.
    Both map directly onto jd.txt's closing instruction to down-weight
    "perfect-on-paper" candidates who aren't actually reachable/available for
    hiring purposes — a candidate who repeatedly drops out of interview loops
    or declines offers is not actually available, independent of technical
    fit. -1 sentinel (no history) -> neutral 0.5, the same convention already
    used for avg_response_time_hours.

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

    # Interview completion / offer acceptance (-1 = no history → neutral)
    interview_score = 0.5 if interview_completion_rate < 0 else float(interview_completion_rate)
    offer_score = 0.5 if offer_acceptance_rate < 0 else float(offer_acceptance_rate)

    # Social proof — capped at 20
    social_proof = min(saved_by_recruiters_30d, 20) / 20.0

    # Reachability composite
    reachability = 0.5 * float(verified_email) + 0.5 * float(verified_phone)

    availability = (
        0.22 * float(open_to_work) +
        0.20 * recency +
        0.16 * float(recruiter_response_rate) +
        0.13 * notice_score +
        0.08 * response_time_score +
        0.07 * interview_score +
        0.05 * offer_score +
        0.04 * social_proof +
        0.05 * reachability
    )
    return min(1.0, max(0.0, availability))


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
    entire_research   = bool(row.get("entire_career_research_only", False))
    shallow_ml_only   = bool(row.get("shallow_recent_ml_only", False))
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

    elif entire_research:
        # Hard disqualifier — pure research/academic career, no production deployment
        s1 = (f"{title} with {yoe_str} years of experience whose career has been entirely "
              f"in research/academic roles with no detected production deployment — the JD "
              f"is explicit that this is not a fit.")

    elif shallow_ml_only:
        # Attenuated, not a hard disqualifier — recent, shallow ML/LLM signal only
        s1 = (f"{title} with {yoe_str} years of experience whose only detected AI/ML signal "
              f"is a single recent role under a year — the JD specifically flags recent "
              f"LangChain/API-only experience without substantial prior ML production work.")

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

    if entire_it or entire_research or shallow_ml_only:
        # For disqualified/attenuated candidates, cite their availability as the reason they appear at all
        if open_to_work and days_ago <= 30:
            s2 = (f"Despite this, they are actively available "
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
    industry_bonus: float = 0.0,
    github_bonus: float = 0.0,
    n_it_services_roles: int = 0,
    job_hop_score: float = 0.0,
) -> float:
    """
    Weighted structural score combining all hard-requirement signals.

    Weights reflect JD priorities:
      company background + ML experience (0.32) > location (0.25)
      > experience band (0.20) > trajectory (0.15) > salary (0.02)
      > bonuses (additive, each individually small)

    salary_fit's weight was cut from 0.08 to 0.02 (not removed — the
    compute_salary_fit() function and its 0-1 "fit" semantics are unchanged,
    so any other consumer reading its raw output directly, e.g. sandbox/app.py,
    is unaffected). jd.txt never states an actual compensation band for this
    role — the "On location, comp, and logistics" section only covers
    location and notice period, no numbers — and Scoring_plan explicitly
    places salary_min_lpa/salary_max_lpa in "Tier 8 — Ignore / Minimal
    Impact... use them only in derived features." Weighting it at 0.08 meant
    ranking against an invented number with real influence; 0.02 keeps it as
    a genuine tiebreaker. trajectory_score's weight was raised from 0.12 to
    0.15 to compensate, since it now also encodes the title-chasing/job-hop
    penalty (see 01_extract_features.py) and is more directly JD-grounded
    than salary ever was.

    industry_bonus / github_bonus: small additive tiebreakers for
    current_industry (HR-tech/recruiting/marketplace background) and
    github_activity_score (open-source signal) — both jd.txt nice-to-haves
    that were previously extracted into features.parquet but never scored.
    See compute_industry_bonus() / compute_github_bonus().

    n_it_services_roles: graduated penalty per Scoring_plan Tier 5.
    entire_career_it_services is the hard disqualifier (score=0.0 from
    company_fit), but a candidate with 2-3 IT services roles and a product
    company isn't a hard disqualifier — they should still lose ground vs a
    pure product-company background. Previously n_it_services_roles was
    extracted into features.parquet but never consumed here.
      0 roles:  no penalty
      1-2 roles: -0.02 (minor — product company present, IT is partial)
      3-4 roles: -0.04 (moderate — spent majority of career in IT services)
      5+ roles:  -0.06 (strong — effectively IT-services-dominant career)

    job_hop_score: title-chasing penalty per Scoring_plan Tier 5.
    JD explicitly disqualifies "optimizing for Senior→Staff→Principal titles
    by switching companies every 1.5 years." job_hop_score =
    years_of_experience / n_total_roles. Higher = more stable (stayed longer
    per company). Values < 1.5 years-per-company get a negative adjustment.
      >= 2.5 yr/company: no penalty (good tenure)
      1.5–2.5 yr/company: -0.02 (title-chasing risk)
      < 1.5 yr/company:  -0.05 (strong title-chasing signal per JD)

    Bug 4 fix: return min(1.0, ...) — bonuses can push the weighted sum past 1.0
    without the cap. Capped to keep all scores in [0, 1].
    """
    raw = (
        0.32 * company_fit +
        0.25 * location_fit +
        0.20 * experience_fit +
        0.15 * trajectory_score +
        0.02 * salary_fit
    )

    # Graduated IT services penalty (Scoring_plan Tier 5 — n_it_services_roles)
    if n_it_services_roles >= 5:
        it_penalty = -0.06
    elif n_it_services_roles >= 3:
        it_penalty = -0.04
    elif n_it_services_roles >= 1:
        it_penalty = -0.02
    else:
        it_penalty = 0.0

    # Job-hopping penalty (Scoring_plan Tier 5 — job_hop_score = yoe / n_total_roles)
    if job_hop_score <= 0:
        hop_penalty = 0.0     # no data — neutral
    elif job_hop_score < 1.5:
        hop_penalty = -0.05   # < 1.5 yr/company: strong title-chasing signal
    elif job_hop_score < 2.5:
        hop_penalty = -0.02   # 1.5–2.5 yr/company: moderate concern
    else:
        hop_penalty = 0.0     # >= 2.5 yr/company: stable enough

    bonuses = skill_assessment_bonus + edu_bonus + industry_bonus + github_bonus
    total = raw + bonuses + it_penalty + hop_penalty
    return min(1.0, max(0.0, total))


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
# Weighted toward dense deliberately: the JD explicitly calls out keyword-
# stuffing as a trap (jd.txt's closing note), so a confident semantic match
# should be able to outscore a borderline keyword match, not just tie with it.
FUSION_BM25_WEIGHT  = 0.35
FUSION_DENSE_WEIGHT = 0.65

# prelim_score = fusion + structural + availability (pre-cross-encoder).
# Structural's weight was dropped slightly (was 0.35) now that the dense
# fusion signal is actually allowed to differentiate (see weighted_score_fusion
# below) and now that has_ml_production_experience's keyword false-positive
# bug is fixed — structural no longer needs to carry as much of the load.
PRELIM_FUSION_WEIGHT       = 0.55
PRELIM_STRUCTURAL_WEIGHT   = 0.25
PRELIM_AVAILABILITY_WEIGHT = 0.20

# final_score = prelim_score + cross-encoder. CE gets more say than before
# (was 0.60) since it's the one stage with real contextual/semantic judgment
# (full cross-attention over JD + candidate text), vs. prelim_score which is
# still partly keyword/rule-driven.
FINAL_PRELIM_WEIGHT = 0.30
FINAL_CE_WEIGHT     = 0.70

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