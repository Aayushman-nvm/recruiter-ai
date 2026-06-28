# ── Fusion stage (BM25 + Dense) ──────────────────────────────────────────────
# This JD is extremely keyword-rich (BM25, FAISS, Pinecone, NDCG, learning-to-rank,
# hybrid search, etc.). Dense handles semantics; BM25 catches exact tech terms.
# Raise BM25 slightly — a candidate who mentions "NDCG" and "hybrid retrieval"
# in their narrative should rank higher than a generic ML engineer whose embedding
# is similar but lacks the specific vocabulary.
FUSION_BM25_WEIGHT  = 0.40
FUSION_DENSE_WEIGHT = 0.60

# ── Prelim blend (fusion + structural + availability) ─────────────────────────
# Structural carries the hard qualification logic (disqualifiers, company type,
# ML recency, narrative evidence). Give it the dominant share.
# Availability is a tiebreaker, not a primary driver — keep it low.
# Fusion retrieval is still important for pulling the right candidates out of 100k.
PRELIM_FUSION_WEIGHT       = 0.40
PRELIM_STRUCTURAL_WEIGHT   = 0.48
PRELIM_AVAILABILITY_WEIGHT = 0.12

# ── Final blend (prelim + cross-encoder) ──────────────────────────────────────
# Cross-encoder is the best signal we have for semantic fit on the full JD text.
# Raise it relative to prelim — it should be able to meaningfully re-order candidates
# that looked similar in prelim.
FINAL_PRELIM_WEIGHT = 0.65
FINAL_CE_WEIGHT     = 0.35

# ── Score cap / penalty for structurally weak candidates ──────────────────────
# Candidates who fail experience fit or have stale ML should be capped lower
# so genuinely qualified candidates push past them.
SCORE_PENALTY_MULTIPLIER   = 0.78   # applied when exp_fit < floor OR ml_recency > threshold
SCORE_CAP_MAX              = 0.68   # hard ceiling after penalty — keeps weak candidates below strong
SCORE_CAP_EXP_FIT_FLOOR    = 0.75   # experience_fit below this triggers penalty (raised from 0.50)
SCORE_CAP_ML_RECENCY_YEARS = 1.5    # years since last ML role above this triggers penalty (tightened)

# ── Notice period scores ───────────────────────────────────────────────────────
# JD: "sub-30-day notice preferred; up to 30-day buyout possible; 30+ day candidates
# are still in scope but the bar gets higher." Be more aggressive penalising long notices.
NOTICE_SCORE_0_15   = 1.00
NOTICE_SCORE_16_30  = 0.88
NOTICE_SCORE_31_60  = 0.50   # meaningful penalty — JD says bar gets higher
NOTICE_SCORE_61_90  = 0.28
NOTICE_SCORE_91_120 = 0.14
NOTICE_SCORE_120P   = 0.05   # near-disqualifying

assert abs(FUSION_BM25_WEIGHT + FUSION_DENSE_WEIGHT - 1.0) < 1e-9, \
    "Fusion weights must sum to 1.0"
assert abs(PRELIM_FUSION_WEIGHT + PRELIM_STRUCTURAL_WEIGHT + PRELIM_AVAILABILITY_WEIGHT - 1.0) < 1e-9, \
    "Prelim weights must sum to 1.0"
assert abs(FINAL_PRELIM_WEIGHT + FINAL_CE_WEIGHT - 1.0) < 1e-9, \
    "Final weights must sum to 1.0"
assert 0.0 < SCORE_CAP_MAX < 1.0, \
    "Score cap must be in (0, 1)"

# ── Structural sub-component weights ─────────────────────────────────────────
# Narrative evidence (does the career history actually show ranking/retrieval/embedding
# work, not just a skills list?) is the single most important signal from the JD.
# Company fit / ML production experience is the second most important disqualifier.
# Experience years is a meaningful but not dominant factor.
# Location is logistical — important but not a quality signal.
# Trajectory and salary are weak tiebreakers.
NARRATIVE_SCORE_WEIGHT   = 0.35   # raised — narrative is the primary quality gate
COMPANY_FIT_WEIGHT       = 0.26   # ML production experience + company type
EXPERIENCE_FIT_WEIGHT    = 0.16   # years of experience band
LOCATION_FIT_WEIGHT      = 0.11   # location / relocation
TRAJECTORY_SCORE_WEIGHT  = 0.03   # trajectory tiebreaker
SALARY_FIT_WEIGHT        = 0.01   # very weak signal — dataset is synthetic

_STRUCTURAL_BASE = (
    NARRATIVE_SCORE_WEIGHT
    + COMPANY_FIT_WEIGHT
    + LOCATION_FIT_WEIGHT
    + EXPERIENCE_FIT_WEIGHT
    + TRAJECTORY_SCORE_WEIGHT
    + SALARY_FIT_WEIGHT
)
assert abs(_STRUCTURAL_BASE - 0.92) < 1e-9, \
    f"Structural base weights must sum to 0.92, got {_STRUCTURAL_BASE}"