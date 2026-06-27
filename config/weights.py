"""
config/weights.py — All pipeline blend weights and score caps in one place.

WHY THIS FILE EXISTS
--------------------
Previously, weights were split between scoring.py (PRELIM_*, FINAL_*) and
pipeline/fusion.py (FUSION_BM25_WEIGHT, FUSION_DENSE_WEIGHT). Both files
imported from config/ for everything else, but had no home for their own
constants. This created a circular-dependency risk: pipeline/fusion.py
couldn't import from scoring.py (scoring.py imports from pipeline/fusion.py),
so the fusion weights were duplicated. Moving everything here breaks the
cycle — any module that needs weights imports from config.weights directly,
with no cross-module dependency.

WEIGHT RATIONALE
----------------
FUSION weights:
  Dense (0.65) > BM25 (0.35): the JD explicitly warns that keyword-stuffing
  is a trap; a confident semantic match must outscore a borderline keyword hit.

PRELIM weights (REBALANCED — see also the inline comment at PRELIM_FUSION_WEIGHT):
  Structural (0.48) now dominates over fusion (0.40). Fusion measures textual
  similarity to the JD (BM25 keyword overlap + dense cosine) — it does NOT
  encode fit, and the JD explicitly warns keyword-stuffing is a trap. At the
  previous 0.53/0.35 split, a keyword-dense profile could outscore a candidate
  structural.py had heavily penalised (disqualifying language ×0.50, ghost
  skills ×0.45, etc.) because fusion's larger weight absorbed the penalty.
  Structural now carries the larger share so disqualifiers and fit signals
  actually move prelim ranking, not just nudge it. Availability (0.12)
  unchanged — it's a tiebreaker, not a driver (see AVAILABILITY WEIGHT note
  below).

FINAL weights:
  CE weight 0.25 → 0.30; prelim 0.75 → 0.70.
  At 0.25 the CE was effectively neutral but still caused score compression
  in the upper band, collapsing separation between the top candidates.
  0.30 gives enough CE signal to differentiate within the structural tier
  without letting a keyword-stuffed profile override structural disqualifiers.
  The cap (SCORE_CAP_MAX) ensures CE cannot rescue over-experience candidates.

SCORE PENALTIES + CAP
---------------------
Why not just cap? A pure cap collapses all breaching candidates to the same
ceiling score, losing differentiation between them. A 16yr candidate with
great availability and one with poor availability become identical at 0.74.
A penalty multiplier first compresses the score proportionally (preserving
relative ordering within the penalised group), then a ceiling prevents the
CE from rescuing anyone above the structural threshold.

Two independent conditions trigger penalty + cap (either is sufficient):
  1. experience_fit < 0.50  — >15 yrs total, JD explicitly flags hands-on risk
  2. years_since_last_ml_role > 2  — stale ML; JD says 18 months without
     production code is disqualifying (2yr gives a small buffer)

Penalty multiplier: 0.82 applied to the raw blended score BEFORE capping.
This means a raw score of 0.88 becomes 0.88 * 0.82 = 0.722 — already under
the cap naturally if the candidate was marginal. A raw 0.95 becomes 0.779,
then capped to 0.74. Candidates within the penalised group still separate.

Cap value: 0.74 — chosen so even the best penalised candidate lands BELOW
the typical active-ML 5-7yr product-company range (~0.76-0.85 in sub9).

NOTICE PERIOD
-------------
Softened the 90-day cliff. Previous: notice > 90 days → score 0.15.
Updated: 90–120 days → 0.25, > 120 days → 0.15.
JD says "30+ day candidates still in scope but the bar gets higher" — a flat
0.15 for anything over 90 days was too aggressive.

AVAILABILITY WEIGHT (0.15) vs original (0.20)
----------------------------------------------
Eval config E (D + availability) scored NDCG@10 = 0.7168, lower than
config D (0.764). Availability adds noise at high weight because platform
engagement correlates poorly with actual ML competency.
"""

# ── Narrative scoring ─────────────────────────────────────────────────────────
NARRATIVE_SCORE_WEIGHT = 0.35   # narrative quality contribution within structural score

# ── Retrieval fusion ──────────────────────────────────────────────────────────
FUSION_BM25_WEIGHT  = 0.35
FUSION_DENSE_WEIGHT = 0.65

# ── Preliminary score (pre-cross-encoder) ─────────────────────────────────────
# REBALANCED — fusion lowered 0.53 → 0.40, structural raised 0.35 → 0.45.
#
# WHY: fusion measures textual similarity (BM25 keyword overlap + dense cosine
# to the JD), not actual fit. jd.txt explicitly warns the "right answer" is not
# "whose skills section contains the most AI keywords" — but at fusion=0.53 vs
# structural=0.35, a keyword-dense profile could still outscore a structurally
# disqualified one, because structural's hard-zero/multiplier penalties
# (disqualifying language ×0.50, ghost-skill ×0.45, etc.) only act on the
# 0.35-weighted term. A candidate whose narrative explicitly admits deployment
# was owned by another team — a near-disqualifier per the JD — was observed
# landing at rank #1 in submission_v15 because fusion's 0.53 weight absorbed
# the structural penalty almost entirely (see rank.py R2 gate fix, same date).
# Structural now carries the larger share so its penalties actually move
# the needle, consistent with "skills/fit > notice > location" as the intended
# priority order.
PRELIM_FUSION_WEIGHT       = 0.40   # lowered from 0.53 — was swamping structural penalties
PRELIM_STRUCTURAL_WEIGHT   = 0.48   # raised from 0.35 — fit + disqualifiers need to actually bite
PRELIM_AVAILABILITY_WEIGHT = 0.12   # unchanged — availability is a tiebreaker, not a driver

# ── Final score (after cross-encoder) ────────────────────────────────────────
FINAL_PRELIM_WEIGHT = 0.70   # lowered from 0.75 — allows CE to add light tiebreaking
FINAL_CE_WEIGHT     = 0.30   # raised from 0.25 — more within-tier separation

# ── Score penalty + cap thresholds ───────────────────────────────────────────
# Candidates who breach EITHER condition get penalised first, then capped.
# Penalty preserves relative ordering within the penalised group.
# Cap prevents CE from rescuing anyone above the structural ceiling.
SCORE_PENALTY_MULTIPLIER   = 0.82   # applied to raw blended score before cap
SCORE_CAP_MAX              = 0.74   # ceiling after penalty — below active 5-7yr range
SCORE_CAP_EXP_FIT_FLOOR    = 0.50   # experience_fit below this → triggers (>15 yrs)
SCORE_CAP_ML_RECENCY_YEARS = 2.0    # years_since_last_ml_role above this → triggers

# ── Notice period score tiers ─────────────────────────────────────────────────
# Used in compute_availability_score. Stored here so they're visible alongside
# the other weight decisions rather than buried in a long scoring function.
#
# JD says: "We'd love sub-30-day notice. We can buy out up to 30 days.
# 30+ day notice candidates are still in scope but the bar gets higher."
# Tiers reflect that 0–30 days = preferred band, 30+ is real friction.
NOTICE_SCORE_0_15   = 1.00
NOTICE_SCORE_16_30  = 0.90   # within JD buyout window — still preferred
NOTICE_SCORE_31_60  = 0.55   # tightened from 0.60 — "bar gets higher" starts here
NOTICE_SCORE_61_90  = 0.30   # tightened from 0.35 — meaningful delay
NOTICE_SCORE_91_120 = 0.18   # tightened from 0.25 — 3-4 month wait is a real blocker
NOTICE_SCORE_120P   = 0.08   # tightened from 0.15 — > 120 days: near-disqualifying

# ── Sanity checks — catch typos at import time ────────────────────────────────
assert abs(FUSION_BM25_WEIGHT + FUSION_DENSE_WEIGHT - 1.0) < 1e-9, \
    "Fusion weights must sum to 1.0"
assert abs(PRELIM_FUSION_WEIGHT + PRELIM_STRUCTURAL_WEIGHT + PRELIM_AVAILABILITY_WEIGHT - 1.0) < 1e-9, \
    "Prelim weights must sum to 1.0"
assert abs(FINAL_PRELIM_WEIGHT + FINAL_CE_WEIGHT - 1.0) < 1e-9, \
    "Final weights must sum to 1.0"
assert 0.0 < SCORE_CAP_MAX < 1.0, \
    "Score cap must be in (0, 1)"

# Structural base weights sum check (narrative + company + location + experience + trajectory + salary)
_STRUCTURAL_BASE = (
    NARRATIVE_SCORE_WEIGHT   # 0.35
    + 0.22                   # company_fit
    + 0.25                   # location_fit
    + 0.20                   # experience_fit
    + 0.05                   # trajectory_score
    + 0.02                   # salary_fit
)
assert abs(_STRUCTURAL_BASE - 1.09) < 1e-9, \
    f"Structural base weights must sum to 1.09, got {_STRUCTURAL_BASE}"