"""
pipeline/fusion.py — Score-based fusion of BM25 and dense retrieval.

Replaces Reciprocal Rank Fusion (RRF). RRF converts both rankings to rank
positions before combining (1/(k+rank+1) per list), discarding how confident
each method actually was. A candidate with dense cosine 0.95 and one with
cosine 0.32 get identical credit if they're both rank #1 in their list.

This module uses min-max normalised weighted sum fusion instead, so how
confident each method was directly affects the fused score, not just
whether it ranked first. This matters specifically for this JD because
keyword overlap is an explicit trap (jd.txt closing note) — a strong
semantic match must be able to outscore a borderline keyword match.

rrf_fusion() is kept for backward compatibility (sandbox/app.py may use it).
"""

import numpy as np

# Weights imported from config via scoring.py constants re-exported there.
# Direct import from scoring here would create a circular dependency since
# scoring.py itself imports from config/. Use the canonical values directly.

# Canonical fusion weights — kept here rather than repeated in rank.py / 04_eval.py.
# Dense is weighted higher because the JD explicitly calls out keyword-stuffing
# as a trap, so a confident semantic match should outscore a borderline keyword match.
FUSION_BM25_WEIGHT  = 0.35
FUSION_DENSE_WEIGHT = 0.65

assert abs(FUSION_BM25_WEIGHT + FUSION_DENSE_WEIGHT - 1.0) < 1e-9


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def weighted_score_fusion(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    candidate_ids: list[str],
    bm25_weight: float = FUSION_BM25_WEIGHT,
    dense_weight: float = FUSION_DENSE_WEIGHT,
) -> dict[str, float]:
    """
    Min-max normalise each score distribution then take a weighted sum.

    bm25_scores / dense_scores must be positionally aligned with candidate_ids
    (bm25_scores[i] and dense_scores[i] both refer to candidate_ids[i]).

    Returns {candidate_id: fused_score} for all candidates.
    """
    bm25_arr  = np.asarray(bm25_scores,  dtype=np.float64)
    dense_arr = np.asarray(dense_scores, dtype=np.float64)
    fused = bm25_weight * _minmax(bm25_arr) + dense_weight * _minmax(dense_arr)
    return dict(zip(candidate_ids, fused.tolist()))


def rrf_fusion(
    rank_list_a: list[str],
    rank_list_b: list[str],
    k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion — kept for backward compatibility.
    New code should use weighted_score_fusion() instead.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(rank_list_a):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(rank_list_b):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores
