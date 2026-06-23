"""
pipeline/cross_encoder.py — Stage 4: Cross-encoder reranking.

The cross-encoder (ms-marco-MiniLM-L-6-v2) performs full cross-attention
over (JD query, candidate text) pairs. It's the most expensive stage —
O(N) model inferences — so it only runs on the top-N candidates by
preliminary score.

Two accuracy-neutral speedups to fit more candidates in the time budget:
  1. Sort pairs by text length before batching, restore order after predict().
     batch_size pads every item in a batch to the longest item in that batch;
     grouping similar lengths cuts wasted padding compute without changing
     any individual prediction.
  2. batch_size 16 → 32. Less predictable on CPU than GPU — revert if slower.
"""

import numpy as np
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    jd_query_text: str,
    candidate_ids: list[str],
    candidate_texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run the cross-encoder over (jd_query_text, candidate_text) pairs.

    candidate_ids and candidate_texts must be positionally aligned.

    Returns CE scores normalised to [0, 1], aligned with candidate_ids.
    """
    # Speedup 1: sort by text length to minimise batch padding waste
    order = np.argsort([len(t) for t in candidate_texts])
    pairs_sorted = [(jd_query_text, candidate_texts[i]) for i in order]

    model = CrossEncoder(model_name)
    # Speedup 2: larger batch size — re-benchmark on your hardware
    scores_sorted = model.predict(pairs_sorted, batch_size=batch_size, show_progress_bar=True)

    # Restore original order
    raw_scores = np.empty(len(candidate_ids), dtype=np.float64)
    raw_scores[order] = scores_sorted

    # Normalise to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        return (raw_scores - s_min) / (s_max - s_min)
    return np.full_like(raw_scores, 0.5)
