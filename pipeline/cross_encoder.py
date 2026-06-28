"""
pipeline/cross_encoder.py — Stage 4: Cross-encoder reranking.

The cross-encoder (ms-marco-MiniLM-L-6-v2) performs full cross-attention
over (JD query, candidate text) pairs. It's the most expensive stage —
O(N) model inferences — so it only runs on the top-N candidates by
preliminary score.

Speed optimisations:
  1. Module-level model singleton — model is loaded once when the module is
     first imported, not on every call to rerank(). On CPU, CrossEncoder
     __init__ + from_pretrained takes ~3–8s per call; at 500 candidates that's
     pure waste. The singleton is reused across calls (e.g. multiple debug runs
     in the same process).
  2. Sort pairs by text length before batching, restore order after predict().
     batch_size pads every item in a batch to the longest item in that batch;
     grouping similar lengths cuts wasted padding compute.
  3. batch_size 64 — larger batches on CPU reduce Python loop overhead per
     item. Revert to 32 if you see OOM on machines with < 4 GB RAM.
  4. show_progress_bar=False — eliminates tqdm callback overhead (~5–10s for
     500 pairs).
"""

import numpy as np
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton — loaded once at import time, reused across calls.
# Saves 3–8s per run vs. constructing CrossEncoder inside rerank().
_model: CrossEncoder | None = None


def _get_model(model_name: str = MODEL_NAME) -> CrossEncoder:
    global _model
    if _model is None or _model.model.name_or_path != model_name:
        _model = CrossEncoder(model_name)
    return _model


def rerank(
    jd_query_text: str,
    candidate_ids: list[str],
    candidate_texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run the cross-encoder over (jd_query_text, candidate_text) pairs.

    candidate_ids and candidate_texts must be positionally aligned.

    Returns CE scores normalised to [0, 1], aligned with candidate_ids.
    """
    model = _get_model(model_name)

    # Sort by text length to minimise batch padding waste
    order = np.argsort([len(t) for t in candidate_texts])
    pairs_sorted = [(jd_query_text, candidate_texts[i]) for i in order]

    scores_sorted = model.predict(pairs_sorted, batch_size=batch_size, show_progress_bar=False)

    # Restore original order
    raw_scores = np.empty(len(candidate_ids), dtype=np.float64)
    raw_scores[order] = scores_sorted

    # Normalise to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        return (raw_scores - s_min) / (s_max - s_min)
    return np.full_like(raw_scores, 0.5)
