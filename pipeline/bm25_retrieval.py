"""
pipeline/bm25_retrieval.py — Stage 1: BM25 keyword retrieval.

Uses bm25s (scipy-sparse, vectorized) rather than rank_bm25 (pure-Python
dict-based). On constrained hardware (<8 GB RAM, weak CPU) rank_bm25 can
stall for minutes and push memory into swap; bm25s is significantly faster.

BM25 uses jd_query.txt (focused tokens) rather than the full jd.txt prose
to reduce false positives from unrelated JD boilerplate.
"""

import numpy as np
import bm25s


def run_bm25(
    candidate_texts: list[str],
    candidate_ids: list[str],
    jd_query_text: str,
) -> np.ndarray:
    """
    Build a BM25 index over candidate_texts and score every candidate
    against jd_query_text.

    Returns an array of raw BM25 scores aligned with candidate_ids
    (i.e. scores[i] corresponds to candidate_ids[i]).
    """
    corpus_tokens = bm25s.tokenize(candidate_texts, stopwords=None, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)

    query_tokens = bm25s.tokenize(jd_query_text, stopwords=None, show_progress=False)

    # retrieve() returns indices/scores in descending-score order.
    # Scatter back into candidate_ids-aligned position so bm25_scores[i]
    # lines up with candidate_ids[i] — required by weighted_score_fusion.
    indices, scores_sorted = retriever.retrieve(
        query_tokens, k=len(candidate_ids), show_progress=True
    )
    bm25_scores = np.empty(len(candidate_ids), dtype=np.float64)
    bm25_scores[indices[0]] = scores_sorted[0]
    return bm25_scores
