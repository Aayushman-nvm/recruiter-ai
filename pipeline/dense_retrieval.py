"""
pipeline/dense_retrieval.py — Stage 2: Dense cosine similarity.

Loads pre-computed candidate embeddings and JD embedding from the
precomputed/ directory. Both are L2-normalised, so a dot product == cosine.

The embedding model used at pre-computation time is all-MiniLM-L6-v2
(see scripts/02_embed_candidates.py and scripts/03_embed_jd.py).
If you change the model, re-run those scripts before ranking — stale
embeddings produce silently wrong dense scores.
"""

import numpy as np


def compute_dense_scores(
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between every candidate embedding and the
    JD embedding.

    embeddings : shape (N, D), float32, L2-normalised
    jd_embedding: shape (D,),  float32, L2-normalised

    Returns shape (N,) float64 scores aligned with the embeddings row order.
    Dot product is used because both sides are already L2-normalised.
    """
    return (embeddings @ jd_embedding).astype(np.float64)
