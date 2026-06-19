"""
Script 03: JD embedding with BGE instruction prefix.

Input:  jd_query.txt  (~170-word compact structured requirements query)
Output: precomputed/jd_embedding.npy  — shape (768,), float32

BGE ASYMMETRIC ENCODING — critical:
  Candidate documents (Script 02): plain text, no prefix.
  JD query (this script): must use the instruction prefix below.
  Without the prefix, retrieval quality degrades noticeably for BGE models.

Why jd_query.txt and not jd.txt?
  jd.txt is ~2000 words of narrative prose. Embedding the full JD averages requirements
  equally with vibe sections, logistics, and disclaimers — diluting the retrieval signal.
  jd_query.txt is ~170 focused words of requirements only. Produces a sharper vector.
  jd.txt is used by Script 04 (Ollama reasoning) and BM25 tokens in rank.py where
  full context richness helps.
"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_NAME = "all-MiniLM-L6-v2"
# BGE query-side instruction prefix (required for asymmetric encoding)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

JD_QUERY_PATH = "jd_query.txt"
OUT_PATH      = "precomputed/jd_embedding.npy"


def main():
    os.makedirs("precomputed", exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    with open(JD_QUERY_PATH, "r", encoding="utf-8") as f:
        jd_query = f.read().strip()

    print(f"JD query length: {len(jd_query.split())} words")

    query_with_prefix = BGE_QUERY_INSTRUCTION + jd_query

    jd_embedding = model.encode(
        query_with_prefix,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(OUT_PATH, jd_embedding)
    print(f"Saved JD embedding → {OUT_PATH}  shape={jd_embedding.shape}")
    # Expected: (768,)


if __name__ == "__main__":
    main()
