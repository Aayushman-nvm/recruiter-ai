"""
Script 02 (v1): Candidate embedding pipeline with bge-base-en-v1.5.

Input:  data/candidates.jsonl.gz
Output:
  precomputed/candidate_embeddings.npy  — shape (N, 768), float16, ~154MB for 100K
  precomputed/candidate_ids.txt         — one CAND_XXXXXXX per line, same row order as .npy
  precomputed/candidate_texts.pkl       — list of N text strings (used by BM25 + cross-encoder)

CRITICAL ordering invariant:
  Row N in candidate_embeddings.npy == line N in candidate_ids.txt == item N in candidate_texts.pkl
  These three files are the only link between vectors and candidate identities.
  Never shuffle them independently. Never sort the JSONL before processing.
"""

import gzip
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import build_candidate_text

# bge-base-en-v1.5: 438MB, 768-dim. Stronger than all-MiniLM-L6-v2 (384-dim) on MTEB.
# BGE uses ASYMMETRIC encoding: documents use plain text (here), queries use instruction prefix (Script 03).
MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Reduced batch size vs MiniLM because bge-base is larger. Stays under ~6GB RAM.
BATCH_SIZE = 128

DATA_PATH = "data/candidates.jsonl.gz"
OUT_DIR   = "precomputed"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    candidate_ids = []
    texts = []

    print("Reading candidates...")
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing JSONL"):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            candidate_ids.append(c["candidate_id"])
            texts.append(build_candidate_text(c))

    print(f"Parsed {len(candidate_ids)} candidates. Starting embedding...")

    # BGE document encoding: plain text, no instruction prefix (asymmetric encoding)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise: dot product == cosine similarity
    )

    # float16 halves disk: 100K × 768 × 2 bytes ≈ 154MB
    emb_path = f"{OUT_DIR}/candidate_embeddings.npy"
    np.save(emb_path, embeddings.astype(np.float16))
    print(f"Saved embeddings → {emb_path}  shape={embeddings.shape}")

    ids_path = f"{OUT_DIR}/candidate_ids.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(candidate_ids))
    print(f"Saved IDs       → {ids_path}")

    texts_path = f"{OUT_DIR}/candidate_texts.pkl"
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)
    print(f"Saved texts     → {texts_path}")

    print(f"\nDone. {len(candidate_ids)} candidates embedded.")


if __name__ == "__main__":
    main()
