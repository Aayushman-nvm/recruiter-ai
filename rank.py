"""
rank.py — 4-stage candidate ranking pipeline.

Usage:
  python rank.py [--precomputed precomputed/] [--out submission.csv] [--no-reasoning]

Hard constraints:
  ≤ 5 min wall-clock | ≤ 16 GB RAM | CPU only | no network calls

Pipeline:
  Stage 1: BM25 keyword retrieval   (bm25s, vectorized — ~2-5s for 100K)
  Stage 2: Dense cosine similarity  (~1s for 100K)
  Fusion:  RRF → shortlist top-2000
  Stage 3: Structural + availability scoring on top-2000
  Stage 4: Cross-encoder reranker on top-500  (slowest stage on weak CPUs, ~60-180s)
  Output:  final top-100

NOTE: Stage 1 previously used rank_bm25 (pure-Python dict-based index).
On constrained hardware (<8GB RAM, weak CPU) that stage can stall for minutes
and push memory into swap. Replaced with bm25s (scipy-sparse, vectorized,
built-in progress bars). `pip install bm25s`.
"""

import argparse
import pickle
import time

import bm25s
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from scoring import (
    compute_availability_score,
    compute_company_fit,
    compute_experience_fit,
    compute_location_fit,
    compute_salary_fit,
    compute_structural_score,
    generate_reasoning,
)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_precomputed(precomputed_dir: str):
    """Load all pre-computed artefacts from the precomputed/ directory."""
    df = pd.read_parquet(f"{precomputed_dir}/features.parquet")

    with open(f"{precomputed_dir}/candidate_ids.txt") as f:
        candidate_ids = f.read().strip().split("\n")

    # Load as float32 for computation (stored as float16 to save disk)
    embeddings = np.load(f"{precomputed_dir}/candidate_embeddings.npy").astype(np.float32)

    jd_embedding = np.load(f"{precomputed_dir}/jd_embedding.npy").astype(np.float32)

    with open(f"{precomputed_dir}/candidate_texts.pkl", "rb") as f:
        candidate_texts = pickle.load(f)

    return df, candidate_ids, embeddings, jd_embedding, candidate_texts


# ─────────────────────────────────────────────────────────────────────────────
# RRF fusion
# ─────────────────────────────────────────────────────────────────────────────


def rrf_fusion(rank_list_a: list, rank_list_b: list, k: int = 60) -> dict:
    """
    Reciprocal Rank Fusion.
    Combines BM25 and dense rankings without normalising heterogeneous score distributions.
    k=60 is the standard default from the original RRF paper.
    """
    scores = {}
    for rank, cid in enumerate(rank_list_a):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(rank_list_b):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main ranking logic
# ─────────────────────────────────────────────────────────────────────────────


def rank_candidates(
    df: pd.DataFrame,
    candidate_ids: list,
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    candidate_texts: list,
) -> pd.DataFrame:
    """
    Run the 4-stage ranking pipeline.
    Returns a DataFrame of the top-100 candidates with scores.
    """

    # ── Stage 1: BM25 keyword retrieval ──────────────────────────────────────
    print("Stage 1: BM25 (bm25s)...")
    t1 = time.time()

    # bm25s: scipy-sparse + vectorized scoring. Built-in tqdm progress bars
    # (show_progress=True below) replace the old silent rank_bm25 indexing step.
    corpus_tokens = bm25s.tokenize(candidate_texts, stopwords=None, show_progress=True)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens, show_progress=True)

    # BM25 uses jd_query.txt — focused tokens reduce false positives vs full JD prose
    with open("jd_query.txt", encoding="utf-8") as f:
        jd_query_text = f.read().strip()
    query_tokens = bm25s.tokenize(jd_query_text, stopwords=None, show_progress=False)

    # Full ranking over all N candidates (needed for RRF rank positions below)
    bm25_indices, _bm25_scores = bm25_retriever.retrieve(
        query_tokens, k=len(candidate_ids), show_progress=True
    )
    bm25_ranked = [candidate_ids[i] for i in bm25_indices[0]]
    print(f"  BM25 done in {time.time() - t1:.1f}s. Top candidate: {bm25_ranked[0]}")

    # ── Stage 2: Dense cosine similarity ─────────────────────────────────────
    print("Stage 2: Dense similarity...")
    t2 = time.time()
    # Dot product works because both embeddings are L2-normalised
    dense_scores = embeddings @ jd_embedding   # shape: (N,), ~1s
    dense_ranked = [candidate_ids[i] for i in np.argsort(-dense_scores)]
    print(f"  Dense done in {time.time() - t2:.1f}s. Top candidate: {dense_ranked[0]}")

    # ── Fusion: RRF → shortlist top-2000 ─────────────────────────────────────
    print("Fusion: RRF → top-2000 shortlist...")
    t3 = time.time()
    rrf_scores = rrf_fusion(bm25_ranked, dense_ranked)
    top2000_ids = sorted(rrf_scores.keys(), key=lambda cid: -rrf_scores[cid])[:2000]
    top2000_set = set(top2000_ids)

    df_top = df[df["candidate_id"].isin(top2000_set)].copy()
    print(f"  Shortlisted {len(df_top)} candidates in {time.time() - t3:.1f}s.")

    # ── Stage 3: Structural + availability scoring ────────────────────────────
    print("Stage 3: Structural + availability scoring...")
    t4 = time.time()

    def row_structural(row) -> float:
        exp_fit  = compute_experience_fit(row["years_of_experience"])
        loc_fit  = compute_location_fit(
            row["is_india_based"], row["is_target_city"], row["willing_to_relocate"]
        )
        comp_fit = compute_company_fit(
            row["entire_career_it_services"], row["has_product_company_exp"],
            row["has_ml_production_experience"], row["years_since_last_ml_role"]
        )
        sal_fit  = compute_salary_fit(row["salary_min_lpa"], row["salary_max_lpa"])
        return compute_structural_score(
            exp_fit, loc_fit, comp_fit,
            row["trajectory_score"], sal_fit,
            row["skill_assessment_bonus"], row["edu_bonus"],
        )

    def row_availability(row) -> float:
        # interview_completion_rate intentionally excluded — removed from signature (I1 fix)
        return compute_availability_score(
            row["open_to_work_flag"],
            row["last_active_days_ago"],
            row["recruiter_response_rate"],
            row["avg_response_time_hours"],
            row["notice_period_days"],
            row["saved_by_recruiters_30d"],
            row["verified_email"],
            row["verified_phone"],
        )

    tqdm.pandas(desc="Structural")
    df_top["structural_score"]  = df_top.progress_apply(row_structural, axis=1)
    tqdm.pandas(desc="Availability")
    df_top["availability_score"] = df_top.progress_apply(row_availability, axis=1)

    # Attach and normalise RRF scores
    df_top["rrf_score"] = df_top["candidate_id"].map(rrf_scores)
    max_rrf = df_top["rrf_score"].max()
    df_top["rrf_score_norm"] = df_top["rrf_score"] / max_rrf if max_rrf > 0 else 0.0

    # Preliminary score (additive — Bug 1 fix: no multiplier, KDD docs match code)
    df_top["prelim_score"] = (
        0.50 * df_top["rrf_score_norm"] +
        0.35 * df_top["structural_score"] +
        0.15 * df_top["availability_score"]
    )

    # Hard disqualifiers: zero out before cross-encoder stage
    df_top.loc[df_top["is_honeypot"],                "prelim_score"] = 0.0
    df_top.loc[df_top["entire_career_it_services"],   "prelim_score"] = 0.0

    print(f"  Structural scoring done in {time.time() - t4:.1f}s. Prelim top candidate: "
          f"{df_top.nlargest(1, 'prelim_score')['candidate_id'].iloc[0]}")

    # ── Stage 4: Cross-encoder reranker on top-300 ────────────────────────────
    # Reduced from top-500 to top-300 to keep Stage 4 under ~90s on constrained CPU.
    # NDCG@10 impact is negligible — the top 100 are well within the top 300 after
    # structural scoring, and the CE is most valuable in the top 10–50 range anyway.
    print("Stage 4: Cross-encoder reranking (top-300)...")
    t5 = time.time()
    top300 = df_top.nlargest(300, "prelim_score").copy()
    top300_ids = top300["candidate_id"].tolist()

    cid_to_text = dict(zip(candidate_ids, candidate_texts))
    pairs = [(jd_query_text, cid_to_text[cid]) for cid in top300_ids]

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    ce_scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=True)

    # Normalise CE scores to [0, 1]
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    if ce_max > ce_min:
        ce_scores_norm = (ce_scores - ce_min) / (ce_max - ce_min)
    else:
        ce_scores_norm = np.ones_like(ce_scores) * 0.5

    top300 = top300.copy()
    top300["ce_score"] = ce_scores_norm

    # Blend preliminary (structural context) with cross-encoder (relevance precision)
    top300["final_score"] = (
        0.40 * top300["prelim_score"] +
        0.60 * top300["ce_score"]
    )

    # Re-enforce hard disqualifiers (CE might score disqualified candidates highly)
    top300.loc[top300["is_honeypot"],               "final_score"] = 0.0
    top300.loc[top300["entire_career_it_services"],  "final_score"] = 0.0

    # Final top-100: sort by final_score desc, candidate_id asc (deterministic tiebreak)
    top100 = top300.nlargest(100, "final_score").copy()
    top100 = top100.sort_values(["final_score", "candidate_id"], ascending=[False, True])
    top100["rank"] = range(1, 101)
    top100["score"] = top100["final_score"].round(6)

    print(f"  Cross-encoder done in {time.time() - t5:.1f}s. Top candidate: {top100.iloc[0]['candidate_id']}")
    print(f"  Total ranking time: {time.time() - t1:.1f}s")
    return top100


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generation (deterministic — no LLM required)
# ─────────────────────────────────────────────────────────────────────────────


def attach_reasoning(
    top100: pd.DataFrame,
    precomputed_dir: str,
    no_reasoning: bool,
) -> pd.DataFrame:
    """
    Generate reasoning for each of the top-100 candidates.

    Reasoning is produced deterministically from features.parquet columns
    using generate_reasoning() in scoring.py. No Ollama, no network calls.

    --no-reasoning flag: skip generation, write top100_ids.txt (used by old
    Script 04 workflow — kept for backward compatibility), return empty strings.
    """
    top100 = top100.copy()
    ids = top100["candidate_id"].tolist()

    if no_reasoning:
        with open(f"{precomputed_dir}/top100_ids.txt", "w") as f:
            f.write("\n".join(ids))
        print(f"Wrote {precomputed_dir}/top100_ids.txt")
        top100["reasoning"] = ""
        return top100

    print("Generating reasoning from feature signals (no LLM)...")
    top100["reasoning"] = top100.apply(
        lambda row: generate_reasoning(row.to_dict()), axis=1
    )
    n_populated = top100["reasoning"].str.len().gt(0).sum()
    print(f"  Generated reasoning for {n_populated}/100 candidates.")
    return top100


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    parser.add_argument("--precomputed", default="precomputed/", help="Path to precomputed/ dir")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Skip reasoning (writes top100_ids.txt for Script 04)")
    args = parser.parse_args()

    print("Loading pre-computed data...")
    df, candidate_ids, embeddings, jd_embedding, candidate_texts = load_precomputed(args.precomputed)
    print(f"  {len(candidate_ids)} candidates loaded. Embeddings shape: {embeddings.shape}")

    print("\nRanking candidates...")
    top100 = rank_candidates(df, candidate_ids, embeddings, jd_embedding, candidate_texts)

    print("\nAttaching reasoning...")
    top100 = attach_reasoning(top100, args.precomputed, args.no_reasoning)

    output = top100[["candidate_id", "rank", "score", "reasoning"]]
    output.to_csv(args.out, index=False)
    print(f"\nWritten {len(output)} rows → {args.out}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert len(output) == 100, f"Expected 100 rows, got {len(output)}"
    assert output["rank"].tolist() == list(range(1, 101)), "Ranks must be 1–100 exactly"
    assert output["candidate_id"].nunique() == 100, "candidate_ids must be unique"
    scores = output["score"].tolist()
    assert all(scores[i] >= scores[i + 1] for i in range(99)), \
        "Scores must be monotonically non-increasing"
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()