"""
rank.py — 4-stage candidate ranking pipeline.

Usage:
  python rank.py [--precomputed precomputed/] [--out submission.csv] [--no-reasoning]
                  [--ce-topn 500]

Hard constraints:
  ≤ 5 min wall-clock | ≤ 16 GB RAM | CPU only | no network calls

Pipeline:
  Stage 1: BM25 keyword retrieval        (bm25s, vectorized — ~2-5s for 100K)
  Stage 2: Dense cosine similarity       (~1s for 100K)
  Fusion:  weighted score fusion → shortlist top-2000  (see scoring.py)
  Stage 3: Structural + availability scoring on top-2000
  Stage 4: Cross-encoder reranker on top-N  (slowest stage on weak CPUs)
  Output:  final top-100

NOTE: Stage 1 previously used rank_bm25 (pure-Python dict-based index).
On constrained hardware (<8GB RAM, weak CPU) that stage can stall for minutes
and push memory into swap. Replaced with bm25s (scipy-sparse, vectorized,
built-in progress bars). `pip install bm25s`.

Fusion was changed from Reciprocal Rank Fusion to a weighted min-max score
fusion — RRF discards how confident each retrieval method was (rank position
only), which let borderline keyword matches tie with strong semantic matches.
See weighted_score_fusion() in scoring.py for the full rationale.

Stage 4 (--ce-topn, default 500): widened from a prior 300 back to 500 now
that the false-positive-bug fixes elsewhere freed up the time budget for it.
Cross-encoder cost is linear in N candidates and dominates total wall time, so
two accuracy-neutral speedups were added to make room instead of just eating
the extra ~110s:
  1. Pairs are sorted by text length before batching, then unsorted after
     predict() — `batch_size` pads every item in a batch to the longest item
     in that batch, so grouping similar lengths together cuts wasted padding
     compute. Does not change any individual prediction, only batch order.
  2. batch_size raised 16 -> 32 (re-benchmark on your hardware; CPU batching
     gains are less predictable than on GPU — revert if it's not faster).
If --ce-topn 500 still doesn't fit your 5-minute budget after these, dial it
down (e.g. --ce-topn 400) — no code changes needed.
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
    FINAL_CE_WEIGHT,
    FINAL_PRELIM_WEIGHT,
    PRELIM_AVAILABILITY_WEIGHT,
    PRELIM_FUSION_WEIGHT,
    PRELIM_STRUCTURAL_WEIGHT,
    compute_availability_score,
    compute_company_fit,
    compute_experience_fit,
    compute_github_bonus,
    compute_industry_bonus,
    compute_location_fit,
    compute_salary_fit,
    compute_structural_score,
    generate_reasoning,
    weighted_score_fusion,
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
# Main ranking logic
# ─────────────────────────────────────────────────────────────────────────────


def rank_candidates(
    df: pd.DataFrame,
    candidate_ids: list,
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    candidate_texts: list,
    ce_topn: int = 500,
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

    # Full ranking over all N candidates. retrieve() returns indices/scores in
    # descending-score order (not candidate_ids order) — scatter back into
    # candidate_ids-aligned position so bm25_raw[i] lines up with candidate_ids[i]
    # and dense_scores[i] below. Raw scores (not just rank position) are needed
    # for weighted_score_fusion.
    bm25_indices, bm25_scores_sorted = bm25_retriever.retrieve(
        query_tokens, k=len(candidate_ids), show_progress=True
    )
    bm25_raw = np.empty(len(candidate_ids), dtype=np.float64)
    bm25_raw[bm25_indices[0]] = bm25_scores_sorted[0]
    print(f"  BM25 done in {time.time() - t1:.1f}s. "
          f"Top candidate: {candidate_ids[bm25_indices[0][0]]}")

    # ── Stage 2: Dense cosine similarity ─────────────────────────────────────
    print("Stage 2: Dense similarity...")
    t2 = time.time()
    # Dot product works because both embeddings are L2-normalised
    dense_scores = embeddings @ jd_embedding   # shape: (N,), ~1s
    print(f"  Dense done in {time.time() - t2:.1f}s. "
          f"Top candidate: {candidate_ids[int(np.argmax(dense_scores))]}")

    # ── Fusion: weighted score fusion → shortlist top-2000 ────────────────────
    # Score-based fusion (not RRF) — see weighted_score_fusion() in scoring.py
    # for why: RRF discards how confident each method was, only who ranked #1.
    print("Fusion: weighted score fusion (BM25 + dense) → top-2000 shortlist...")
    t3 = time.time()
    fusion_scores = weighted_score_fusion(bm25_raw, dense_scores, candidate_ids)
    top2000_ids = sorted(fusion_scores.keys(), key=lambda cid: -fusion_scores[cid])[:2000]
    top2000_set = set(top2000_ids)

    df_top = df[df["candidate_id"].isin(top2000_set)].copy()
    print(f"  Shortlisted {len(df_top)} candidates in {time.time() - t3:.1f}s.")

    # ── Stage 3: Structural + availability scoring ────────────────────────────
    print("Stage 3: Structural + availability scoring...")
    t4 = time.time()

    def row_structural(row) -> float:
        exp_fit  = compute_experience_fit(row["years_of_experience"])
        loc_fit  = compute_location_fit(
            row["is_india_based"], row["is_target_city"], row["willing_to_relocate"],
            row["is_primary_city"],
        )
        comp_fit = compute_company_fit(
            row["entire_career_it_services"], row["has_product_company_exp"],
            row["has_ml_production_experience"], row["years_since_last_ml_role"],
            row["entire_career_research_only"], row["shallow_recent_ml_only"],
        )
        sal_fit  = compute_salary_fit(row["salary_min_lpa"], row["salary_max_lpa"])
        return compute_structural_score(
            exp_fit, loc_fit, comp_fit,
            row["trajectory_score"], sal_fit,
            row["skill_assessment_bonus"], row["edu_bonus"],
            compute_industry_bonus(row["current_industry"]),
            compute_github_bonus(row["github_activity_score"]),
            int(row.get("n_it_services_roles", 0) or 0),
            float(row.get("job_hop_score", 0.0) or 0.0),
        )

    def row_availability(row) -> float:
        return compute_availability_score(
            row["open_to_work_flag"],
            row["last_active_days_ago"],
            row["recruiter_response_rate"],
            row["avg_response_time_hours"],
            row["notice_period_days"],
            row["saved_by_recruiters_30d"],
            row["verified_email"],
            row["verified_phone"],
            row["interview_completion_rate"],
            row["offer_acceptance_rate"],
        )

    tqdm.pandas(desc="Structural")
    df_top["structural_score"]  = df_top.progress_apply(row_structural, axis=1)
    tqdm.pandas(desc="Availability")
    df_top["availability_score"] = df_top.progress_apply(row_availability, axis=1)

    # Attach and normalise fusion scores. weighted_score_fusion() already
    # min-max normalizes each component globally (over all 100K candidates),
    # so re-normalize here to the top-of-shortlist so the best candidate in
    # this top-2000 gets the fusion term's full weight — consistent with how
    # structural_score/availability_score are independently scaled to [0, 1].
    df_top["fusion_score"] = df_top["candidate_id"].map(fusion_scores)
    max_fusion = df_top["fusion_score"].max()
    df_top["fusion_score_norm"] = df_top["fusion_score"] / max_fusion if max_fusion > 0 else 0.0

    # Preliminary score (additive — Bug 1 fix: no multiplier, KDD docs match code)
    df_top["prelim_score"] = (
        PRELIM_FUSION_WEIGHT * df_top["fusion_score_norm"] +
        PRELIM_STRUCTURAL_WEIGHT * df_top["structural_score"] +
        PRELIM_AVAILABILITY_WEIGHT * df_top["availability_score"]
    )

    # Hard disqualifiers: zero out before cross-encoder stage
    df_top.loc[df_top["is_honeypot"],                  "prelim_score"] = 0.0
    df_top.loc[df_top["entire_career_it_services"],     "prelim_score"] = 0.0
    df_top.loc[df_top["entire_career_research_only"],   "prelim_score"] = 0.0

    print(f"  Structural scoring done in {time.time() - t4:.1f}s. Prelim top candidate: "
          f"{df_top.nlargest(1, 'prelim_score')['candidate_id'].iloc[0]}")

    # ── Stage 4: Cross-encoder reranker on top-N ──────────────────────────────
    # ce_topn widened back to 500 (was reduced to 300 purely for time budget,
    # not because 300 was structurally correct) now that the keyword/date bugs
    # fixed elsewhere freed up time. Two accuracy-neutral speedups make room
    # for the extra candidates instead of just eating the added time — see
    # module docstring for the full rationale.
    print(f"Stage 4: Cross-encoder reranking (top-{ce_topn})...")
    t5 = time.time()
    topN = df_top.nlargest(ce_topn, "prelim_score").copy()
    topN_ids = topN["candidate_id"].tolist()

    cid_to_text = dict(zip(candidate_ids, candidate_texts))
    topN_texts = [cid_to_text[cid] for cid in topN_ids]

    # Speedup 1: sort pairs by text length before batching, predict, then
    # restore original order. batch_size pads every item in a batch to that
    # batch's longest item — grouping similar lengths cuts wasted padding
    # compute without changing any individual prediction.
    order = np.argsort([len(t) for t in topN_texts])
    pairs_sorted = [(jd_query_text, topN_texts[i]) for i in order]

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    # Speedup 2: batch_size 16 -> 32. Re-benchmark on your hardware — CPU
    # batching gains are less predictable than on GPU; revert if it's slower.
    ce_scores_sorted = cross_encoder.predict(pairs_sorted, batch_size=32, show_progress_bar=True)

    ce_scores = np.empty(len(topN_ids), dtype=np.float64)
    ce_scores[order] = ce_scores_sorted

    # Normalise CE scores to [0, 1]
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    if ce_max > ce_min:
        ce_scores_norm = (ce_scores - ce_min) / (ce_max - ce_min)
    else:
        ce_scores_norm = np.ones_like(ce_scores) * 0.5

    topN["ce_score"] = ce_scores_norm

    # Blend preliminary (structural context) with cross-encoder (relevance
    # precision). CE's weight was raised (was 0.40/0.60) — it's the one stage
    # with real contextual judgment (full cross-attention over JD + candidate
    # text), vs. prelim_score which is still partly keyword/rule-driven.
    topN["final_score"] = (
        FINAL_PRELIM_WEIGHT * topN["prelim_score"] +
        FINAL_CE_WEIGHT * topN["ce_score"]
    )

    # Re-enforce hard disqualifiers (CE might score disqualified candidates highly)
    topN.loc[topN["is_honeypot"],                 "final_score"] = 0.0
    topN.loc[topN["entire_career_it_services"],    "final_score"] = 0.0
    topN.loc[topN["entire_career_research_only"],  "final_score"] = 0.0

    # Final top-100: sort by final_score desc, candidate_id asc (deterministic tiebreak)
    top100 = topN.nlargest(100, "final_score").copy()
    top100 = top100.sort_values(["final_score", "candidate_id"], ascending=[False, True])
    top100["rank"] = range(1, 101)
    top100["score"] = top100["final_score"].round(6)

    elapsed_ce = time.time() - t5
    print(f"  Cross-encoder done in {elapsed_ce:.1f}s "
          f"({elapsed_ce / max(1, len(topN_ids)):.3f}s/candidate). "
          f"Top candidate: {top100.iloc[0]['candidate_id']}")
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
    parser.add_argument("--ce-topn", type=int, default=500,
                        help="How many candidates (by prelim_score) the cross-encoder "
                             "reranks in Stage 4. Dial this down (e.g. 400/350) if your "
                             "hardware can't fit 500 in the 5-minute budget — no code "
                             "changes needed.")
    args = parser.parse_args()

    print("Loading pre-computed data...")
    df, candidate_ids, embeddings, jd_embedding, candidate_texts = load_precomputed(args.precomputed)
    print(f"  {len(candidate_ids)} candidates loaded. Embeddings shape: {embeddings.shape}")

    print("\nRanking candidates...")
    top100 = rank_candidates(df, candidate_ids, embeddings, jd_embedding, candidate_texts,
                              ce_topn=args.ce_topn)

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