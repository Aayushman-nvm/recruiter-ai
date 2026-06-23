"""
rank.py — 4-stage candidate ranking pipeline.

Usage:
  python rank.py [--precomputed precomputed/] [--out submission.csv]
                 [--no-reasoning] [--ce-topn 500]

Hard constraints: ≤5 min | ≤16 GB RAM | CPU only | no network calls

Pipeline stages (each in its own module under pipeline/):
  Stage 1: BM25 keyword retrieval        pipeline/bm25_retrieval.py
  Stage 2: Dense cosine similarity       pipeline/dense_retrieval.py
  Fusion:  weighted score fusion         pipeline/fusion.py
  Stage 3: Structural + availability     scoring.py
  Stage 4: Cross-encoder reranking       pipeline/cross_encoder.py
"""

import argparse
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.bm25_retrieval  import run_bm25
from pipeline.dense_retrieval import compute_dense_scores
from pipeline.fusion          import weighted_score_fusion
from pipeline.cross_encoder   import rerank
from scoring import (
    FINAL_CE_WEIGHT, FINAL_PRELIM_WEIGHT,
    PRELIM_AVAILABILITY_WEIGHT, PRELIM_FUSION_WEIGHT, PRELIM_STRUCTURAL_WEIGHT,
    compute_availability_score, compute_company_fit, compute_experience_fit,
    compute_github_bonus, compute_industry_bonus, compute_location_fit,
    compute_salary_fit, compute_structural_score, generate_reasoning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_precomputed(precomputed_dir: str):
    df = pd.read_parquet(f"{precomputed_dir}/features.parquet")
    with open(f"{precomputed_dir}/candidate_ids.txt") as f:
        candidate_ids = f.read().strip().split("\n")
    embeddings   = np.load(f"{precomputed_dir}/candidate_embeddings.npy").astype(np.float32)
    jd_embedding = np.load(f"{precomputed_dir}/jd_embedding.npy").astype(np.float32)
    with open(f"{precomputed_dir}/candidate_texts.pkl", "rb") as f:
        candidate_texts = pickle.load(f)
    return df, candidate_ids, embeddings, jd_embedding, candidate_texts


# ─────────────────────────────────────────────────────────────────────────────
# Structural / availability row scorers
# ─────────────────────────────────────────────────────────────────────────────

def _row_structural(row) -> float:
    return compute_structural_score(
        compute_experience_fit(row["years_of_experience"]),
        compute_location_fit(
            row["is_india_based"], row["is_target_city"],
            row["willing_to_relocate"], row["is_primary_city"],
        ),
        compute_company_fit(
            row["entire_career_it_services"], row["has_product_company_exp"],
            row["has_ml_production_experience"], row["years_since_last_ml_role"],
            row["entire_career_research_only"], row["shallow_recent_ml_only"],
        ),
        row["trajectory_score"],
        compute_salary_fit(row["salary_min_lpa"], row["salary_max_lpa"]),
        row["skill_assessment_bonus"],
        row["edu_bonus"],
        compute_industry_bonus(row["current_industry"]),
        compute_github_bonus(row["github_activity_score"]),
        int(row.get("n_it_services_roles", 0) or 0),
        float(row.get("job_hop_score", 0.0) or 0.0),
    )


def _row_availability(row) -> float:
    return compute_availability_score(
        row["open_to_work_flag"],       row["last_active_days_ago"],
        row["recruiter_response_rate"], row["avg_response_time_hours"],
        row["notice_period_days"],      row["saved_by_recruiters_30d"],
        row["verified_email"],          row["verified_phone"],
        row["interview_completion_rate"], row["offer_acceptance_rate"],
    )


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

    with open("jd_query.txt", encoding="utf-8") as f:
        jd_query_text = f.read().strip()

    # Stage 1: BM25
    print("Stage 1: BM25...")
    t0 = time.time()
    bm25_scores = run_bm25(candidate_texts, candidate_ids, jd_query_text)
    print(f"  BM25 done in {time.time() - t0:.1f}s.")

    # Stage 2: Dense
    print("Stage 2: Dense similarity...")
    t1 = time.time()
    dense_scores = compute_dense_scores(embeddings, jd_embedding)
    print(f"  Dense done in {time.time() - t1:.1f}s.")

    # Fusion → top-2000 shortlist
    print("Fusion: weighted score fusion → top-2000...")
    t2 = time.time()
    fusion_scores = weighted_score_fusion(bm25_scores, dense_scores, candidate_ids)
    top2000_ids   = sorted(fusion_scores, key=lambda c: -fusion_scores[c])[:2000]
    df_top = df[df["candidate_id"].isin(set(top2000_ids))].copy()
    print(f"  Shortlisted {len(df_top)} in {time.time() - t2:.1f}s.")

    # Stage 3: Structural + availability
    print("Stage 3: Structural + availability scoring...")
    t3 = time.time()
    tqdm.pandas(desc="Structural")
    df_top["structural_score"]  = df_top.progress_apply(_row_structural, axis=1)
    tqdm.pandas(desc="Availability")
    df_top["availability_score"] = df_top.progress_apply(_row_availability, axis=1)

    df_top["fusion_score"]      = df_top["candidate_id"].map(fusion_scores)
    max_fusion                  = df_top["fusion_score"].max()
    df_top["fusion_score_norm"] = df_top["fusion_score"] / max_fusion if max_fusion > 0 else 0.0

    df_top["prelim_score"] = (
        PRELIM_FUSION_WEIGHT       * df_top["fusion_score_norm"] +
        PRELIM_STRUCTURAL_WEIGHT   * df_top["structural_score"]  +
        PRELIM_AVAILABILITY_WEIGHT * df_top["availability_score"]
    )
    # Hard disqualifiers
    for col in ("is_honeypot", "entire_career_it_services", "entire_career_research_only"):
        df_top.loc[df_top[col], "prelim_score"] = 0.0

    print(f"  Structural done in {time.time() - t3:.1f}s.")

    # Stage 4: Cross-encoder
    print(f"Stage 4: Cross-encoder (top-{ce_topn})...")
    t4 = time.time()
    topN    = df_top.nlargest(ce_topn, "prelim_score").copy()
    topN_ids   = topN["candidate_id"].tolist()
    cid_to_text = dict(zip(candidate_ids, candidate_texts))
    topN_texts  = [cid_to_text[cid] for cid in topN_ids]

    ce_scores_norm = rerank(jd_query_text, topN_ids, topN_texts)
    topN["ce_score"] = ce_scores_norm

    topN["final_score"] = (
        FINAL_PRELIM_WEIGHT * topN["prelim_score"] +
        FINAL_CE_WEIGHT     * topN["ce_score"]
    )
    for col in ("is_honeypot", "entire_career_it_services", "entire_career_research_only"):
        topN.loc[topN[col], "final_score"] = 0.0

    print(f"  Cross-encoder done in {time.time() - t4:.1f}s.")

    top100 = topN.nlargest(100, "final_score").copy()
    top100 = top100.sort_values(["final_score", "candidate_id"], ascending=[False, True])
    top100["rank"]  = range(1, 101)
    top100["score"] = top100["final_score"].round(6)
    return top100


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning + output
# ─────────────────────────────────────────────────────────────────────────────

def attach_reasoning(top100: pd.DataFrame, precomputed_dir: str, no_reasoning: bool) -> pd.DataFrame:
    top100 = top100.copy()
    if no_reasoning:
        with open(f"{precomputed_dir}/top100_ids.txt", "w") as f:
            f.write("\n".join(top100["candidate_id"].tolist()))
        top100["reasoning"] = ""
        return top100
    print("Generating reasoning...")
    top100["reasoning"] = top100.apply(lambda r: generate_reasoning(r.to_dict()), axis=1)
    return top100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed", default="precomputed/")
    parser.add_argument("--out",         default="submission.csv")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--ce-topn", type=int, default=500)
    args = parser.parse_args()

    print("Loading pre-computed data...")
    df, candidate_ids, embeddings, jd_embedding, candidate_texts = load_precomputed(args.precomputed)
    print(f"  {len(candidate_ids)} candidates. Embeddings: {embeddings.shape}")

    top100 = rank_candidates(df, candidate_ids, embeddings, jd_embedding,
                             candidate_texts, ce_topn=args.ce_topn)
    top100 = attach_reasoning(top100, args.precomputed, args.no_reasoning)

    output = top100[["candidate_id", "rank", "score", "reasoning"]]
    output.to_csv(args.out, index=False)
    print(f"Written {len(output)} rows → {args.out}")

    assert len(output) == 100
    assert output["rank"].tolist() == list(range(1, 101))
    assert output["candidate_id"].nunique() == 100
    scores = output["score"].tolist()
    assert all(scores[i] >= scores[i + 1] for i in range(99))
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
