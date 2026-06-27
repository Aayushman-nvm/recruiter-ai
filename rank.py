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

DEBUG DUMPS
  When --debug is passed, the top-100 ranking after each stage is written
  to bin/<run_id>/stage_N_<name>.csv so you can inspect how the ordering
  changes through the pipeline.

  ⚠️  SUBMISSION WARNING: --debug writes extra files to bin/. Do NOT pass
  --debug when producing a competition submission. The flag is off by default
  so a plain `python rank.py` is always submission-safe.
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.bm25_retrieval  import run_bm25
from pipeline.dense_retrieval import compute_dense_scores
from pipeline.fusion          import weighted_score_fusion
from pipeline.cross_encoder   import rerank
from config.weights import (
    PRELIM_AVAILABILITY_WEIGHT,
    PRELIM_FUSION_WEIGHT,
    PRELIM_STRUCTURAL_WEIGHT,
    SCORE_CAP_EXP_FIT_FLOOR,
    SCORE_CAP_MAX,
    SCORE_CAP_ML_RECENCY_YEARS,
    SCORE_PENALTY_MULTIPLIER,
)
from scoring import (
    compute_availability_score,
    compute_company_fit,
    compute_experience_fit,
    compute_final_score_with_cap,
    compute_github_bonus,
    compute_industry_bonus,
    compute_location_fit,
    compute_salary_fit,
    compute_structural_score,
    generate_reasoning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_precomputed(precomputed_dir: str):
    df = pd.read_parquet(f"{precomputed_dir}/features.parquet")
    REQUIRED_NEW_COLUMNS = [
        "narrative_text", "narrative_embedding_score",
        "has_disqualifying_language", "n_ghost_skills",
        "is_ghost_skill_candidate", "is_cv_speech_primary",
        "is_cv_speech_no_nlp", "top_jd_skills", "is_tier_1_city",
    ]
    missing = [col for col in REQUIRED_NEW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"features.parquet is missing columns: {missing}. "
            "Re-run scripts/01_extract_features.py to regenerate."
        )
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
            row.get("is_tier_1_city", False),
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
        float(row.get("narrative_embedding_score", 0.0) or 0.0),   # narrative_embedding_score
        bool(row.get("has_disqualifying_language", False)),         # has_disqualifying_language
        bool(row.get("is_ghost_skill_candidate", False)),           # is_ghost_skill_candidate
        bool(row.get("is_cv_speech_no_nlp", False)),                # is_cv_speech_no_nlp
        int(row.get("notice_period_days", 0) or 0),                 # notice_period_days
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
# Debug dump helper
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️  SUBMISSION WARNING: dump_stage() writes to bin/. Only called when
# --debug is passed. Never include debug output in a competition submission.
def dump_stage(
    run_dir: str,
    stage_name: str,
    candidate_ids: list[str],
    scores: "np.ndarray | dict",
    df_features: pd.DataFrame | None = None,
    top_n: int = 100,
) -> None:
    """
    Write the top-N candidates at a pipeline stage to bin/<run_id>/<stage>.csv.

    Columns always present: rank, candidate_id, score
    Extra columns from df_features if provided:
      current_title, current_company, years_of_experience, location, country,
      years_since_last_ml_role, notice_period_days, is_india_based
    """
    # ⚠️  SUBMISSION WARNING: this function writes files — remove --debug flag before submitting
    os.makedirs(run_dir, exist_ok=True)

    if isinstance(scores, dict):
        score_series = pd.Series(scores).rename("score")
    else:
        score_series = pd.Series(scores, index=candidate_ids).rename("score")

    df_dump = score_series.reset_index().rename(columns={"index": "candidate_id"})
    df_dump = df_dump.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    df_dump.insert(0, "rank", range(1, len(df_dump) + 1))

    if df_features is not None:
        extra_cols = [
            "candidate_id", "current_title", "current_company",
            "years_of_experience", "location", "country",
            "years_since_last_ml_role", "notice_period_days", "is_india_based",
        ]
        available = [c for c in extra_cols if c in df_features.columns]
        df_extra = df_features[available].copy()
        df_dump = df_dump.merge(df_extra, on="candidate_id", how="left")

    out_path = os.path.join(run_dir, f"{stage_name}.csv")
    df_dump.to_csv(out_path, index=False)
    print(f"  [DEBUG] Stage dump → {out_path}  ({len(df_dump)} rows)")  # ⚠️ SUBMISSION WARNING


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
    fusion_topn: int = 5000,
    debug_dir: str | None = None,
) -> pd.DataFrame:
    """
    ce_topn:     how many candidates (by prelim score) enter the cross-encoder.
    fusion_topn: how many candidates (by fusion score) proceed to structural scoring.
                 Default raised from 2000 → 5000 to improve recall of the genuine-fit
                 cluster. Stage 3 is pure pandas arithmetic (~2–8s per 1000 rows on CPU);
                 5000 rows adds approximately 10–40s versus the 2000-row default.
                 At 5000 rows, total pipeline time remains within the 5-min cap.
    debug_dir: if set, write per-stage CSVs to this directory.
               ⚠️  SUBMISSION WARNING: only pass debug_dir during development.
    """

    t_pipeline_start = time.time()

    with open("jd_query.txt", encoding="utf-8") as f:
        jd_query_text = f.read().strip()

    # Sort candidate_ids and reorder aligned arrays so BM25 normalisation and
    # CE batch order are identical across all runs (non-determinism fix).
    sort_order      = sorted(range(len(candidate_ids)), key=lambda i: candidate_ids[i])
    candidate_ids   = [candidate_ids[i]  for i in sort_order]
    embeddings      = embeddings[sort_order]
    candidate_texts = [candidate_texts[i] for i in sort_order]

    # ── Stage 1: BM25 ────────────────────────────────────────────────────────
    print("Stage 1: BM25...")
    t0 = time.time()
    bm25_scores = run_bm25(candidate_texts, candidate_ids, jd_query_text)
    t_bm25 = time.time() - t0
    print(f"  BM25 done in {t_bm25:.1f}s.")

    if debug_dir:  # ⚠️  SUBMISSION WARNING: remove --debug before submitting
        dump_stage(debug_dir, "stage1_bm25", candidate_ids, bm25_scores, df)

    # ── Stage 2: Dense ───────────────────────────────────────────────────────
    print("Stage 2: Dense similarity...")
    t1 = time.time()
    dense_scores = compute_dense_scores(embeddings, jd_embedding)
    t_dense = time.time() - t1
    print(f"  Dense done in {t_dense:.1f}s.")

    if debug_dir:  # ⚠️  SUBMISSION WARNING: remove --debug before submitting
        dump_stage(debug_dir, "stage2_dense", candidate_ids, dense_scores, df)

    # ── Fusion → shortlist ────────────────────────────────────────────────────
    print(f"Fusion: weighted score fusion → top-{fusion_topn}...")
    t2 = time.time()
    fusion_scores  = weighted_score_fusion(bm25_scores, dense_scores, candidate_ids)
    shortlist_ids  = sorted(fusion_scores, key=lambda c: -fusion_scores[c])[:fusion_topn]
    df_top = df[df["candidate_id"].isin(set(shortlist_ids))].copy()
    t_fusion = time.time() - t2
    print(f"  Shortlisted {len(df_top)} in {t_fusion:.1f}s.")

    if debug_dir:  # ⚠️  SUBMISSION WARNING: remove --debug before submitting
        dump_stage(debug_dir, "stage2b_fusion", candidate_ids, fusion_scores, df)

    # ── Stage 3: Structural + availability ───────────────────────────────────
    print("Stage 3: Structural + availability scoring...")
    t3 = time.time()
    tqdm.pandas(desc="Structural")
    df_top["structural_score"]   = df_top.progress_apply(_row_structural,   axis=1)
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

    # Apply the same penalty+cap at prelim stage so structurally penalised
    # candidates don't consume prime CE slots in the top-N.
    # The stage dumps confirmed CAND_0091534 (16.6yr) was prelim rank 4 (0.8049)
    # — entering CE at high priority even though the final cap drops them anyway.
    # Applying it here pushes them down in prelim ordering without zeroing them out.
    df_top["experience_fit_prelim"] = df_top["years_of_experience"].apply(compute_experience_fit)
    needs_prelim_penalty = (
        (df_top["experience_fit_prelim"] < SCORE_CAP_EXP_FIT_FLOOR) |
        (df_top["years_since_last_ml_role"] > SCORE_CAP_ML_RECENCY_YEARS)
    )
    df_top.loc[needs_prelim_penalty, "prelim_score"] = (
        df_top.loc[needs_prelim_penalty, "prelim_score"] * SCORE_PENALTY_MULTIPLIER
    ).clip(upper=SCORE_CAP_MAX)

    for col in ("is_honeypot", "entire_career_it_services", "entire_career_research_only"):
        df_top.loc[df_top[col], "prelim_score"] = 0.0

    # R5: CV/speech primary + no NLP narrative → zero prelim score
    cv_no_nlp_zero = (
        df_top["is_cv_speech_no_nlp"] &
        (df_top["narrative_embedding_score"] == 0.0)
    )
    df_top.loc[cv_no_nlp_zero, "prelim_score"] = 0.0

    # R2: disqualifying language → zero prelim score, mirrors the final-stage
    # gate below. Previously this gate only existed AFTER the cross-encoder,
    # which meant disqualified candidates could still consume a CE slot that
    # should have gone to a genuinely strong candidate (ce_topn is a hard
    # budget — wasting slots on candidates that get zeroed anyway hurts
    # recall for borderline-but-clean candidates just outside ce_topn).
    # See the matching comment near `disq_zero` in the final-score block for
    # why `narrative_embedding_score < 0.667` rather than `== 0.0` is correct.
    disq_zero_prelim = (
        df_top["has_disqualifying_language"] &
        (df_top["narrative_embedding_score"] < 0.667)
    )
    df_top.loc[disq_zero_prelim, "prelim_score"] = 0.0

    t_structural = time.time() - t3
    print(f"  Structural done in {t_structural:.1f}s.")

    if debug_dir:  # ⚠️  SUBMISSION WARNING: remove --debug before submitting
        prelim_dict = dict(zip(df_top["candidate_id"], df_top["prelim_score"]))
        dump_stage(debug_dir, "stage3_prelim", candidate_ids, prelim_dict, df_top)

    # ── Stage 4: Cross-encoder ────────────────────────────────────────────────
    print(f"Stage 4: Cross-encoder (top-{ce_topn})...")
    t4 = time.time()
    # R7: exclude hard-zeroed candidates before taking top-N for CE
    df_eligible = df_top[df_top["prelim_score"] > 0.0]
    topN        = df_eligible.nlargest(ce_topn, "prelim_score").copy()
    topN_ids    = topN["candidate_id"].tolist()
    cid_to_text = dict(zip(candidate_ids, candidate_texts))
    topN_texts  = [cid_to_text[cid] for cid in topN_ids]

    ce_scores_norm   = rerank(jd_query_text, topN_ids, topN_texts)
    topN["ce_score"] = ce_scores_norm

    topN["experience_fit_for_cap"] = topN["years_of_experience"].apply(compute_experience_fit)
    topN["final_score"] = topN.apply(
        lambda r: compute_final_score_with_cap(
            r["prelim_score"],
            r["ce_score"],
            r["experience_fit_for_cap"],
            r["years_since_last_ml_role"],
        ),
        axis=1,
    )
    for col in ("is_honeypot", "entire_career_it_services", "entire_career_research_only"):
        topN.loc[topN[col], "final_score"] = 0.0

    # R2: disqualifying language → hard zero, UNLESS the candidate shows strong
    # independent JD-relevant evidence elsewhere (narrative_embedding_score >= 0.667,
    # i.e. 2 of 3 JD signal categories evidenced in their OWN narrative).
    #
    # Previous gate also required `not has_ml_production_experience`, which meant
    # the zero only fired for candidates who had no ML experience at all — exactly
    # the candidates who already scored low on every other axis. The actual leak
    # was candidates WITH general ML production experience whose narrative still
    # explicitly admits deployment ownership belonged to someone else for the
    # ranking/retrieval system the JD cares about. Generic "has done ML" should
    # not shield an explicit non-ownership admission; only strong, specific
    # narrative evidence (cat_a/cat_b/cat_c signals — embeddings/vectorDB,
    # eval-framework, recent hands-on ML) should.
    disq_zero = (
        topN["has_disqualifying_language"] &
        (topN["narrative_embedding_score"] < 0.667)
    )
    topN.loc[disq_zero, "final_score"] = 0.0

    # R5: CV/speech primary + no NLP narrative → hard zero final score
    cv_no_nlp_final = (
        topN["is_cv_speech_no_nlp"] &
        (topN["narrative_embedding_score"] == 0.0)
    )
    topN.loc[cv_no_nlp_final, "final_score"] = 0.0

    t_ce = time.time() - t4
    print(f"  Cross-encoder done in {t_ce:.1f}s.")

    if debug_dir:  # ⚠️  SUBMISSION WARNING: remove --debug before submitting
        final_dict = dict(zip(topN["candidate_id"], topN["final_score"]))
        dump_stage(debug_dir, "stage4_final", candidate_ids, final_dict, topN)

    top100 = topN.nlargest(100, "final_score").copy()
    top100 = top100.sort_values(["final_score", "candidate_id"], ascending=[False, True])
    top100["rank"]  = range(1, 101)
    top100["score"] = top100["final_score"].round(6)

    t_total = time.time() - t_pipeline_start
    print(f"\n{'─'*55}")
    print(f"  Stage 1  BM25               {t_bm25:>7.1f}s")
    print(f"  Stage 2  Dense              {t_dense:>7.1f}s")
    print(f"  Fusion   weighted sum (top-{fusion_topn})  {t_fusion:>7.1f}s")
    print(f"  Stage 3  Structural+avail   {t_structural:>7.1f}s")
    print(f"  Stage 4  Cross-encoder      {t_ce:>7.1f}s")
    print(f"{'─'*55}")
    print(f"  TOTAL                       {t_total:>7.1f}s")
    print(f"{'─'*55}\n")

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
    parser.add_argument("--ce-topn",      type=int, default=500,
                        help="Candidates entering the cross-encoder (by prelim score).")
    parser.add_argument("--fusion-topn",  type=int, default=5000,
                        help="Shortlist size after fusion (before structural scoring). "
                             "Widening improves recall; Stage 3 costs ~2-8s per 1000 rows.")
    # ⚠️  SUBMISSION WARNING: --debug writes stage CSVs to bin/. Never pass
    # this flag when producing a competition submission.
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-stage top-100 CSVs to bin/<run_id>/. Development only — NOT for submission.",
    )
    args = parser.parse_args()

    # Build a run-id from the output filename stem so multiple runs don't overwrite each other
    run_id    = os.path.splitext(os.path.basename(args.out))[0]
    debug_dir = os.path.join("bin", run_id) if args.debug else None

    if args.debug:  # ⚠️  SUBMISSION WARNING: block below only runs with --debug
        print(f"[DEBUG MODE] Stage dumps will be written to: {debug_dir}")
        print("[DEBUG MODE] Do NOT use --debug for a competition submission.\n")

    t_total_start = time.time()

    print("Loading pre-computed data...")
    df, candidate_ids, embeddings, jd_embedding, candidate_texts = load_precomputed(args.precomputed)
    print(f"  {len(candidate_ids)} candidates. Embeddings: {embeddings.shape}")

    top100 = rank_candidates(
        df, candidate_ids, embeddings, jd_embedding,
        candidate_texts, ce_topn=args.ce_topn,
        fusion_topn=args.fusion_topn,
        debug_dir=debug_dir,
    )
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

    t_end = time.time() - t_total_start
    print(f"Total wall-clock time (including data load + reasoning): {t_end:.1f}s")


if __name__ == "__main__":
    main()