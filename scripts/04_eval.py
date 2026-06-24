"""
Script 04: Offline evaluation — ablation table on sample candidates.

Input:  dataset/sample_candidates.json   (50 sample candidates)
        eval/manual_labels.json          (hand-labeled relevance scores 0–3)
Output: eval/eval_results.json           (NDCG@10 per configuration)

Configurations:
  A. BM25 only
  B. Dense (all-MiniLM-L6-v2) only
  C. BM25 + Dense (weighted score fusion)
  D. C + structural features
  E. D + availability  (= rank.py prelim_score)
  F. E + cross-encoder (= rank.py final_score)

Feature extraction uses pipeline/feature_extraction.py — same logic as
01_extract_features.py. Any divergence would silently corrupt the ablation.
"""

import json
import sys
from pathlib import Path

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.feature_extraction import extract_features
from pipeline.bm25_retrieval     import run_bm25
from pipeline.dense_retrieval    import compute_dense_scores
from pipeline.fusion             import weighted_score_fusion
from pipeline.cross_encoder      import rerank, MODEL_NAME as CE_MODEL_NAME
# FIX: weights now imported from config.weights (single source of truth).
# scoring.py re-exports them for backward compat, but importing directly
# from config.weights avoids any risk of circular imports in future.
from config.weights import (
    FINAL_CE_WEIGHT, FINAL_PRELIM_WEIGHT,
    PRELIM_AVAILABILITY_WEIGHT, PRELIM_FUSION_WEIGHT, PRELIM_STRUCTURAL_WEIGHT,
)
from scoring import (
    compute_availability_score, compute_company_fit, compute_experience_fit,
    compute_final_score_with_cap,
    compute_github_bonus, compute_industry_bonus, compute_location_fit,
    compute_salary_fit, compute_structural_score,
)
from utils import build_candidate_text

SAMPLE_PATH   = ROOT / "dataset" / "sample_candidates.json"
LABELS_PATH   = ROOT / "eval"    / "manual_labels.json"
OUT_PATH      = ROOT / "eval"    / "eval_results.json"
JD_QUERY_PATH = ROOT / "jd_query.txt"
DENSE_MODEL   = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# NDCG@10
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list, k: int = 10) -> float:
    r = np.array(relevances[:k], dtype=float)
    if not len(r):
        return 0.0
    return float(np.sum((2.0 ** r - 1.0) / np.log2(np.arange(2, len(r) + 2))))


def ndcg_at_k(ranked_relevances: list, k: int = 10) -> float:
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    return 0.0 if ideal == 0.0 else dcg_at_k(ranked_relevances, k) / ideal


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_structural(feat: dict) -> float:
    return compute_structural_score(
        compute_experience_fit(feat["years_of_experience"]),
        compute_location_fit(feat["is_india_based"], feat["is_target_city"],
                             feat["willing_to_relocate"], feat["is_primary_city"]),
        compute_company_fit(feat["entire_career_it_services"], feat["has_product_company_exp"],
                            feat["has_ml_production_experience"], feat["years_since_last_ml_role"],
                            feat["entire_career_research_only"], feat["shallow_recent_ml_only"]),
        feat["trajectory_score"],
        compute_salary_fit(feat["salary_min_lpa"], feat["salary_max_lpa"]),
        feat["skill_assessment_bonus"], feat["edu_bonus"],
        compute_industry_bonus(feat["current_industry"]),
        compute_github_bonus(feat["github_activity_score"]),
        int(feat.get("n_it_services_roles", 0) or 0),
        float(feat.get("job_hop_score", 0.0) or 0.0),
    )


def score_availability(feat: dict) -> float:
    return compute_availability_score(
        feat["open_to_work_flag"],       feat["last_active_days_ago"],
        feat["recruiter_response_rate"], feat["avg_response_time_hours"],
        feat["notice_period_days"],      feat["saved_by_recruiters_30d"],
        feat["verified_email"],          feat["verified_phone"],
        feat["interview_completion_rate"], feat["offer_acceptance_rate"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not SAMPLE_PATH.exists():
        print(f"ERROR: {SAMPLE_PATH} not found."); sys.exit(1)
    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found.")
        print('Create eval/manual_labels.json: { "CAND_XXXXXXX": 3, ... }')
        sys.exit(1)

    OUT_PATH.parent.mkdir(exist_ok=True)

    with open(SAMPLE_PATH) as f: candidates = json.load(f)
    with open(LABELS_PATH)  as f: labels    = json.load(f)
    with open(JD_QUERY_PATH) as f: jd_query = f.read().strip()

    print(f"Loaded {len(candidates)} candidates, {len(labels)} labels.")

    texts    = [build_candidate_text(c) for c in candidates]
    features = [extract_features(c, detect_honeypot=False) for c in candidates]
    cids     = [c["candidate_id"] for c in candidates]
    feat_map = {c["candidate_id"]: feat for c, feat in zip(candidates, features)}

    # Config A: BM25
    print("\nConfig A: BM25...")
    bm25_scores = run_bm25(texts, cids, jd_query)
    bm25_ranked = sorted(cids, key=lambda c: -float(bm25_scores[cids.index(c)]))
    ndcg_a = ndcg_at_k([labels.get(c, 0) for c in bm25_ranked])

    # Config B: Dense
    print("Config B: Dense...")
    encoder    = SentenceTransformer(DENSE_MODEL)
    cand_embs  = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    jd_emb     = encoder.encode(jd_query, normalize_embeddings=True, convert_to_numpy=True)
    dense_scores = compute_dense_scores(cand_embs, jd_emb)
    dense_ranked = sorted(cids, key=lambda c: -float(dense_scores[cids.index(c)]))
    ndcg_b = ndcg_at_k([labels.get(c, 0) for c in dense_ranked])

    # Config C: Fusion
    print("Config C: Weighted fusion...")
    fusion        = weighted_score_fusion(bm25_scores, dense_scores, cids)
    fusion_ranked = sorted(cids, key=lambda c: -fusion.get(c, 0.0))
    ndcg_c        = ndcg_at_k([labels.get(c, 0) for c in fusion_ranked])

    # Config D: + structural (proportional weights, no availability yet)
    print("Config D: + structural...")
    max_f = max(fusion.values()) or 1.0
    fn    = {c: v / max_f for c, v in fusion.items()}
    _tot  = PRELIM_FUSION_WEIGHT + PRELIM_STRUCTURAL_WEIGHT
    fw, sw = PRELIM_FUSION_WEIGHT / _tot, PRELIM_STRUCTURAL_WEIGHT / _tot
    config_d = sorted(cids, key=lambda c: -(fw * fn.get(c, 0) + sw * score_structural(feat_map[c])))
    ndcg_d   = ndcg_at_k([labels.get(c, 0) for c in config_d])

    # Config E: + availability (= prelim_score)
    print("Config E: + availability...")
    def prelim(c):
        return (PRELIM_FUSION_WEIGHT       * fn.get(c, 0)
                + PRELIM_STRUCTURAL_WEIGHT   * score_structural(feat_map[c])
                + PRELIM_AVAILABILITY_WEIGHT * score_availability(feat_map[c]))
    config_e = sorted(cids, key=lambda c: -prelim(c))
    ndcg_e   = ndcg_at_k([labels.get(c, 0) for c in config_e])

    # Config F: + cross-encoder (= final_score)
    # FIX: use compute_final_score_with_cap() so the eval faithfully reflects
    # the same capping logic used in rank.py (score cap for >15 yrs or stale ML).
    print("Config F: + cross-encoder...")
    prelim_scores = {c: prelim(c) for c in cids}
    top_ids  = sorted(cids, key=lambda c: -prelim_scores[c])[:min(20, len(cids))]
    cid_text = dict(zip(cids, texts))
    ce_norm  = rerank(jd_query, top_ids, [cid_text[c] for c in top_ids])

    final: dict = {}
    for i, c in enumerate(top_ids):
        feat = feat_map[c]
        exp_fit = compute_experience_fit(feat["years_of_experience"])
        final[c] = compute_final_score_with_cap(
            prelim_scores[c],
            float(ce_norm[i]),
            exp_fit,
            feat["years_since_last_ml_role"],
        )
    for c in cids:
        if c not in final:
            final[c] = prelim_scores[c] * FINAL_PRELIM_WEIGHT
    config_f = sorted(cids, key=lambda c: -final[c])
    ndcg_f   = ndcg_at_k([labels.get(c, 0) for c in config_f])

    # Results
    results = {
        "A": {"description": "BM25 only",                      "ndcg_at_10": round(ndcg_a, 4)},
        "B": {"description": "Dense (all-MiniLM-L6-v2) only",  "ndcg_at_10": round(ndcg_b, 4)},
        "C": {"description": "BM25 + Dense (weighted fusion)",  "ndcg_at_10": round(ndcg_c, 4)},
        "D": {"description": "C + structural features",         "ndcg_at_10": round(ndcg_d, 4)},
        "E": {"description": "D + availability",                "ndcg_at_10": round(ndcg_e, 4)},
        "F": {"description": "E + cross-encoder reranker",      "ndcg_at_10": round(ndcg_f, 4)},
    }
    best = max(results, key=lambda k: results[k]["ndcg_at_10"])

    print("\n" + "=" * 65)
    print("ABLATION TABLE — NDCG@10")
    print("=" * 65)
    for cfg, info in results.items():
        delta  = f"  ({info['ndcg_at_10'] - results['A']['ndcg_at_10']:+.4f} vs BM25)" if cfg != "A" else ""
        marker = " ← best" if cfg == best else ""
        print(f"  Config {cfg}  {info['description']:40s}  {info['ndcg_at_10']:.4f}{delta}{marker}")
    print("=" * 65)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
