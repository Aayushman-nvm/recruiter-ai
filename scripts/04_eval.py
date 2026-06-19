"""
Script 05: Offline evaluation — ablation table on sample candidates.

Input:  dataset/sample_candidates.json   (50 sample candidates from bundle)
        eval/manual_labels.json          (hand-labeled relevance scores)
Output: eval/eval_results.json           (NDCG@10 per configuration)
        prints ablation table to stdout

NOTE: manual_labels.json is gitignored — it contains your subjective judgments.
      eval_results.json IS committed — it's the output that goes in the README.

STEP 1 — Manual labeling (do this ONCE):
  For each of the 50 sample candidates, assign:
    3 = Strong fit (would shortlist for interview)
    2 = Moderate fit (worth reviewing)
    1 = Weak fit (probably not)
    0 = Not a fit (clear disqualifier)
  Save as: eval/manual_labels.json  { "CAND_XXXXXXX": 3, ... }

STEP 2 — Run this script from the project root:
  python scripts/05_eval.py

Configurations tested:
  A. BM25 only                           (keyword baseline)
  B. Dense (bge-base) only               (semantic baseline)
  C. BM25 + Dense (RRF)                  (hybrid retrieval)
  D. C + structural features             (+ company/location/ML signals)
  E. D + availability (additive)         (+ reachability signals)
  F. E + cross-encoder reranker          (full pipeline — expected best)

The NDCG@10 numbers from Config A vs F show how much lift the full pipeline
gives over naive keyword matching. This is what goes in the README.
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Path setup — import scoring.py and utils.py from project root ─────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scoring import (
    IT_SERVICES_COMPANIES,
    REFERENCE_DATE,
    TARGET_CITIES,
    compute_availability_score,
    compute_company_fit,
    compute_education_bonus,
    compute_experience_fit,
    compute_location_fit,
    compute_salary_fit,
    compute_skill_assessment_bonus,
    compute_structural_score,
)
from utils import build_candidate_text

# ── Paths (relative to project root — run from project root) ──────────────────
SAMPLE_PATH   = ROOT / "dataset" / "sample_candidates.json"
LABELS_PATH   = ROOT / "eval" / "manual_labels.json"
OUT_PATH      = ROOT / "eval" / "eval_results.json"
JD_QUERY_PATH = ROOT / "jd_query.txt"

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Same ML_KEYWORDS as 01_extract_features.py — must stay in sync
ML_KEYWORDS = [
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "llm", "fine-tun", "rag", "semantic search", "sentence-transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "bert", "transformer", "nlp", "information retrieval", "learning to rank",
    "xgboost ranking", "neural ranker", "reranker", "dense retrieval",
    "search engine", "knowledge graph", "question answering", "vector database",
    "approximate nearest neighbor", "ann", "hnsw", "cosine similarity",
]


# ─────────────────────────────────────────────────────────────────────────────
# NDCG@10 implementation
# ─────────────────────────────────────────────────────────────────────────────


def dcg_at_k(relevances: list, k: int = 10) -> float:
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains     = 2.0 ** relevances - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_relevances: list, k: int = 10) -> float:
    ideal = sorted(ranked_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal_dcg


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — mirrors 01_extract_features.py exactly
# Changes here must be mirrored there and vice versa.
# ─────────────────────────────────────────────────────────────────────────────


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def extract_features_inline(c: dict) -> dict:
    """
    Feature extraction for a single candidate dict.

    MUST mirror scripts/01_extract_features.py — if you change the logic there,
    change it here too. The eval is only meaningful if it uses the same features
    as the actual pipeline.

    Key difference from parquet version: returns a plain dict, not a DataFrame row.
    Also skips honeypot detection (not needed on the small 50-candidate eval set).
    """
    p      = c["profile"]
    sig    = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    edu    = c.get("education", [])
    skills = c.get("skills", [])

    yoe      = float(p.get("years_of_experience", 0) or 0)
    country  = (p.get("country") or "").strip()
    location = (p.get("location") or "").strip()

    # ── Location ──────────────────────────────────────────────────────────────
    is_india_based     = country.lower() == "india"
    is_target_city     = any(city in location.lower() for city in TARGET_CITIES)
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))

    # ── Company type ──────────────────────────────────────────────────────────
    n_total = len(career)
    n_it    = sum(
        1 for r in career
        if any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
    )
    entire_it   = n_total > 0 and n_it == n_total
    has_product = any(
        not any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
        for r in career
    )

    # ── ML production experience ──────────────────────────────────────────────
    # Scan ALL career history (no 6-year cutoff) — mirrors 01_extract_features.py fix.
    # years_since_last_ml_role captures recency; a hard cutoff caused false negatives.
    has_ml        = False
    yrs_since_ml  = 99.0   # sentinel: "never did ML"

    for role in career:
        desc = (role.get("description") or "").lower()
        if any(kw in desc for kw in ML_KEYWORDS):
            has_ml = True
            end_raw = role.get("end_date")
            end     = _parse_date(end_raw) if end_raw else REFERENCE_DATE
            yrs_ago = max(0.0, (REFERENCE_DATE - (end or REFERENCE_DATE)).days / 365.25)
            yrs_since_ml = min(yrs_since_ml, yrs_ago)

    # ── Salary ────────────────────────────────────────────────────────────────
    sal     = (sig.get("expected_salary_range_inr_lpa") or {})
    sal_min = float(sal.get("min", 0) or 0)
    sal_max = float(sal.get("max", 999) or 999)

    # ── Skill assessment bonus ────────────────────────────────────────────────
    skill_bonus = compute_skill_assessment_bonus(sig.get("skill_assessment_scores") or {})

    # ── Education ─────────────────────────────────────────────────────────────
    edu_tier  = edu[0].get("tier",           "unknown") if edu else "unknown"
    edu_field = edu[0].get("field_of_study", "")        if edu else ""
    edu_b     = compute_education_bonus(edu_tier, edu_field)

    # ── Trajectory ────────────────────────────────────────────────────────────
    durations  = [r.get("duration_months", 0) or 0 for r in career]
    avg_tenure = float(sum(durations) / len(durations)) if durations else 0.0

    SENIORITY_HIGH   = {"principal","staff engineer","head of","vp","director","distinguished","fellow","chief"}
    SENIORITY_SENIOR = {"senior","lead","tech lead","sr.","sr ","staff"}
    SENIORITY_JUNIOR = {"junior","associate","entry","fresher","intern","trainee"}

    def _seniority(title: str) -> int:
        t = (title or "").lower()
        if any(k in t for k in SENIORITY_HIGH):   return 5
        if any(k in t for k in SENIORITY_SENIOR): return 4
        if any(k in t for k in SENIORITY_JUNIOR): return 1
        return 3

    tit_now   = _seniority(p.get("current_title", ""))
    tit_first = _seniority(career[-1].get("title", "") if career else "")
    traj_up   = tit_now > tit_first

    tenure_stab  = 1.0 if avg_tenure > 24 else (0.7 if avg_tenure > 12 else 0.3)
    # ml_rec_score uses sentinel-aware logic: 99.0 means no ML → 0.0
    ml_rec_score = (
        1.0 if yrs_since_ml <= 0 else
        (0.5 if yrs_since_ml <= 2 else
         (0.2 if yrs_since_ml <= 4 else 0.0))
    )
    traj_score = 0.4 * tenure_stab + 0.3 * float(traj_up) + 0.3 * ml_rec_score

    # ── Availability ──────────────────────────────────────────────────────────
    try:
        last_active = date.fromisoformat(str(sig.get("last_active_date", ""))[:10])
        last_days   = (REFERENCE_DATE - last_active).days
    except Exception:
        last_days = 999

    return {
        "candidate_id":                c["candidate_id"],
        "years_of_experience":          yoe,
        "is_india_based":               is_india_based,
        "is_target_city":               is_target_city,
        "willing_to_relocate":          willing_to_relocate,
        "entire_career_it_services":    entire_it,
        "has_product_company_exp":      has_product,
        "has_ml_production_experience": has_ml,
        "years_since_last_ml_role":     yrs_since_ml,
        "salary_min_lpa":               sal_min,
        "salary_max_lpa":               sal_max,
        "skill_assessment_bonus":       skill_bonus,
        "edu_bonus":                    edu_b,
        "trajectory_score":             traj_score,
        "open_to_work_flag":            bool(sig.get("open_to_work_flag", False)),
        "last_active_days_ago":         last_days,
        "recruiter_response_rate":      float(sig.get("recruiter_response_rate", 0) or 0),
        "avg_response_time_hours":      float(sig.get("avg_response_time_hours")
                                              if sig.get("avg_response_time_hours") is not None else -1),
        "notice_period_days":           int(sig.get("notice_period_days", 90) or 90),
        "saved_by_recruiters_30d":      int(sig.get("saved_by_recruiters_30d", 0) or 0),
        "verified_email":               bool(sig.get("verified_email", False)),
        "verified_phone":               bool(sig.get("verified_phone", False)),
        "is_honeypot":                  False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring wrappers
# ─────────────────────────────────────────────────────────────────────────────


def score_structural(feat: dict) -> float:
    return compute_structural_score(
        compute_experience_fit(feat["years_of_experience"]),
        compute_location_fit(feat["is_india_based"], feat["is_target_city"], feat["willing_to_relocate"]),
        compute_company_fit(
            feat["entire_career_it_services"], feat["has_product_company_exp"],
            feat["has_ml_production_experience"], feat["years_since_last_ml_role"]
        ),
        feat["trajectory_score"],
        compute_salary_fit(feat["salary_min_lpa"], feat["salary_max_lpa"]),
        feat["skill_assessment_bonus"],
        feat["edu_bonus"],
    )


def score_availability(feat: dict) -> float:
    return compute_availability_score(
        feat["open_to_work_flag"],   feat["last_active_days_ago"],
        feat["recruiter_response_rate"], feat["avg_response_time_hours"],
        feat["notice_period_days"],  feat["saved_by_recruiters_30d"],
        feat["verified_email"],      feat["verified_phone"],
    )


def rrf_fusion(rank_a: list, rank_b: list, k: int = 60) -> dict:
    scores: dict = {}
    for rank, cid in enumerate(rank_a):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(rank_b):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Guard: input files must exist ─────────────────────────────────────────
    if not SAMPLE_PATH.exists():
        print(f"ERROR: {SAMPLE_PATH} not found.")
        sys.exit(1)
    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found.")
        print("Create eval/manual_labels.json with relevance scores 0–3 per candidate ID.")
        print('Format: { "CAND_0000031": 3, "CAND_0000038": 2, ... }')
        sys.exit(1)

    OUT_PATH.parent.mkdir(exist_ok=True)

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    with open(JD_QUERY_PATH, encoding="utf-8") as f:
        jd_query_text = f.read().strip()

    print(f"Loaded {len(candidates)} candidates, {len(labels)} labels.")
    n_relevant = sum(1 for v in labels.values() if v > 0)
    print(f"  Relevant (label > 0): {n_relevant} | Strong fit (label=3): "
          f"{sum(1 for v in labels.values() if v == 3)}")

    # ── Prepare ───────────────────────────────────────────────────────────────
    texts    = [build_candidate_text(c) for c in candidates]
    features = [extract_features_inline(c) for c in candidates]
    cids     = [c["candidate_id"] for c in candidates]
    feat_map = {c["candidate_id"]: feat for c, feat in zip(candidates, features)}

    # ── Config A: BM25 only ───────────────────────────────────────────────────
    print("\nConfig A: BM25 only (bm25s)...")
    corpus_tokens = bm25s.tokenize(texts, stopwords=None, show_progress=False)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens, show_progress=False)

    query_tokens = bm25s.tokenize(jd_query_text, stopwords=None, show_progress=False)
    bm25_indices, _ = bm25_retriever.retrieve(query_tokens, k=len(cids), show_progress=False)
    bm25_ranked     = [cids[i] for i in bm25_indices[0]]
    ndcg_a = ndcg_at_k([labels.get(cid, 0) for cid in bm25_ranked])

    # ── Config B: Dense (bge-base) only ──────────────────────────────────────
    print("Config B: Dense (bge-base) only...")
    encoder       = SentenceTransformer("BAAI/bge-base-en-v1.5")
    cand_embs     = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    jd_emb        = encoder.encode(BGE_QUERY_INSTRUCTION + jd_query_text,
                                   normalize_embeddings=True, convert_to_numpy=True)
    dense_raw     = cand_embs @ jd_emb
    dense_ranked  = sorted(cids, key=lambda cid: -float(dense_raw[cids.index(cid)]))
    ndcg_b = ndcg_at_k([labels.get(cid, 0) for cid in dense_ranked])

    # ── Config C: BM25 + Dense (RRF) ─────────────────────────────────────────
    print("Config C: BM25 + Dense (RRF)...")
    rrf           = rrf_fusion(bm25_ranked, dense_ranked)
    rrf_ranked    = sorted(cids, key=lambda cid: -rrf.get(cid, 0.0))
    ndcg_c        = ndcg_at_k([labels.get(cid, 0) for cid in rrf_ranked])

    # ── Config D: C + structural ──────────────────────────────────────────────
    print("Config D: RRF + structural features...")
    max_rrf   = max(rrf.values()) if rrf else 1.0
    rrf_norm  = {cid: v / max_rrf for cid, v in rrf.items()}

    def config_d_score(cid: str) -> float:
        return 0.65 * rrf_norm.get(cid, 0.0) + 0.35 * score_structural(feat_map[cid])

    config_d_ranked = sorted(cids, key=lambda cid: -config_d_score(cid))
    ndcg_d = ndcg_at_k([labels.get(cid, 0) for cid in config_d_ranked])

    # ── Config E: D + availability ────────────────────────────────────────────
    print("Config E: RRF + structural + availability...")

    def config_e_score(cid: str) -> float:
        return (0.50 * rrf_norm.get(cid, 0.0)
                + 0.35 * score_structural(feat_map[cid])
                + 0.15 * score_availability(feat_map[cid]))

    config_e_ranked = sorted(cids, key=lambda cid: -config_e_score(cid))
    ndcg_e = ndcg_at_k([labels.get(cid, 0) for cid in config_e_ranked])

    # ── Config F: E + cross-encoder ──────────────────────────────────────────
    print("Config F: Full pipeline + cross-encoder (~30–60s)...")
    prelim_scores = {cid: config_e_score(cid) for cid in cids}
    # On 50 candidates, run CE on the top 20 (those candidates are all we have)
    top_n   = min(20, len(cids))
    top_ids = sorted(cids, key=lambda cid: -prelim_scores[cid])[:top_n]
    cid_to_text = dict(zip(cids, texts))

    ce_model  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs     = [(jd_query_text, cid_to_text[cid]) for cid in top_ids]
    ce_scores = ce_model.predict(pairs, batch_size=16)
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    ce_norm = (ce_scores - ce_min) / (ce_max - ce_min + 1e-9)

    final_scores: dict = {}
    for i, cid in enumerate(top_ids):
        final_scores[cid] = 0.40 * prelim_scores[cid] + 0.60 * float(ce_norm[i])
    # Candidates outside top_n get a deflated prelim score (CE not run on them)
    for cid in cids:
        if cid not in final_scores:
            final_scores[cid] = prelim_scores[cid] * 0.40

    config_f_ranked = sorted(cids, key=lambda cid: -final_scores[cid])
    ndcg_f = ndcg_at_k([labels.get(cid, 0) for cid in config_f_ranked])

    # ── Print ablation table ──────────────────────────────────────────────────
    results = {
        "A": {"description": "BM25 only",                 "ndcg_at_10": round(ndcg_a, 4)},
        "B": {"description": "Dense (bge-base) only",      "ndcg_at_10": round(ndcg_b, 4)},
        "C": {"description": "BM25 + Dense (RRF)",         "ndcg_at_10": round(ndcg_c, 4)},
        "D": {"description": "C + structural features",    "ndcg_at_10": round(ndcg_d, 4)},
        "E": {"description": "D + availability",           "ndcg_at_10": round(ndcg_e, 4)},
        "F": {"description": "E + cross-encoder reranker", "ndcg_at_10": round(ndcg_f, 4)},
    }

    best_cfg = max(results, key=lambda k: results[k]["ndcg_at_10"])

    print("\n" + "=" * 65)
    print("ABLATION TABLE — NDCG@10 on 50 sample candidates")
    print("=" * 65)
    for cfg, info in results.items():
        marker = " ← best" if cfg == best_cfg else ""
        lift   = ""
        if cfg != "A":
            delta = info["ndcg_at_10"] - results["A"]["ndcg_at_10"]
            lift  = f"  ({'+' if delta >= 0 else ''}{delta:.4f} vs BM25 baseline)"
        print(f"  Config {cfg}  {info['description']:35s}  "
              f"NDCG@10 = {info['ndcg_at_10']:.4f}{lift}{marker}")
    print("=" * 65)
    print(f"\nFull pipeline lift over BM25 baseline: "
          f"{results['F']['ndcg_at_10'] - results['A']['ndcg_at_10']:+.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {OUT_PATH}")
    print("\nCopy the table above into README.md under 'Offline Evaluation'.")


if __name__ == "__main__":
    main()
