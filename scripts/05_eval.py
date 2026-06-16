"""
Script 05: Offline evaluation — ablation table on sample candidates.

Input:  dataset/sample_candidates.json   (50 sample candidates from bundle)
        eval/manual_labels.json          (hand-labeled relevance scores — create locally)
Output: eval/eval_results.json           (NDCG@10 per configuration)
        prints ablation table to stdout

STEP 1 — Manual labeling (do this ONCE, ~30 minutes):
  For each of the 50 sample candidates, assign:
    3 = Strong fit (would shortlist for interview)
    2 = Moderate fit (worth reviewing)
    1 = Weak fit (probably not)
    0 = Not a fit (clear disqualifier)
  Save as: eval/manual_labels.json  { "CAND_XXXXXXX": 3, ... }

STEP 2 — Run this script:
  python scripts/05_eval.py

Configurations tested:
  A. BM25 only
  B. Dense (bge-base) only
  C. BM25 + Dense (RRF)
  D. C + structural features
  E. D + availability (additive)
  F. E + cross-encoder reranker   ← expected best
"""

import json
import os
import sys
from datetime import date

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Add parent directory so we can import scoring and utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_PATH = "dataset/sample_candidates.json"
LABELS_PATH = "eval/manual_labels.json"
OUT_PATH    = "eval/eval_results.json"
JD_QUERY_PATH = "jd_query.txt"

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

ML_KEYWORDS = [
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "llm", "fine-tun", "rag", "semantic search", "sentence-transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "bert", "transformer", "nlp", "information retrieval", "learning to rank",
    "reranker", "dense retrieval", "search engine", "vector database",
    "approximate nearest neighbor", "ann", "cosine similarity",
]


# ─────────────────────────────────────────────────────────────────────────────
# NDCG implementation
# ─────────────────────────────────────────────────────────────────────────────


def dcg_at_k(relevances: list, k: int = 10) -> float:
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains = 2.0 ** relevances - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_relevances: list, k: int = 10) -> float:
    ideal = sorted(ranked_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal_dcg


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (mirrors Script 01 — no parquet dependency for eval)
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
    Inline feature extraction for a single candidate dict.
    Mirrors scripts/01_extract_features.py but returns a plain dict
    (not a parquet row) for use in the eval script without needing precomputed/.
    """
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    education = c.get("education", [])
    skills = c.get("skills", [])

    yoe = float(p.get("years_of_experience", 0) or 0)
    country = (p.get("country") or "").strip()
    location = (p.get("location") or "").strip()

    is_india_based = country.lower() == "india"
    is_target_city = any(city in location.lower() for city in TARGET_CITIES)
    willing_to_relocate = bool(sig.get("willing_to_relocate", False))

    n_total = len(career)
    n_it = sum(
        1 for r in career
        if any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
    )
    entire_it = n_total > 0 and n_it == n_total
    has_product = any(
        not any(it in (r.get("company") or "").lower() for it in IT_SERVICES_COMPANIES)
        for r in career
    )

    cutoff = date(REFERENCE_DATE.year - 6, REFERENCE_DATE.month, REFERENCE_DATE.day)
    has_ml = False
    yrs_since_ml = float(yoe)
    for role in career:
        start = _parse_date(role.get("start_date"))
        if start is None or start < cutoff:
            continue
        desc = (role.get("description") or "").lower()
        if any(kw in desc for kw in ML_KEYWORDS):
            has_ml = True
            end_raw = role.get("end_date")
            end = _parse_date(end_raw) if end_raw else REFERENCE_DATE
            yrs_ago = max(0.0, (REFERENCE_DATE - (end or REFERENCE_DATE)).days / 365.25)
            yrs_since_ml = min(yrs_since_ml, yrs_ago)

    sal = (sig.get("expected_salary_range_inr_lpa") or {})
    sal_min = float(sal.get("min", 0) or 0)
    sal_max = float(sal.get("max", 999) or 999)

    skill_bonus = compute_skill_assessment_bonus(sig.get("skill_assessment_scores") or {})

    edu_tier  = education[0].get("tier", "unknown") if education else "unknown"
    edu_field = education[0].get("field_of_study", "") if education else ""
    edu_b = compute_education_bonus(edu_tier, edu_field)

    # Trajectory
    durations = [r.get("duration_months", 0) or 0 for r in career]
    avg_tenure = float(sum(durations) / len(durations)) if durations else 0.0

    SENIORITY_HIGH   = {"principal", "staff engineer", "head of", "vp", "director", "distinguished", "fellow"}
    SENIORITY_SENIOR = {"senior", "lead", "tech lead", "sr.", "sr "}
    SENIORITY_JUNIOR = {"junior", "associate", "entry", "fresher", "intern"}

    def seniority(title: str) -> int:
        t = (title or "").lower()
        if any(k in t for k in SENIORITY_HIGH):   return 5
        if any(k in t for k in SENIORITY_SENIOR): return 4
        if any(k in t for k in SENIORITY_JUNIOR): return 1
        return 3

    tit_now   = seniority(p.get("current_title", ""))
    tit_first = seniority(career[-1].get("title", "") if career else "")
    traj_up   = tit_now > tit_first
    tenure_stab = 1.0 if avg_tenure > 24 else (0.7 if avg_tenure > 12 else 0.3)
    ml_rec_score = (1.0 if yrs_since_ml <= 0 else
                    (0.5 if yrs_since_ml <= 2 else (0.2 if yrs_since_ml <= 4 else 0.0)))
    traj_score = 0.4 * tenure_stab + 0.3 * float(traj_up) + 0.3 * ml_rec_score

    try:
        last_active = date.fromisoformat(str(sig.get("last_active_date", ""))[:10])
        last_days = (REFERENCE_DATE - last_active).days
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
        "avg_response_time_hours":      float(sig.get("avg_response_time_hours") if sig.get("avg_response_time_hours") is not None else -1),
        "notice_period_days":           int(sig.get("notice_period_days", 90) or 90),
        "saved_by_recruiters_30d":      int(sig.get("saved_by_recruiters_30d", 0) or 0),
        "verified_email":               bool(sig.get("verified_email", False)),
        "verified_phone":               bool(sig.get("verified_phone", False)),
        "is_honeypot":                  False,  # not computing honeypot in eval — small set
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers for individual configurations
# ─────────────────────────────────────────────────────────────────────────────


def score_structural(feat: dict) -> float:
    exp  = compute_experience_fit(feat["years_of_experience"])
    loc  = compute_location_fit(feat["is_india_based"], feat["is_target_city"], feat["willing_to_relocate"])
    comp = compute_company_fit(
        feat["entire_career_it_services"], feat["has_product_company_exp"],
        feat["has_ml_production_experience"], feat["years_since_last_ml_role"]
    )
    sal  = compute_salary_fit(feat["salary_min_lpa"], feat["salary_max_lpa"])
    return compute_structural_score(
        exp, loc, comp, feat["trajectory_score"], sal,
        feat["skill_assessment_bonus"], feat["edu_bonus"]
    )


def score_availability(feat: dict) -> float:
    return compute_availability_score(
        feat["open_to_work_flag"], feat["last_active_days_ago"],
        feat["recruiter_response_rate"], feat["avg_response_time_hours"],
        feat["notice_period_days"], feat["saved_by_recruiters_30d"],
        feat["verified_email"], feat["verified_phone"]
    )


def rrf_fusion(rank_a: list, rank_b: list, k: int = 60) -> dict:
    scores = {}
    for rank, cid in enumerate(rank_a):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(rank_b):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Check for input files ─────────────────────────────────────────────────
    if not os.path.exists(SAMPLE_PATH):
        print(f"ERROR: {SAMPLE_PATH} not found. Place sample_candidates.json in dataset/.")
        sys.exit(1)

    if not os.path.exists(LABELS_PATH):
        print(f"ERROR: {LABELS_PATH} not found.")
        print("Create eval/manual_labels.json with relevance scores 0–3 per candidate ID.")
        print('Format: { "CAND_000001": 3, "CAND_000002": 1, ... }')
        sys.exit(1)

    os.makedirs("eval", exist_ok=True)

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    with open(JD_QUERY_PATH, encoding="utf-8") as f:
        jd_query_text = f.read().strip()

    print(f"Loaded {len(candidates)} candidates, {len(labels)} labels.")

    # ── Prepare texts and features ────────────────────────────────────────────
    texts = [build_candidate_text(c) for c in candidates]
    features = [extract_features_inline(c) for c in candidates]
    cids = [c["candidate_id"] for c in candidates]

    # ── Config A: BM25 only ───────────────────────────────────────────────────
    print("\nConfig A: BM25 only...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    jd_tokens = jd_query_text.lower().split()
    bm25_raw = bm25.get_scores(jd_tokens)
    bm25_ranked = sorted(range(len(cids)), key=lambda i: -bm25_raw[i])
    config_a_relevances = [labels.get(cids[i], 0) for i in bm25_ranked]
    ndcg_a = ndcg_at_k(config_a_relevances)

    # ── Config B: Dense (bge-base) only ──────────────────────────────────────
    print("Config B: Dense (bge-base) only...")
    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    cand_embeddings = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    jd_embedding = encoder.encode(
        BGE_QUERY_INSTRUCTION + jd_query_text,
        normalize_embeddings=True, convert_to_numpy=True
    )
    dense_raw = cand_embeddings @ jd_embedding
    dense_ranked = sorted(range(len(cids)), key=lambda i: -float(dense_raw[i]))
    config_b_relevances = [labels.get(cids[i], 0) for i in dense_ranked]
    ndcg_b = ndcg_at_k(config_b_relevances)

    # ── Config C: BM25 + Dense (RRF) ─────────────────────────────────────────
    print("Config C: BM25 + Dense RRF...")
    bm25_cid_ranked = [cids[i] for i in bm25_ranked]
    dense_cid_ranked = [cids[i] for i in dense_ranked]
    rrf = rrf_fusion(bm25_cid_ranked, dense_cid_ranked)
    rrf_ranked_cids = sorted(cids, key=lambda cid: -rrf.get(cid, 0.0))
    config_c_relevances = [labels.get(cid, 0) for cid in rrf_ranked_cids]
    ndcg_c = ndcg_at_k(config_c_relevances)

    # ── Config D: C + structural features ────────────────────────────────────
    print("Config D: RRF + structural...")
    max_rrf = max(rrf.values()) if rrf else 1.0
    rrf_norm = {cid: v / max_rrf for cid, v in rrf.items()}
    feat_map = {c["candidate_id"]: feat for c, feat in zip(candidates, features)}

    def config_d_score(cid):
        rrf_s = rrf_norm.get(cid, 0.0)
        struct_s = score_structural(feat_map[cid])
        return 0.65 * rrf_s + 0.35 * struct_s

    config_d_ranked = sorted(cids, key=lambda cid: -config_d_score(cid))
    config_d_relevances = [labels.get(cid, 0) for cid in config_d_ranked]
    ndcg_d = ndcg_at_k(config_d_relevances)

    # ── Config E: D + availability ────────────────────────────────────────────
    print("Config E: RRF + structural + availability...")

    def config_e_score(cid):
        rrf_s    = rrf_norm.get(cid, 0.0)
        struct_s = score_structural(feat_map[cid])
        avail_s  = score_availability(feat_map[cid])
        return 0.50 * rrf_s + 0.35 * struct_s + 0.15 * avail_s

    config_e_ranked = sorted(cids, key=lambda cid: -config_e_score(cid))
    config_e_relevances = [labels.get(cid, 0) for cid in config_e_ranked]
    ndcg_e = ndcg_at_k(config_e_relevances)

    # ── Config F: E + cross-encoder reranker ─────────────────────────────────
    print("Config F: Full pipeline + cross-encoder (this takes ~1-2 min)...")
    prelim_scores = {cid: config_e_score(cid) for cid in cids}
    top_n = min(20, len(cids))   # top-20 on small eval set (≤50 candidates)
    top_ids = sorted(cids, key=lambda cid: -prelim_scores[cid])[:top_n]
    cid_to_text = dict(zip(cids, texts))

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(jd_query_text, cid_to_text[cid]) for cid in top_ids]
    ce_scores = cross_encoder.predict(pairs, batch_size=16)
    ce_min, ce_max = ce_scores.min(), ce_scores.max()
    ce_norm = (ce_scores - ce_min) / (ce_max - ce_min + 1e-9)

    final_scores = {}
    for i, cid in enumerate(top_ids):
        final_scores[cid] = 0.40 * prelim_scores[cid] + 0.60 * float(ce_norm[i])
    # Candidates not in top_n keep their prelim score
    for cid in cids:
        if cid not in final_scores:
            final_scores[cid] = prelim_scores[cid] * 0.40   # CE not run → lower weight

    config_f_ranked = sorted(cids, key=lambda cid: -final_scores[cid])
    config_f_relevances = [labels.get(cid, 0) for cid in config_f_ranked]
    ndcg_f = ndcg_at_k(config_f_relevances)

    # ── Print ablation table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION TABLE — NDCG@10 on sample candidates")
    print("=" * 60)
    results = {
        "A": {"description": "BM25 only",                "ndcg_at_10": round(ndcg_a, 4)},
        "B": {"description": "Dense (bge-base) only",     "ndcg_at_10": round(ndcg_b, 4)},
        "C": {"description": "BM25 + Dense (RRF)",        "ndcg_at_10": round(ndcg_c, 4)},
        "D": {"description": "C + structural features",   "ndcg_at_10": round(ndcg_d, 4)},
        "E": {"description": "D + availability",          "ndcg_at_10": round(ndcg_e, 4)},
        "F": {"description": "E + cross-encoder reranker","ndcg_at_10": round(ndcg_f, 4)},
    }
    for cfg, info in results.items():
        marker = " ← best" if cfg == "F" else ""
        print(f"  Config {cfg} ({info['description']:35s}):  NDCG@10 = {info['ndcg_at_10']:.4f}{marker}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {OUT_PATH}")
    print("Use these numbers in README.md and your presentation deck.")


if __name__ == "__main__":
    main()
