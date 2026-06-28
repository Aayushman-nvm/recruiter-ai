"""
sandbox/app.py — Streamlit demo for the Redrob Candidate Ranker.

Usage: streamlit run sandbox/app.py

Handles ≤100 candidates uploaded as a JSON array.
Uses pipeline/feature_extraction.py for feature extraction — same logic as
rank.py and scripts/01_extract_features.py. No formula drift.
Runs cross-encoder since ≤100 candidates is fast enough for live demo.
"""

import io
import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature_extraction import extract_features
from pipeline.cross_encoder import rerank
from utils import build_candidate_text
from config.weights import (
    PRELIM_FUSION_WEIGHT,
    PRELIM_STRUCTURAL_WEIGHT,
    PRELIM_AVAILABILITY_WEIGHT,
    FINAL_PRELIM_WEIGHT,
    FINAL_CE_WEIGHT,
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

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker — Demo")
st.caption("Upload ≤100 candidates (JSON array) and rank them against the JD query.")

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@st.cache_resource
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


encoder = load_encoder()

# ─────────────────────────────────────────────────────────────────────────────
# UI inputs
# ─────────────────────────────────────────────────────────────────────────────

jd_query_default = ""
jd_query_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jd_query.txt")
if os.path.exists(jd_query_path):
    with open(jd_query_path, encoding="utf-8") as f:
        jd_query_default = f.read()

jd_text = st.text_area("JD Query (compact requirements)", value=jd_query_default, height=150)
candidates_file = st.file_uploader("Candidates JSON (array of candidate objects)", type=["json"])

# ─────────────────────────────────────────────────────────────────────────────
# Structural / availability row scorers (mirrors rank.py exactly)
# ─────────────────────────────────────────────────────────────────────────────


def _row_structural(feat: dict) -> float:
    return compute_structural_score(
        compute_experience_fit(feat["years_of_experience"]),
        compute_location_fit(
            feat["is_india_based"],
            feat["is_target_city"],
            feat["willing_to_relocate"],
            feat["is_primary_city"],
            feat.get("is_tier_1_city", False),
        ),
        compute_company_fit(
            feat["entire_career_it_services"],
            feat["has_product_company_exp"],
            feat["has_ml_production_experience"],
            feat["years_since_last_ml_role"],
            feat["entire_career_research_only"],
            feat["shallow_recent_ml_only"],
        ),
        feat["trajectory_score"],
        compute_salary_fit(feat["salary_min_lpa"], feat["salary_max_lpa"]),
        feat["skill_assessment_bonus"],
        feat["edu_bonus"],
        compute_industry_bonus(feat["current_industry"]),
        compute_github_bonus(feat["github_activity_score"]),
        int(feat.get("n_it_services_roles", 0) or 0),
        float(feat.get("job_hop_score", 0.0) or 0.0),
        float(feat.get("narrative_embedding_score", 0.0) or 0.0),
        bool(feat.get("has_disqualifying_language", False)),
        bool(feat.get("is_ghost_skill_candidate", False)),
        bool(feat.get("is_cv_speech_no_nlp", False)),
        int(feat.get("notice_period_days", 0) or 0),
        float(feat.get("recruiter_response_rate", 0.5) or 0.5),
        bool(feat.get("is_junior_stagnant", False)),
    )


def _row_availability(feat: dict) -> float:
    return compute_availability_score(
        feat["open_to_work_flag"],
        feat["last_active_days_ago"],
        feat["recruiter_response_rate"],
        feat["avg_response_time_hours"],
        feat["notice_period_days"],
        feat["saved_by_recruiters_30d"],
        feat["verified_email"],
        feat["verified_phone"],
        feat["interview_completion_rate"],
        feat["offer_acceptance_rate"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

if st.button("Rank") and candidates_file and jd_text:
    with st.spinner("Embedding, scoring, reranking..."):
        candidates = json.load(candidates_file)

        if len(candidates) > 100:
            st.warning(f"Loaded {len(candidates)} candidates — truncating to first 100 for the demo.")
            candidates = candidates[:100]

        # Build text and extract features using the same logic as rank.py
        texts    = [build_candidate_text(c) for c in candidates]
        features = [extract_features(c, detect_honeypot=True) for c in candidates]

        # Embed candidates (plain text — BGE asymmetric)
        cand_embeddings = encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        # Embed JD query (with instruction prefix — BGE asymmetric)
        jd_embedding = encoder.encode(
            BGE_QUERY_INSTRUCTION + jd_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        dense_scores = cand_embeddings @ jd_embedding

        rows = []
        for i, (c, feat) in enumerate(zip(candidates, features)):
            p = c["profile"]

            structural   = _row_structural(feat)
            availability = _row_availability(feat)
            sem          = float(dense_scores[i])

            prelim = (
                PRELIM_FUSION_WEIGHT       * max(sem, 0.0) +
                PRELIM_STRUCTURAL_WEIGHT   * structural +
                PRELIM_AVAILABILITY_WEIGHT * availability
            )

            rows.append({
                "candidate_id": c["candidate_id"],
                "name":         p.get("anonymized_name", ""),
                "title":        p.get("current_title", ""),
                "prelim_score": prelim,
                "semantic":     round(max(sem, 0.0), 3),
                "structural":   round(structural, 3),
                "availability": round(availability, 3),
                "text":         texts[i],
                "_feat":        feat,   # kept for reasoning; dropped before display
            })

        # Cross-encoder reranking on all candidates (fast for ≤100)
        cids      = [r["candidate_id"] for r in rows]
        row_texts = [r["text"] for r in rows]
        ce_norm   = rerank(jd_text, cids, row_texts)

        for i, r in enumerate(rows):
            feat = r["_feat"]
            exp_fit = compute_experience_fit(feat["years_of_experience"])
            r["ce_score"]    = round(float(ce_norm[i]), 3)
            r["final_score"] = compute_final_score_with_cap(
                r["prelim_score"],
                r["ce_score"],
                exp_fit,
                feat["years_since_last_ml_role"],
            )

        rows.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
        for i, r in enumerate(rows):
            r["rank"] = i + 1
            r["score"] = round(r["final_score"], 4)

        # Generate reasoning (same function as rank.py)
        for r in rows:
            feat = r["_feat"]
            reasoning_row = {**feat, "rank": r["rank"], "ce_score": r["ce_score"]}
            r["reasoning"] = generate_reasoning(reasoning_row)

        # Drop internal field before building DataFrame
        for r in rows:
            del r["_feat"]

        df_out = pd.DataFrame(rows)

    # ── Visual: Score breakdown per candidate (top 20) ───────────────────────
    st.subheader("Score Component Breakdown (top 20)")
    top20 = df_out.head(20)
    fig = px.bar(
        top20,
        x="candidate_id",
        y=["semantic", "structural", "availability", "ce_score"],
        title="Score Components per Candidate",
        barmode="stack",
        labels={"value": "Score", "variable": "Component"},
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Visual: Our top-10 vs keyword-only top-10 ─────────────────────────────
    st.subheader("Our Top-10 vs. Keyword-Only Top-10")
    keyword_top10 = sorted(rows, key=lambda r: -r["semantic"])[:10]
    our_top10     = rows[:10]
    keyword_ids   = {r["candidate_id"] for r in keyword_top10}
    our_ids       = {r["candidate_id"] for r in our_top10}
    overlap       = len(keyword_ids & our_ids)
    st.info(
        f"Overlap between keyword-only and our ranking: **{overlap}/10** candidates. "
        f"We surface **{10 - overlap}** different candidates through structural + reranking."
    )

    # ── Full ranked table ──────────────────────────────────────────────────────
    st.subheader("Full Ranked Output")
    display_cols = ["rank", "candidate_id", "name", "title", "score",
                    "semantic", "structural", "availability", "ce_score"]
    st.dataframe(df_out[display_cols], use_container_width=True)

    # ── Reasoning for top 10 ─────────────────────────────────────────────────
    st.subheader("Reasoning — Top 10")
    for _, row in df_out.head(10).iterrows():
        with st.expander(f"#{row['rank']} {row['candidate_id']} — {row['title']} (score: {row['score']})"):
            st.write(row["reasoning"])

    # ── CSV download ────────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    df_out[["candidate_id", "rank", "score", "reasoning"]].to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name="ranked.csv",
        mime="text/csv",
    )
