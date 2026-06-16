"""
sandbox/app.py — Streamlit demo for the Redrob Candidate Ranker.

Usage: streamlit run sandbox/app.py

Handles ≤100 candidates uploaded as a JSON array.
Imports all scoring logic from scoring.py — no formula divergence from rank.py.
Embeds on the fly (no precomputed files needed).
Runs cross-encoder since ≤100 candidates is fast enough for live demo.
"""

import io
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing scoring.py and utils.py from parent directory
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

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker — Demo")
st.caption("Upload ≤100 candidates (JSON array) and rank them against the JD query.")

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

ML_KEYWORDS = [
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "llm", "rag", "semantic search", "faiss", "pinecone", "bert", "transformer", "nlp",
]


@st.cache_resource
def load_models():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return encoder, reranker


encoder, reranker = load_models()

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
# Feature extraction helper (inline — avoids needing features.parquet)
# ─────────────────────────────────────────────────────────────────────────────


def extract_candidate_features(c: dict) -> dict:
    """
    Extract scoring features inline for a single candidate.
    Mirrors Script 01 logic. years_since_last_ml_role hardcoded to 0
    (full date parsing not worth the complexity for a ≤100 demo).
    """
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    education = c.get("education", [])

    country = (p.get("country") or "").strip().lower()
    location = (p.get("location") or "").strip().lower()
    is_india = country == "india"
    in_target = any(tc in location for tc in TARGET_CITIES)

    co_names = [(r.get("company") or "").lower() for r in career]
    all_it = all(
        any(it in co for it in IT_SERVICES_COMPANIES) for co in co_names
    ) if co_names else False
    has_product = any(
        not any(it in co for it in IT_SERVICES_COMPANIES) for co in co_names
    )

    desc_text = " ".join((r.get("description") or "") for r in career[:3]).lower()
    has_ml = any(kw in desc_text for kw in ML_KEYWORDS)

    sal = (sig.get("expected_salary_range_inr_lpa") or {})
    sal_min = float(sal.get("min", 0) or 0)
    sal_max = float(sal.get("max", 999) or 999)

    skill_bonus = compute_skill_assessment_bonus(sig.get("skill_assessment_scores") or {})
    edu_tier  = education[0].get("tier", "unknown") if education else "unknown"
    edu_field = education[0].get("field_of_study", "") if education else ""
    edu_b = compute_education_bonus(edu_tier, edu_field)

    # Trajectory: simplified — no date parsing for demo
    durations = [r.get("duration_months", 0) or 0 for r in career]
    avg_tenure = float(sum(durations) / len(durations)) if durations else 0.0
    tenure_stab = 1.0 if avg_tenure > 24 else (0.7 if avg_tenure > 12 else 0.3)
    traj_score = 0.4 * tenure_stab + 0.3 * float(has_product) + 0.3 * float(has_ml)

    try:
        last_active = date.fromisoformat(str(sig.get("last_active_date", ""))[:10])
        days_ago = (REFERENCE_DATE - last_active).days
    except Exception:
        days_ago = 999

    return {
        "yoe":            float(p.get("years_of_experience", 0) or 0),
        "is_india":       is_india,
        "in_target":      in_target,
        "relocate":       bool(sig.get("willing_to_relocate", False)),
        "all_it":         all_it,
        "has_product":    has_product,
        "has_ml":         has_ml,
        "sal_min":        sal_min,
        "sal_max":        sal_max,
        "skill_bonus":    skill_bonus,
        "edu_bonus":      edu_b,
        "traj_score":     traj_score,
        "open_to_work":   bool(sig.get("open_to_work_flag", False)),
        "days_ago":       days_ago,
        "resp_rate":      float(sig.get("recruiter_response_rate", 0.5) or 0.5),
        "resp_time":      float(sig.get("avg_response_time_hours") if sig.get("avg_response_time_hours") is not None else -1),
        "notice":         int(sig.get("notice_period_days", 90) or 90),
        "saves":          int(sig.get("saved_by_recruiters_30d", 0) or 0),
        "v_email":        bool(sig.get("verified_email", False)),
        "v_phone":        bool(sig.get("verified_phone", False)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────


if st.button("Rank") and candidates_file and jd_text:
    with st.spinner("Embedding, scoring, reranking..."):
        candidates = json.load(candidates_file)

        if len(candidates) > 100:
            st.warning(f"Loaded {len(candidates)} candidates — truncating to first 100 for the demo.")
            candidates = candidates[:100]

        texts = [build_candidate_text(c) for c in candidates]

        # Embed candidates (plain text — BGE asymmetric)
        cand_embeddings = encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        # Embed JD query (with instruction prefix — BGE asymmetric)
        jd_embedding = encoder.encode(
            BGE_QUERY_INSTRUCTION + jd_text,
            normalize_embeddings=True, convert_to_numpy=True
        )
        dense_scores = cand_embeddings @ jd_embedding

        rows = []
        for i, c in enumerate(candidates):
            p = c["profile"]
            feat = extract_candidate_features(c)

            exp_fit  = compute_experience_fit(feat["yoe"])
            loc_fit  = compute_location_fit(feat["is_india"], feat["in_target"], feat["relocate"])
            comp_fit = compute_company_fit(
                feat["all_it"], feat["has_product"], feat["has_ml"],
                0.0,  # years_since_last_ml_role — not computed in sandbox
            )
            sal_fit  = compute_salary_fit(feat["sal_min"], feat["sal_max"])
            struct   = compute_structural_score(
                exp_fit, loc_fit, comp_fit, feat["traj_score"],
                sal_fit, feat["skill_bonus"], feat["edu_bonus"]
            )
            avail = compute_availability_score(
                feat["open_to_work"], feat["days_ago"], feat["resp_rate"],
                feat["resp_time"], feat["notice"], feat["saves"],
                feat["v_email"], feat["v_phone"]
            )

            sem = float(dense_scores[i])
            prelim = 0.50 * max(sem, 0.0) + 0.35 * struct + 0.15 * avail

            rows.append({
                "candidate_id": c["candidate_id"],
                "name":    p.get("anonymized_name", ""),
                "title":   p.get("current_title", ""),
                "prelim_score":  prelim,
                "semantic":      round(max(sem, 0.0), 3),
                "structural":    round(struct, 3),
                "availability":  round(avail, 3),
                "text":          texts[i],
            })

        # Cross-encoder reranking (fast for ≤100 candidates)
        pairs = [(jd_text, r["text"]) for r in rows]
        ce_scores = reranker.predict(pairs, batch_size=32)
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        ce_norm = (ce_scores - ce_min) / (ce_max - ce_min + 1e-9)

        for i, r in enumerate(rows):
            r["ce_score"] = round(float(ce_norm[i]), 3)
            r["score"]    = round(0.40 * r["prelim_score"] + 0.60 * r["ce_score"], 4)

        rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
        for i, r in enumerate(rows):
            r["rank"] = i + 1

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

    # ── Visual: Our top-10 vs keyword-only top-10 ────────────────────────────
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

    # ── Full ranked table ─────────────────────────────────────────────────────
    st.subheader("Full Ranked Output")
    display_cols = ["rank", "candidate_id", "name", "title", "score",
                    "semantic", "structural", "availability", "ce_score"]
    st.dataframe(df_out[display_cols], use_container_width=True)

    # ── CSV download ──────────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    df_out[["candidate_id", "rank", "score"]].to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name="ranked.csv",
        mime="text/csv",
    )
