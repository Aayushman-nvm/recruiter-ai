"""
utils.py — Shared candidate text builder.

Imported by:
  - scripts/02_embed_candidates.py  (embedding pre-computation)
  - scripts/05_eval.py              (offline evaluation)

Keeping this in one place eliminates the risk of the two scripts diverging
in how they represent candidates, which would corrupt any comparison.

Reviewed against jd.txt's closing "keyword trap" warning (a candidate with
every AI keyword in their skills list but a "Marketing Manager" title is not
a fit) and Scoring_plan.md: deliberately left build_candidate_text()'s
behavior unchanged here rather than capping/de-weighting the skills list.
Two reasons:
  1. The actual disqualification mechanism for that trap lives downstream in
     the structural layer, not in the embedding text — has_ml_production_experience
     is derived from career_history *descriptions*, not the skills list, and
     is_honeypot explicitly flags implausible skill claims (expert proficiency
     with 0 duration, expert-duration totals exceeding plausible YoE). Dense/
     BM25 retrieval is a recall stage (top-2000 shortlist); precision against
     this exact trap is enforced later, by design.
  2. Changing this function changes the text fed into the embedding model,
     which would silently invalidate any already-computed
     precomputed/candidate_embeddings.npy. If you do change this function,
     re-run scripts/02_embed_candidates.py before the next scripts/rank.py
     run, or fusion/cross-encoder scores will be computed against stale text.
"""

# JD_Narrative_Keywords used to detect whether career descriptions signal
# JD-required technical depth. Mirrors the categories in narrative_embedding_score
# (R1) but used here for the [narrative_signal: low] retrieval hint only.
_JD_NARRATIVE_KEYWORDS = {
    "embedding", "vector", "vector database", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "retrieval", "semantic search", "hybrid search", "hybrid retrieval",
    "dense retrieval", "ndcg", "mrr", "map", "a/b test",
    "evaluation framework", "learning to rank", "reranker",
    "information retrieval",
}


def build_candidate_text(c: dict) -> str:
    """
    Build the semantic text representation of a candidate for embedding + BM25.

    Structure (R6):
      1. [narrative_signal: low] tag — prepended if fewer than 3 career
         descriptions contain any JD_Narrative_Keyword. Gives BM25 and the
         cross-encoder a quality hint without excluding the candidate entirely.
      2. Core identity (headline, summary)
      3. [career narrative] block — role descriptions only, recent 2 repeated
      4. [skills] block — skills with duration_months > 0 only (zero-duration
         skills are suppressed as unverified / honeypot signals)
      5. Certifications
      6. Education

    Design choices:
      - Narrative-first layout ensures BM25 and CE score on what candidates
        actually did, not what they claim to know.
      - Zero-duration skills suppressed — they are honeypot signals or
        unverified self-reported entries.
      - [narrative_signal: low] tag gives the cross-encoder a quality hint
        without removing the candidate from retrieval entirely (skills section
        retained as recall signal per R6.6).
      - Recent roles (first 2) repeated for embedding up-weighting (unchanged).
      - [career narrative] header emitted once at section level; not repeated
        with individual role descriptions.

    ⚠️  Changing this function invalidates precomputed/candidate_embeddings.npy
    and precomputed/candidate_texts.pkl. Re-run scripts/02_embed_candidates.py
    before the next rank.py run.
    """
    parts = []
    p = c["profile"]
    career = c.get("career_history", [])

    # Determine narrative signal quality (R6.3)
    narrative_keyword_hits = sum(
        1 for role in career
        if any(kw in (role.get("description") or "").lower()
               for kw in _JD_NARRATIVE_KEYWORDS)
    )
    if narrative_keyword_hits < 3:
        parts.append("[narrative_signal: low]")

    # Core identity
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))

    # [career narrative] block (R6.1, R6.4)
    # [career narrative] emitted once at section level only
    parts.append("[career narrative]")
    for i, role in enumerate(career[:5]):
        role_text = (f"{role.get('title', '')} at {role.get('company', '')}: "
                     f"{role.get('description', '')}")
        parts.append(role_text)
        if i < 2:
            parts.append(role_text)  # repeat 2 most recent roles for up-weighting

    # [skills] block — zero-duration skills suppressed (R6.2, R6.6)
    parts.append("[skills]")
    PROFICIENCY_ORDER = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills = sorted(
        c.get("skills", []),
        key=lambda s: (
            PROFICIENCY_ORDER.get(s.get("proficiency", ""), 0),
            s.get("endorsements", 0),
        ),
        reverse=True,
    )
    skill_parts = []
    for s in skills:
        dur = s.get("duration_months", 0) or 0
        if dur == 0:
            continue   # suppress zero-duration skills (R6.2)
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        skill_parts.append(f"{name} ({prof}, {dur}mo)")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    # Certifications
    certs = c.get("certifications", [])
    if certs:
        cert_names = [cert.get("name", "") for cert in certs[:5] if cert.get("name")]
        if cert_names:
            parts.append("Certifications: " + ", ".join(cert_names))

    # Education
    edu = c.get("education", [])
    if edu:
        field = edu[0].get("field_of_study", "")
        inst  = edu[0].get("institution", "")
        if field or inst:
            parts.append(f"Education: {field} at {inst}")

    return " ".join(filter(None, parts))