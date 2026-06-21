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


def build_candidate_text(c: dict) -> str:
    """
    Build the semantic text representation of a candidate for embedding + BM25.

    Design choices:
    - Skills include proficiency level and duration ("Python (expert, 60mo)")
      so the embedding captures not just skill presence but seniority depth.
    - Recent roles (first 2) are repeated to weight them more heavily in the
      averaged embedding — a crude but effective trick with mean-pooled models.
    - All skills included (not just top 10) — the embedding handles the long tail.
    - Certifications included — direct signal for named ML/cloud tools.
    - Career history capped at 5 roles — older roles add noise, not signal.
    """
    parts = []
    p = c["profile"]

    # Core identity
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))

    # Career history: weight recent roles by repeating descriptions twice
    career = c.get("career_history", [])
    for i, role in enumerate(career[:5]):
        role_text = f"{role.get('title', '')} at {role.get('company', '')}: {role.get('description', '')}"
        parts.append(role_text)
        if i < 2:
            parts.append(role_text)  # repeat 2 most recent roles to up-weight them

    # Skills: sort by proficiency first, then endorsements
    PROFICIENCY_ORDER = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills = sorted(
        c.get("skills", []),
        key=lambda s: (
            PROFICIENCY_ORDER.get(s.get("proficiency", ""), 0),
            s.get("endorsements", 0)
        ),
        reverse=True,
    )
    skill_parts = []
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0) or 0
        skill_parts.append(f"{name} ({prof}, {dur}mo)")
    if skill_parts:
        parts.append("Skills: " + ", ".join(skill_parts))

    # Certifications (directly relevant when they name ML/cloud tools)
    certs = c.get("certifications", [])
    if certs:
        cert_names = [cert.get("name", "") for cert in certs[:5] if cert.get("name")]
        if cert_names:
            parts.append("Certifications: " + ", ".join(cert_names))

    # Education: field of study and institution
    edu = c.get("education", [])
    if edu:
        field = edu[0].get("field_of_study", "")
        inst = edu[0].get("institution", "")
        if field or inst:
            parts.append(f"Education: {field} at {inst}")

    return " ".join(filter(None, parts))