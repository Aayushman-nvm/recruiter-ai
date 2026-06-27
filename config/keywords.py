"""
config/keywords.py — All keyword/vocabulary constants for the ranking pipeline.

Includes:
  SKILL_ASSESSMENT_JD_RELEVANT  — for skill-assessment bonus scoring (replaces JD_RELEVANT_SKILLS)
  DISQUALIFYING_PHRASES         — explicit non-ownership / delegation phrases in career narrative
  GHOST_SKILL_KEYWORDS          — JD-relevant skills that may appear in skills list without narrative evidence
  CV_SPEECH_DOMAIN_TITLES       — role titles indicating computer vision / speech / robotics primary domain
  NLP_IR_CROSSOVER_KEYWORDS     — keywords indicating meaningful NLP/IR crossover in narrative
  RESEARCH_ONLY_* indicators    — academic/research role detection
  INDUSTRY_RELEVANT_KEYWORDS    — HR-tech / marketplace bonus
  ML_KEYWORDS                   — production ML experience detection
  ML_KEYWORD_PATTERN            — compiled regex (leading word boundary)
  has_ml_keyword()              — convenience function

Design notes for ML_KEYWORDS:
  - Dropped bare "ann": caused ~52% false positives on unrelated titles
    ("channel", "planning", "announce" all contain "ann" as a substring).
    "approximate nearest neighbor" and "hnsw" cover the legitimate case.
  - Uses leading word boundary (\\b before keyword, not after): stems like
    "fine-tun" and "sentence-transformer" are meant to match
    "fine-tuning"/"sentence-transformers", so no trailing boundary is set.
    A leading boundary stops "llm" matching inside "fulfillment" and
    "bert" matching inside "Robert"/"Albert".
  - "opensearch", "hybrid search", "hybrid retrieval" added — jd_query.txt
    names these explicitly as required infra/technique.
  - TIGHTENED (false-positive audit): four bare terms were matching common
    non-ML usage and inflating has_ml_production_experience on candidates
    with no real ML/IR background. jd.txt and jd_query.txt always use these
    concepts in ML-specific compound forms, never bare — so requiring the
    fuller phrase costs no real recall against this JD:
      "transformer"   → false-positives on electrical/industrial transformers
                         ("transformer station", "power transformer"). The
                         JD only ever says "sentence-transformers" or "BERT/
                         transformer-based model" — replaced with
                         "sentence-transformer" (already present) and added
                         "transformer model"/"transformer-based" as the
                         legitimate compound forms.
      "embedding"      → false-positives on "embedded systems", "embedding
                         firmware" (hardware/IoT contexts). jd.txt always
                         says "embeddings" (plural, as a retrieval concept)
                         or "embedding drift"/"embeddings-based" — replaced
                         bare "embedding" with "embeddings" (plural) plus
                         "embedding drift" / "embedding model" as compounds.
      "ranking"        → false-positives on generic business usage ("ranking
                         criteria for procurement", "sales ranking dashboard").
                         jd.txt always pairs it with a system/algorithm noun
                         ("ranking system", "learning to rank", "re-ranking").
                         Replaced bare "ranking" with "ranking system",
                         "re-ranking", "reranking" (learning-to-rank already
                         covers the other legitimate case).
      "search engine"  → ambiguous with SEO/marketing usage ("search engine
                         optimization", "search engine marketing") AND already
                         appears in INDUSTRY_RELEVANT_KEYWORDS with that exact
                         meaning. Removed from ML_KEYWORDS entirely — "hybrid
                         search"/"semantic search"/"dense retrieval" already
                         cover the legitimate ML-search case without the SEO
                         collision.
  - "vector" and "retrieval" (bare) were left as-is: spot-checked against
    plausible false positives ("vector" in non-ML engineering contexts is
    rare enough in resume narrative text to not be worth tightening; same
    for "retrieval" outside of records/document retrieval, which is itself
    adjacent enough to IR to be a reasonable signal).
"""

import re

# Replaces JD_RELEVANT_SKILLS (R4)
SKILL_ASSESSMENT_JD_RELEVANT = {
    "embeddings", "vector database", "information retrieval", "ranking",
    "semantic search", "nlp", "natural language processing", "search",
    "retrieval", "python", "machine learning", "deep learning", "pytorch",
    "tensorflow", "fine-tuning", "lora", "peft", "learning to rank",
    "recommendation", "bm25", "faiss", "elasticsearch", "opensearch",
}

# R2: used by feature_extraction.py
DISQUALIFYING_PHRASES = [
    "deployment was handled by the platform team",
    "deployment was handled by",
    "production deployment was handled",
    "my role was more on modeling",
    "my role was more focused on modeling",
    "still building depth on the engineering side",
    "still developing depth on the engineering side",
    "primarily on the research side",
    "i was not responsible for",
    "handled by the infra team",
    "handled by the infrastructure team",
    "handled by ops",
    "engineering was owned by",
]

# R3: used by feature_extraction.py
GHOST_SKILL_KEYWORDS = {
    "pinecone", "qdrant", "weaviate", "milvus", "faiss",
    "elasticsearch", "opensearch", "bm25", "embeddings",
    "vector database", "semantic search", "hybrid search",
    "hybrid retrieval", "dense retrieval", "ndcg", "mrr", "map",
}

# R5: used by feature_extraction.py
CV_SPEECH_DOMAIN_TITLES = {
    "computer vision engineer", "cv engineer", "vision engineer",
    "speech engineer", "speech recognition", "asr engineer",
    "robotics engineer",
}

NLP_IR_CROSSOVER_KEYWORDS = {
    "nlp", "natural language", "information retrieval", "search",
    "ranking", "recommendation", "text", "language model",
}

RESEARCH_ONLY_COMPANY_INDICATORS = {
    "university", "institute of technology", "iisc",
    "indian institute of science", "research institute",
    "academy of sciences", "college of engineering",
}

RESEARCH_ONLY_TITLE_INDICATORS = {
    "phd", "postdoc", "post-doc", "doctoral researcher", "research fellow",
    "research scientist", "research intern", "graduate researcher",
}

# jd.txt nice-to-have: "Prior exposure to HR-tech, recruiting tech, or
# marketplace products." Deliberately avoids bare "search" (substring of
# "Research") and other collision-prone substrings.
INDUSTRY_RELEVANT_KEYWORDS = {
    "hr tech", "hrtech", "human resources", "recruit", "talent",
    "staffing", "marketplace", "e-commerce", "ecommerce",
    "search engine", "classifieds", "job board", "jobs platform",
}

ML_KEYWORDS = [
    "vector", "retrieval", "recommendation system", "recommender system",
    "recommendation engine", "recommendations pipeline", "recommender pipeline",
    "collaborative filtering", "matrix factorization",
    "llm", "fine-tun", "rag", "semantic search", "sentence-transformer",
    "transformer model", "transformer-based", "transformer architecture",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "bert", "nlp", "information retrieval",
    "embeddings", "embedding drift", "embedding model", "embedding-based",
    "learning to rank", "xgboost ranking", "neural ranker", "reranker",
    "re-ranking", "reranking", "ranking system", "ranking algorithm",
    "feed ranker", "content ranker", "search ranker", "ranking model",
    "dense retrieval", "hybrid search", "hybrid retrieval",
    "knowledge graph", "question answering", "vector database",
    "approximate nearest neighbor", "hnsw", "cosine similarity",
]

ML_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in ML_KEYWORDS) + r")"
)


def has_ml_keyword(desc: str) -> bool:
    """True if desc (a role description, any case) contains any ML/IR keyword."""
    return ML_KEYWORD_PATTERN.search((desc or "").lower()) is not None