"""
config/keywords.py — All keyword/vocabulary constants for the ranking pipeline.

Includes:
  JD_RELEVANT_SKILLS          — for skill-assessment bonus scoring
  RESEARCH_ONLY_* indicators  — academic/research role detection
  INDUSTRY_RELEVANT_KEYWORDS  — HR-tech / marketplace bonus
  ML_KEYWORDS                 — production ML experience detection
  ML_KEYWORD_PATTERN          — compiled regex (leading word boundary)
  has_ml_keyword()            — convenience function

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
"""

import re

JD_RELEVANT_SKILLS = {
    "python", "nlp", "machine learning", "deep learning",
    "information retrieval", "ranking", "search", "embeddings",
    "recommendation systems", "data science", "pytorch", "tensorflow",
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
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "llm", "fine-tun", "rag", "semantic search", "sentence-transformer",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "bert", "transformer", "nlp", "information retrieval",
    "learning to rank", "xgboost ranking", "neural ranker", "reranker",
    "dense retrieval", "hybrid search", "hybrid retrieval", "search engine",
    "knowledge graph", "question answering", "vector database",
    "approximate nearest neighbor", "hnsw", "cosine similarity",
]

ML_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in ML_KEYWORDS) + r")"
)


def has_ml_keyword(desc: str) -> bool:
    """True if desc (a role description, any case) contains any ML/IR keyword."""
    return ML_KEYWORD_PATTERN.search((desc or "").lower()) is not None
