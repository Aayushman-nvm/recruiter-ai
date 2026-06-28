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
    "machine learning", "deep learning", "pytorch", "tensorflow",
    "scikit-learn",
]

ML_KEYWORDS_BOTH_BOUNDARIES = ["ml", "sklearn"]

ML_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in ML_KEYWORDS) + r")"
    r"|\b(?:" + "|".join(re.escape(kw) for kw in ML_KEYWORDS_BOTH_BOUNDARIES) + r")\b"
)


def has_ml_keyword(desc: str) -> bool:
    """True if desc (a role description, any case) contains any ML/IR keyword."""
    return ML_KEYWORD_PATTERN.search((desc or "").lower()) is not None