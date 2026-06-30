"""
Stage 1 — JD Intelligence: structured blueprint extracted (by hand, once) from
job_description.docx. Hand-extraction is deliberate: the JD is a one-time input,
not part of the 100K-candidate hot path, so there's no value in building an
LLM extraction agent for it. Re-run this extraction manually if the JD changes.
"""

JD_FULL_TEXT = """
Senior AI Engineer — Founding Team, Redrob AI (Series A AI-native talent intelligence platform).
Own the intelligence layer: ranking, retrieval, and matching systems deciding what recruiters
see when they search for candidates and what candidates see when they search for roles.
First 90 days: audit existing BM25 + rule-based scoring, ship a v2 ranking system with embeddings,
hybrid retrieval, and LLM-based re-ranking, then build offline/online evaluation infrastructure
(NDCG, MRR, MAP, A/B testing, recruiter-feedback loops). Long-term: architecture for candidate-JD
matching at scale, mentoring as team grows from 4 to 12 engineers.
Wants deep technical depth in modern ML systems (embeddings, retrieval, ranking, LLMs, fine-tuning)
combined with a scrappy product-engineering attitude — ships a working ranker in a week even if
the ML is suboptimal. Tilts toward shipper over pure researcher.
"""

REQUIRED_SKILL_TERMS = [
    "embeddings", "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5 embeddings", "vector database", "vector db", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "hybrid search",
    "hybrid retrieval", "retrieval", "ranking", "recommendation system", "recommender system",
    "search system", "search infrastructure", "ndcg", "mrr", "map", "precision at k",
    "a/b test", "ab testing", "offline evaluation", "online evaluation", "evaluation framework",
    "learning to rank", "information retrieval", "semantic search",
]

PREFERRED_SKILL_TERMS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "xgboost", "neural ranking",
    "hr tech", "hrtech", "recruiting tech", "marketplace", "distributed systems",
    "large-scale inference", "inference optimization", "open source", "open-source",
    "rag", "retrieval augmented generation",
]

# Used for the "recent LangChain-only AI exp" disqualifier check.
PRE_LLM_ML_TERMS = [
    "scikit-learn", "sklearn", "tensorflow", "pytorch", "nlp", "natural language processing",
    "search", "ranking", "recommendation", "machine learning", "ml", "information retrieval",
    "bm25", "tf-idf", "word2vec", "elasticsearch", "solr",
]

LANGCHAIN_SHALLOW_TERMS = ["langchain", "openai api", "gpt wrapper", "chatgpt api"]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini",
]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection", "speech recognition",
    "speech-to-text", "text-to-speech", "robotics", "autonomous driving", "slam",
    "image segmentation", "ocr",
]

NLP_IR_TERMS = [
    "nlp", "natural language processing", "search", "retrieval", "ranking", "information retrieval",
    "embeddings", "language model", "text classification", "named entity",
]

SENIOR_TITLE_RANK = {
    "intern": 0, "trainee": 0,
    "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3,
    "staff": 4, "lead": 4, "tech lead": 4,
    "principal": 5, "architect": 5, "director": 5, "head": 5, "vp": 6, "founder": 6, "cto": 6,
    "manager": 4,
}

NON_CODING_TITLE_TERMS = ["architect", "tech lead", "engineering manager", "director", "vp", "head of"]

TIER1_INDIAN_LOCATIONS = [
    "pune", "noida", "bangalore", "bengaluru", "mumbai", "delhi", "ncr", "gurgaon",
    "gurugram", "hyderabad",
]

EDU_RELEVANT_FIELDS = [
    "computer science", "computer engineering", "data science", "machine learning",
    "artificial intelligence", "statistics", "mathematics", "information technology",
    "software engineering", "electrical engineering",
]

TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}

MAX_NOTICE_PREFERRED_DAYS = 30
