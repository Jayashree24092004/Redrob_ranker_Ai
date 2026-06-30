"""
Stage 4 — Semantic Matching.

Design note: the JD's tech-stack wishlist suggests sentence-transformers /
BGE embeddings. Those are the right choice in production, but they require
downloading a model checkpoint, which conflicts with this challenge's "no
network during ranking" rule unless the weights are vendored into the repo
(multi-hundred-MB git artifact) and loaded from disk every run. TF-IDF +
cosine similarity is the pragmatic, fully-offline, dependency-light substitute
that still captures lexical-semantic overlap far better than naive keyword
counting, fits comfortably in the CPU/RAM/time budget for 100K candidates,
and needs nothing beyond scikit-learn. swap-in note: if you vendor a small
sentence-transformers checkpoint (e.g. all-MiniLM-L6-v2, ~90MB) into the repo,
replace `fit_semantic_scores` with batched `model.encode(...)` + cosine sim —
the rest of the pipeline is agnostic to where this score comes from.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_semantic_scores(jd_text, candidate_texts):
    """Returns a list of cosine-similarity scores in [0,1] aligned with
    candidate_texts, scaled by min-max across the batch so the score is
    relative to this candidate pool (avoids everyone clustering near 0,
    which is typical of raw TF-IDF cosine sim against a short JD).

    Unigrams only + a capped vocabulary: bigrams on a 100K-document corpus
    build a vocabulary in the tens of millions of entries before min_df
    pruning kicks in, which blows well past a CPU-only 16GB budget. Unigrams
    with min_df/max_df pruning keep peak memory in the low hundreds of MB
    while still capturing the JD's required-skill terms (most of which are
    matched separately, exactly, in skill_match_score anyway)."""
    vectorizer = TfidfVectorizer(
        max_features=20_000, ngram_range=(1, 1), min_df=3, max_df=0.5,
        stop_words="english", dtype=np.float32,
    )
    corpus = [jd_text] + candidate_texts
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]
    sims = cosine_similarity(jd_vec, cand_vecs)[0]
    lo, hi = sims.min(), sims.max()
    if hi - lo < 1e-9:
        return [0.5] * len(sims)
    return ((sims - lo) / (hi - lo)).tolist()
