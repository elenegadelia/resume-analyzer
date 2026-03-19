"""
scorer.py — TF-IDF vectorization and cosine similarity scoring.

CONCEPT — TF-IDF (Term Frequency–Inverse Document Frequency):
  - TF: how often a word appears in one document (resume or job description)
  - IDF: how rare that word is across all documents (penalizes common words)
  - Multiply them: words that appear often in THIS doc but rarely elsewhere = important

CONCEPT — Cosine Similarity:
  - Each document becomes a vector (a list of numbers, one per word)
  - We measure the angle between the two vectors
  - cos(0°) = 1.0 → identical documents
  - cos(90°) = 0.0 → completely different documents
  - We multiply by 100 to get a percentage

Why not use transformers (like BERT/GPT)?
  - TF-IDF is fast, transparent, and explainable
  - For keyword-level matching, it's highly effective
  - Transformers add semantic understanding but are overkill for this use case
    and require much more setup and compute
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_similarity(resume_text: str, job_text: str) -> dict:
    """
    Vectorize both texts with TF-IDF and compute a composite match score.

    We use TWO complementary metrics:

    1. TF-IDF Cosine Similarity — the classic NLP approach.
       Measures overall document-level similarity in vector space.
       Works best with large corpora; with 2 docs it can produce conservative scores.

    2. Keyword Overlap Score — practical and highly interpretable.
       "What fraction of the job's important terms appear in the resume?"
       This directly mirrors how ATS (Applicant Tracking Systems) work.

    Final score = weighted blend of both:
      score = 0.4 * tfidf_cosine + 0.6 * keyword_overlap

    This blend is honest (both metrics are real ML/NLP techniques) and produces
    scores that are intuitive and explainable to a non-technical audience.

    Args:
        resume_text: lightly cleaned resume text (lowercase, no punctuation)
        job_text: lightly cleaned job description text

    Returns:
        dict with score, vectorizer, vectors, feature_names, and component scores
    """
    # TfidfVectorizer with stop_words handles English stopwords internally.
    # ngram_range=(1, 2) captures both single words and two-word phrases.
    # We use the raw (not lemmatized) cleaned text so ngrams like
    # "machine learning" stay intact.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        sublinear_tf=True,   # apply log normalization to term frequency (reduces impact of very frequent terms)
    )

    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    resume_vector = tfidf_matrix[0]
    job_vector = tfidf_matrix[1]

    # ── Metric 1: TF-IDF Cosine Similarity ──────────────────────────────────
    tfidf_score = float(cosine_similarity(resume_vector, job_vector)[0][0]) * 100

    # ── Metric 2: Keyword Overlap Score ─────────────────────────────────────
    # Get top N terms from the job vector (what the job cares about most)
    feature_names = vectorizer.get_feature_names_out()
    job_scores = job_vector.toarray().flatten()

    # Take the top 30 job terms by TF-IDF weight
    top_indices = job_scores.argsort()[-30:][::-1]
    top_job_terms = [feature_names[i] for i in top_indices if job_scores[i] > 0]

    # Check how many of those terms appear in the resume text
    resume_words = set(resume_text.lower().split())
    overlap_count = 0
    for term in top_job_terms:
        term_words = set(term.lower().split())
        if term_words.issubset(resume_words):
            overlap_count += 1

    keyword_overlap = (overlap_count / len(top_job_terms) * 100) if top_job_terms else 0

    # ── Composite Score ──────────────────────────────────────────────────────
    composite_score = round(0.4 * tfidf_score + 0.6 * keyword_overlap, 2)

    return {
        "score": composite_score,
        "tfidf_score": round(tfidf_score, 2),
        "keyword_overlap": round(keyword_overlap, 2),
        "vectorizer": vectorizer,
        "resume_vector": resume_vector,
        "job_vector": job_vector,
        "feature_names": feature_names,
    }


def get_top_tfidf_terms(vector, feature_names: np.ndarray, top_n: int = 20) -> list[tuple[str, float]]:
    """
    Extract the top N terms from a TF-IDF vector by their weight.

    Higher weight = more important/distinctive term in this document.
    Returns a list of (term, weight) tuples sorted by weight descending.
    """
    # Convert sparse matrix to a flat array
    scores = vector.toarray().flatten()

    # Pair each term with its score, filter out zeros, sort by score
    term_scores = [
        (feature_names[i], round(float(scores[i]), 4))
        for i in range(len(scores))
        if scores[i] > 0
    ]
    term_scores.sort(key=lambda x: x[1], reverse=True)
    return term_scores[:top_n]
