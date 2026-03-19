"""
keywords.py — Matched and missing keyword analysis.

After scoring, we need to explain what drove the score.
This module answers:
  - Which important job description terms appear in the resume? (matched)
  - Which important job description terms are missing? (gaps to address)
  - What are the most distinctive terms in the resume overall?

Why is this useful?
  Many ATS (Applicant Tracking Systems) do exactly this: they scan resumes for
  keywords from the job description. By showing which ones are missing, we help
  the candidate improve their chances.
"""

import numpy as np
from src.scorer import get_top_tfidf_terms


def get_job_keywords(job_vector, feature_names: np.ndarray, top_n: int = 30) -> list[str]:
    """
    Extract the most important terms from the job description's TF-IDF vector.
    These represent what the job is actually looking for.
    """
    top_terms = get_top_tfidf_terms(job_vector, feature_names, top_n=top_n)
    return [term for term, _ in top_terms]


def analyze_keywords(
    resume_text: str,
    job_vector,
    feature_names: np.ndarray,
    top_n: int = 30,
) -> dict:
    """
    Compare job description keywords against the resume text.

    Args:
        resume_text: the preprocessed (cleaned) resume text
        job_vector: TF-IDF vector of the job description
        feature_names: vocabulary from the TF-IDF vectorizer
        top_n: how many top job keywords to check

    Returns:
        dict with:
          - matched: list of keywords found in both
          - missing: list of keywords from job not in resume
          - job_keywords: all top job keywords considered
    """
    job_keywords = get_job_keywords(job_vector, feature_names, top_n=top_n)

    # Convert resume text to a set of words for fast lookup
    resume_words = set(resume_text.lower().split())

    matched = []
    missing = []

    for keyword in job_keywords:
        # A keyword may be a phrase ("machine learning") — check if all its
        # words appear somewhere in the resume text
        keyword_words = set(keyword.lower().split())
        if keyword_words.issubset(resume_words):
            matched.append(keyword)
        else:
            missing.append(keyword)

    return {
        "matched": matched,
        "missing": missing,
        "job_keywords": job_keywords,
    }


def get_resume_highlights(resume_vector, feature_names: np.ndarray, top_n: int = 15) -> list[tuple[str, float]]:
    """
    Return the most distinctive terms in the resume according to TF-IDF.
    These are the terms the resume 'leads with' — what it's most about.
    """
    return get_top_tfidf_terms(resume_vector, feature_names, top_n=top_n)
