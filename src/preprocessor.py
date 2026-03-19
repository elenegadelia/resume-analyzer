"""
preprocessor.py — Clean and normalize text before NLP analysis.

Why preprocess? Raw text has noise: punctuation, capitals, filler words.
Preprocessing ensures "Python", "python", and "PYTHON" all count as the same word,
and that common words like "the" don't skew our similarity scores.

Pipeline steps:
  1. Lowercase everything
  2. Remove special characters and punctuation
  3. Tokenize (split into words)
  4. Remove stopwords (filler words like "the", "and", "is")
  5. Lemmatize (reduce words to their base form: "running" → "run")
"""

import re
import spacy

# Load the small English model. This gives us:
# - Tokenization (splitting into words)
# - Lemmatization (finding root forms)
# - POS tagging (part of speech) — useful for filtering
# - Named Entity Recognition (NER) — used in keywords.py
_nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Basic string cleaning before spaCy processing.
    - Lowercase
    - Remove URLs
    - Replace newlines/tabs with spaces
    - Remove characters that aren't letters, digits, or spaces
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[\n\r\t]", " ", text)               # flatten whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # keep only alphanumeric
    text = re.sub(r"\s+", " ", text).strip()             # collapse multiple spaces
    return text


def preprocess(text: str, lemmatize: bool = True) -> str:
    """
    Full preprocessing pipeline. Returns a cleaned, normalized string.

    This is the version used for TF-IDF vectorization.
    The output is a single string of meaningful tokens, space-separated.

    Args:
        text: raw input text
        lemmatize: if True, reduce words to their root form

    Returns:
        cleaned string ready for vectorization
    """
    text = clean_text(text)
    doc = _nlp(text)

    tokens = []
    for token in doc:
        # Skip stopwords (e.g., "the", "is", "and")
        if token.is_stop:
            continue
        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue
        # Skip very short tokens (1 character) — usually noise
        if len(token.text) <= 1:
            continue

        word = token.lemma_ if lemmatize else token.text
        tokens.append(word)

    return " ".join(tokens)


def light_clean(text: str) -> str:
    """
    Lightweight cleaning for TF-IDF vectorization.

    We intentionally do NOT lemmatize here, so that multi-word phrases like
    "machine learning" stay intact and match correctly across documents.
    The TfidfVectorizer handles stopword removal internally.

    Use this function when passing text to compute_similarity().
    Use preprocess() when you need lemmatized tokens for deeper analysis.
    """
    return clean_text(text)


def get_tokens(text: str) -> list[str]:
    """
    Return a list of meaningful tokens from the text.
    Useful for keyword analysis (not just a joined string).
    """
    processed = preprocess(text)
    return processed.split()


def get_named_entities(text: str) -> dict[str, list[str]]:
    """
    Use spaCy's Named Entity Recognition (NER) to find entities
    like organizations, skills (as nouns), and locations.

    NER is a technique where the model identifies real-world named things:
    "Google" → ORG, "New York" → GPE (geopolitical entity)

    We use this for the 'extracted information' section of the UI.
    """
    doc = _nlp(text[:100_000])  # spaCy has a character limit for efficiency

    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        label = ent.label_
        value = ent.text.strip()
        if label not in entities:
            entities[label] = []
        if value not in entities[label]:
            entities[label].append(value)

    return entities
