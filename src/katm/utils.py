"""Utility functions for KATM package."""

import re
import unicodedata
from collections import Counter
from typing import List, Set

import nltk

# Ensure NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_stopwords = set(nltk.corpus.stopwords.words("english"))

# Extra tokens that carry no topical signal (headers, punctuation artefacts,
# newsgroup/web boilerplate) that TF-IDF would otherwise promote.
_EXTRA_STOP = {
    "edu", "com", "org", "net", "gov", "www", "http", "re", "cc", "writes",
    "article", "subject", "lines", "like", "just", "don", "use", "know",
    "think", "time", "want", "good", "make", "look", "said", "say", "way",
    "does", "did", "got", "ll", "ve", "haven", "isn", "aren", "wasn",
    "doesn", "didn", "wouldn", "couldn", "shouldn",
}
_ALL_STOP = _stopwords | _EXTRA_STOP


def clean_text(text: str) -> str:
    """Strip extra whitespace, normalize unicode, basic cleaning.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    # Normalize unicode (NFKD normalization)
    text = unicodedata.normalize("NFKD", text)
    # Remove non-printable characters
    text = "".join(char for char in text if char.isprintable())
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vocabulary(texts: List[str], min_df: int = 3) -> Set[str]:
    """Tokenize and return words appearing at least min_df times.

    Args:
        texts: List of raw text strings.
        min_df: Minimum document frequency threshold.

    Returns:
        Set of words meeting the frequency threshold.
    """
    word_counts = Counter()
    for text in texts:
        # Simple whitespace tokenization, lowercased
        tokens = text.lower().split()
        word_counts.update(tokens)

    return {word for word, count in word_counts.items() if count >= min_df}


def extract_content_words(texts: List[str], min_df: int = 3,
                          method: str = "tfidf") -> List[str]:
    """Extract content words from a corpus.

    Args:
        texts:  List of raw text strings.
        min_df: Minimum document frequency threshold.
        method: ``"tfidf"`` (default) or ``"spacy"``.

                ``"tfidf"`` — fits a CountVectorizer and returns alphabetic
                unigrams that pass stopword and sanity filters.  ~300× faster
                than spaCy on large corpora; ~84% vocabulary overlap.

                ``"spacy"`` — POS-tags every document with en_core_web_sm and
                keeps nouns, verbs, adjectives, adverbs, and proper nouns
                (lemmatised).  More linguistically precise but very slow on
                corpora > ~5,000 docs.

    Returns:
        Deduplicated sorted list of content words.
    """
    if method == "spacy":
        return _extract_content_words_spacy(texts, min_df)
    return _extract_content_words_tfidf(texts, min_df)


def _extract_content_words_tfidf(texts: List[str], min_df: int) -> List[str]:
    import re as _re
    from sklearn.feature_extraction.text import CountVectorizer

    _bad_token = _re.compile(r"(.)\1{2,}|.{26,}")
    n = len(texts)
    vec = CountVectorizer(
        ngram_range=(1, 1),
        stop_words="english",
        min_df=min(min_df, n),
        max_df=0.98,
        token_pattern=r"(?u)\b[a-zA-Z]{4,}\b",
    )
    vec.fit(texts)
    vocab = vec.get_feature_names_out()
    return sorted(
        w for w in vocab
        if w.lower() not in _ALL_STOP and not _bad_token.search(w)
    )


def _extract_content_words_spacy(texts: List[str], min_df: int) -> List[str]:
    import spacy
    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "senter"])

    word_doc_count: Counter = Counter()
    for doc in nlp.pipe(texts, batch_size=64, n_process=1):
        doc_words: set = set()
        for token in doc:
            if token.pos_ in _CONTENT_POS:
                w = token.lemma_.lower()
                if w.isalpha() and len(w) >= 2 and w not in _stopwords:
                    doc_words.add(w)
        word_doc_count.update(doc_words)

    return sorted(w for w, cnt in word_doc_count.items() if cnt >= min_df)