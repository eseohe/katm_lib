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

# Standard German alphabet only (a-z + äöüß). Only meaningful for
# language="german" content extraction — a no-op character-set filter isn't
# applicable to English, where loanwords (café, naïve) are common enough
# that the same strict filter would be wrong.
_GERMAN_ALPHABET_RE = re.compile(r"^[a-zäöüß]+$")


def _is_german_alphabet(word: str) -> bool:
    """True if every character in `word` (lowercased) is a standard German
    letter. Multi-word phrases are checked word-by-word."""
    return all(_GERMAN_ALPHABET_RE.match(w) for w in word.lower().split())


def _stopwords_for(language: str) -> set:
    """Stopword set for the given language, via NLTK's multilingual stopword
    corpus. ``"english"`` (the default everywhere in KATM) returns the same
    232-word list already loaded above. Any other NLTK-supported language
    code (e.g. ``"german"``, ``"french"``) returns that language's plain
    NLTK list — no extra augmentation beyond what NLTK ships, and falls back
    to the English list if NLTK doesn't recognize the code. Used by
    keyphrase extraction (per-document KeyBERT's stop_words and
    keyphrase_scope="global"'s candidate-vocabulary filter); content-word
    extraction (extract_content_words below) is English-only for now.
    """
    if language == "english":
        return set(_stopwords)
    try:
        return set(nltk.corpus.stopwords.words(language))
    except OSError:
        return set(_stopwords)


_SPACY_MODEL = {"english": "en_core_web_sm", "german": "de_core_news_sm", "french": "fr_core_news_sm"}


def lemma_key_map(words: List[str], language: str = "english") -> dict:
    """Map each word to a within-topic dedup key, batch-lemmatized via spaCy
    with no sentence context (only isolated words are available at this
    stage - content_words is already a flat, deduplicated vocabulary, not
    tied back to the sentences it came from).

    This is deliberately a *dedup key*, not a replacement for the returned
    topic word: callers keep displaying `words` as-is and only use this map
    to decide whether two candidates are "the same word" for the purposes of
    excluding both from one topic's word list. Used by
    topic_assignment.WordTopicProjector.project_vocabulary's lemma_dedup.

    Empirically, isolated-token spaCy lemmatization recovers most
    inflectional variants correctly even without sentence context, though
    not all - strong-verb ablaut pairs can still land on different keys.
    It's a real improvement over no dedup, not a perfect lemmatizer.
    """
    import spacy
    model_name = _SPACY_MODEL.get(language, language)
    nlp = spacy.load(model_name, disable=["parser", "ner", "senter"])
    keys = {}
    for w, doc in zip(words, nlp.pipe(words, batch_size=256)):
        keys[w] = doc[0].lemma_.lower() if len(doc) else w.lower()
    return keys


# Experimental (see build_noun_final_vocab's exclude_abstract_suffixes):
# German abstract-nominalization suffixes. -ung/-heit/-keit/-nis/-schaft/
# -tum are the classic, highly productive deverbal/deadjectival
# abstract-noun formants (Verwunderung, Dankbarkeit, Einsamkeit, Verständnis,
# Freundschaft, Reichtum) - grammatically NOUN, so they pass the noun-final
# filter below untouched, but semantically closer to an adjective/verb than
# a concrete referent. Known false positives exist (Zeitung "newspaper",
# Mannschaft "team/crew", Bistum "diocese" are concrete nouns that happen to
# carry these suffixes) - this is a blunt, suffix-only heuristic, not a
# semantic classifier, kept opt-in and exploratory for exactly that reason.
_GERMAN_ABSTRACT_SUFFIXES = ("ung", "heit", "keit", "nis", "schaft", "tum")


def _ends_in_abstract_suffix(word: str) -> bool:
    w = word.lower()
    return any(w.endswith(suf) and len(w) > len(suf) + 2 for suf in _GERMAN_ABSTRACT_SUFFIXES)


def build_noun_final_vocab(documents, language: str, ngram_range: tuple,
                            exclude_abstract_suffixes: bool = False) -> set:
    """Shared candidate-restriction vocabulary: n-grams that are right-headed
    noun phrases (last token NOUN/PROPN), stopword/punct/non-alpha tokens
    excluded from any span. Used by keyphrase_extractor.KeyphraseExtractor's
    keybert_candidate_pos_filter to restrict KeyBERT's own candidate
    generation (not just the final display vocabulary) to noun-final
    phrases - diagnosed as the actually-effective place to apply this
    restriction: filtering the display vocabulary afterward left the anchor
    pool (and therefore the GMM cluster centroids themselves) full of
    pure-verb/adverb candidates, which produced register/grammatical-class
    clusters (all speech verbs, all reproach words) instead of subject-
    matter ones.

    Right-headedness (last token, not "any token") is deliberate: it rules
    out phrases like "tripped doubtless" or "escape clouded" (last token an
    adverb/participle) while keeping phrases like "mill wall" or "aeroplane"
    (last token a concrete noun) - the same heuristic that makes "the tall
    man" a noun phrase and "very tall" not.

    exclude_abstract_suffixes: False (default, unchanged behavior). If
    True, additionally drops any span whose head (last token) ends in a
    German abstract-nominalization suffix (see _GERMAN_ABSTRACT_SUFFIXES) -
    experimental. No-op for language != "german" regardless of this flag
    (the suffix list is German-specific).

    Returns a set of lowercased, space-joined n-gram strings. Empty set if
    spaCy finds nothing (caller should fall back to an unrestricted
    vectorizer, not silently produce zero candidates).
    """
    import spacy

    model_name = _SPACY_MODEL.get(language, language)
    nlp = spacy.load(model_name, disable=["ner", "parser"])
    n_min, n_max = ngram_range
    drop_suffixes = exclude_abstract_suffixes and language == "german"

    vocab = set()
    for doc in nlp.pipe(documents, batch_size=64, n_process=1):
        tokens = [t for t in doc if not t.is_space]
        n = len(tokens)
        for i in range(n):
            for length in range(n_min, n_max + 1):
                j = i + length
                if j > n:
                    break
                span = tokens[i:j]
                last = span[-1]
                if last.pos_ not in ("NOUN", "PROPN"):
                    continue
                if any(t.is_stop or t.is_punct or not t.is_alpha for t in span):
                    continue
                if drop_suffixes and _ends_in_abstract_suffix(last.text):
                    continue
                vocab.add(" ".join(t.text.lower() for t in span))
    return vocab


# Uglanova & Gius 2020 ("The Order of Things", CHR 2020) found that
# restricting a literary corpus's vocabulary to NN/ADJA/ADJD/VVFIN (nouns,
# attributive adjectives, predicative/adverbial adjectives, finite full
# verbs) - i.e. content words with grammatical number/person/tense stripped
# of non-finite verb forms, adverbs, and proper nouns - gave their cleanest,
# most linearly coherence-vs-topic-count models on German literary prose.
# German spaCy is trained directly on the STTS tagset, so token.tag_ IS
# these labels verbatim - no approximation needed. English spaCy uses Penn
# Treebank tags, which have no NN/ADJA/ADJD/VVFIN equivalents; the closest
# analogous restriction (content nouns/adjectives + finite lexical verbs
# only, excluding infinitives/gerunds/participles/modals/auxiliaries) is
# hand-mapped below. French spaCy (fr_core_news_sm) has neither fine-grained
# tag_ values (tag_ == pos_ always for this model, unlike German/English)
# nor a usable coarse VERB/AUX split fine enough on its own for finiteness -
# that detail lives in token.morph's VerbForm feature instead (Fin vs.
# Part/Inf), so the French branch below checks morph rather than tag_.
# Motivation: a KATM run on literary prose can produce topics that are
# tight, low-overlap clusters of verb forms sharing a grammatical/register
# class (all speech-act verbs, all reproach/annoyance words, all perception
# verbs, ...) rather than a shared subject matter - a cohesion/register
# signal, not a thematic one. Restricting which finite verbs this filter
# can draw from is a direct, targeted response to that failure mode - the
# one relevant to literary/narrative prose analysis in particular.
_PAPER_RESTRICTED_TAGS_DE = {"NN", "ADJA", "ADJD", "VVFIN"}
_PAPER_RESTRICTED_NOUN_ADJ_TAGS_EN = {"NN", "NNS", "JJ", "JJR", "JJS"}
_PAPER_RESTRICTED_FINITE_VERB_TAGS_EN = {"VBZ", "VBP", "VBD"}
_PAPER_RESTRICTED_AUX_LEMMAS_EN = {"be", "have", "do"}


def _passes_paper_restricted_filter(token, language: str) -> bool:
    """True iff `token` is one of NN/ADJA/ADJD/VVFIN (German, exact STTS
    match) or its hand-mapped English/French equivalent (see module comment
    above _PAPER_RESTRICTED_TAGS_DE)."""
    if language == "german":
        return token.tag_ in _PAPER_RESTRICTED_TAGS_DE
    if language == "french":
        # fr_core_news_sm's tag_ carries no finiteness distinction (it just
        # mirrors pos_) - VERB already excludes AUX in this model's own
        # tagset (unlike English, where finite auxiliaries share VBZ/VBP/VBD
        # with lexical verbs), so the only extra check needed is morph's
        # VerbForm=Fin, which excludes participles/infinitives.
        if token.pos_ in ("NOUN", "ADJ"):
            return True
        if token.pos_ == "VERB":
            return token.morph.get("VerbForm") == ["Fin"]
        return False
    if token.tag_ in _PAPER_RESTRICTED_NOUN_ADJ_TAGS_EN:
        return True
    if token.tag_ in _PAPER_RESTRICTED_FINITE_VERB_TAGS_EN:
        # VBZ/VBP/VBD also catches finite auxiliaries/modals ("is", "has",
        # "did") which the paper's VVFIN (a *full*-verb tag in STTS,
        # excluding auxiliaries by definition) would not - excluded here by
        # lemma and by spaCy's own AUX tag so the restriction stays a
        # faithful match rather than a looser approximation.
        return token.pos_ != "AUX" and token.lemma_.lower() not in _PAPER_RESTRICTED_AUX_LEMMAS_EN
    return False


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
                          method: str = "tfidf", language: str = "english",
                          pos_filter: "str | None" = None, max_df: float = 1.0) -> List[str]:
    """Extract content words from a corpus.

    Args:
        texts:  List of raw text strings.
        min_df: Minimum document frequency threshold.
        method: ``"tfidf"`` (default) or ``"spacy"``.

                ``"tfidf"`` — fits a CountVectorizer and returns alphabetic
                unigrams that pass stopword and sanity filters.  ~300× faster
                than spaCy on large corpora; ~84% vocabulary overlap.
                `pos_filter` has no effect with this method (no POS tags
                are available) and is silently ignored.

                ``"spacy"`` — POS-tags every document with a spaCy pipeline
                (selected by `language`) and keeps content words per
                `pos_filter` (lemmatised for grouping, but the most frequent
                genuine surface form is returned — see
                _extract_content_words_spacy). More linguistically precise
                but very slow on corpora > ~5,000 docs.
        language: ``"english"`` (default), ``"german"``, or ``"french"``.
                Selects the spaCy model (``"spacy"`` method only).
        pos_filter: ``None`` (default) — keep the broad, longstanding set
                {NOUN, VERB, ADJ, ADV, PROPN} (spaCy universal POS), same as
                always. ``"paper_restricted"`` — keep only NN/ADJA/ADJD/
                VVFIN (German, exact) or its English/French equivalent
                (nouns, adjectives, and *finite lexical* verbs only — see
                _passes_paper_restricted_filter) per Uglanova & Gius 2020,
                useful for literary/narrative prose where broader POS sets
                can produce grammatical-register clusters instead of
                subject-matter ones. Only affects ``method="spacy"``.
        max_df: Maximum document-frequency *ratio* (0-1) — words appearing
                in more than this fraction of documents are dropped, same
                sense as sklearn's CountVectorizer max_df. 1.0 (default) is
                a no-op. Only affects ``method="spacy"``; ``"tfidf"`` already
                hardcodes its own max_df=0.98 internally.

    Returns:
        Deduplicated sorted list of content words.
    """
    if method == "spacy":
        return _extract_content_words_spacy(texts, min_df, language, pos_filter, max_df)
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


def _extract_content_words_spacy(texts: List[str], min_df: int, language: str = "english",
                                  pos_filter: "str | None" = None, max_df: float = 1.0) -> List[str]:
    """Lemma-groups document-frequency (so e.g. inflectional variants of one
    word all pool their counts under one lemma) — but the lemma string
    itself is never returned as a vocabulary entry, since spaCy's
    lemmatizer can occasionally produce a form that isn't a real word at
    all for morphologically rich languages. Instead, each lemma group's
    most frequent genuine surface form (the actual inflected token as it
    appears in the text) is returned. Lemma is used here purely to
    group/count, exactly as lemma_dedup uses it purely to compare in
    topic_assignment.project_vocabulary — neither ever emits the lemma
    itself as output.
    """
    import spacy
    from collections import defaultdict
    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    model_name = _SPACY_MODEL.get(language, language)  # allow passing a raw model name too
    nlp = spacy.load(model_name, disable=["parser", "ner", "senter"])
    stop = _stopwords_for(language)

    lemma_doc_count: Counter = Counter()
    lemma_surface_count: dict = defaultdict(Counter)
    for doc in nlp.pipe(texts, batch_size=64, n_process=1):
        doc_lemmas: set = set()
        for token in doc:
            keep = (_passes_paper_restricted_filter(token, language) if pos_filter == "paper_restricted"
                    else token.pos_ in _CONTENT_POS)
            if keep:
                lemma = token.lemma_.lower()
                surface = token.text.lower()
                if (surface.isalpha() and len(surface) >= 2
                        and lemma not in stop and surface not in stop
                        and (language != "german" or (_is_german_alphabet(lemma)
                                                        and _is_german_alphabet(surface)))):
                    doc_lemmas.add(lemma)
                    lemma_surface_count[lemma][surface] += 1
        lemma_doc_count.update(doc_lemmas)

    n_docs = len(texts)
    max_df_count = max_df * n_docs
    result = {
        lemma_surface_count[lemma].most_common(1)[0][0]
        for lemma, cnt in lemma_doc_count.items()
        if min_df <= cnt <= max_df_count
    }
    return sorted(result)