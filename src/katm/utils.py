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

# d-Prose-specific: confirmed on a 500-doc KATM corpus-level run to appear in
# 8 of 12 topics as pure narrative filler, not thematic signal.
#
# The English/French function words below (the, and, of, ...) are here
# because spaCy's de_core_news_sm tags unrecognized tokens PROPN by default
# (verified directly: nlp("Er sagte the und ging weiter durch the Straße.")
# tags "the" PROPN), and PROPN is one of _CONTENT_POS in
# _extract_content_words_spacy. The character-alphabet filter
# (_is_german_alphabet) does NOT catch these — t/h/e are all valid German
# letters, so a script/character-set filter is structurally the wrong tool
# for foreign *words* made of in-alphabet letters. This list is a stopgap
# against exactly the token that was confirmed leaking into topic word lists
# ("the" in Topic 2 of a corpus-level run); it is not (and cannot be, as a
# fixed list) a complete language-ID solution for stray foreign-language
# passages elsewhere in the corpus.
_EXTRA_STOP_DE = {
    "wissenschaftlich",
    # English function words
    "the", "and", "of", "is", "was", "are", "were", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "or", "but", "not", "be", "this",
    "that", "it", "he", "she", "they", "you", "we",
    # French function words (also plausible in this corpus per the same
    # foreign-passage mechanism)
    "le", "la", "les", "de", "et", "un", "une", "des", "est", "que", "qui",
}

# Standard German alphabet only (a-z + äöüß). Confirmed necessary on d-Prose:
# "åbo" (å is Swedish/Danish/Norwegian, not German — from a placename
# appearing in just 2 of 2511 documents, "Das Bombardement von Åbo") leaked
# into 8-13 of 12-15 topics regardless of how high the embedding-model
# similarity thresholds were pushed (tried 0.92/0.95/0.97) — raising a
# similarity threshold can't fix a token that shouldn't be in the candidate
# pool at all. A character-set filter is the correct fix: cheaper, more
# targeted, and doesn't risk excluding legitimate words that happen to score
# high against a centroid. Applied only for language="german" — English
# extraction already accepts any Unicode letter via the token_pattern fix,
# and loanwords (café, naïve) are common enough in English prose that the
# same strict filter would be wrong there.
_GERMAN_ALPHABET_RE = re.compile(r"^[a-zäöüß]+$")


def _is_german_alphabet(word: str) -> bool:
    """True if every character in `word` (lowercased) is a standard German
    letter. Multi-word phrases are checked word-by-word."""
    return all(_GERMAN_ALPHABET_RE.match(w) for w in word.lower().split())


def lemma_key_map(words: List[str], language: str = "english") -> dict:
    """Map each word to a within-topic dedup key, batch-lemmatized via spaCy
    with no sentence context (only isolated words are available at this
    stage - content_words is already a flat, deduplicated vocabulary, not
    tied back to the sentences it came from).

    This is deliberately a *dedup key*, not a replacement for the returned
    topic word: callers keep displaying `words` as-is and only use this map
    to decide whether two candidates are "the same word" for the purposes of
    excluding both from one topic's word list.

    Empirically (checked directly against this project's actual morphology
    problem): isolated-token spaCy lemmatization recovers most inflectional
    variants correctly (sang/singen/gesungen -> singen; verdunkelt/verdunkeln
    -> verdunkeln) even without sentence context, though not all — strong-verb
    ablaut pairs like geschmolzen/schmelzen can still land on different keys.
    It's a real improvement over no dedup, not a perfect lemmatizer; note
    that verbatim-context lemmatization already ran once during content-word
    extraction and evidently still let some of these variants through as
    distinct vocabulary entries, so a from-scratch pass here (not reusing
    that earlier lemma) is the only way to actually catch them.
    """
    import spacy
    model_name = _SPACY_MODEL.get(language, language)
    nlp = spacy.load(model_name, disable=["parser", "ner", "senter"])
    keys = {}
    for w, doc in zip(words, nlp.pipe(words, batch_size=256)):
        keys[w] = doc[0].lemma_.lower() if len(doc) else w.lower()
    return keys


def _stopwords_for(language: str) -> set:
    """Stopword set for the given language. English uses NLTK's list
    (232 words, adequate). German additionally merges in spaCy's German
    list (543 words) - NLTK's German list alone was found, empirically on
    d-Prose, to miss nearly every common narrative discourse word (schon,
    immer, sagte, konnte, wohl, kam, ...), leaving LDA/KATM topics dominated
    by them."""
    if language == "english":
        stop = set(_stopwords)
    else:
        try:
            stop = set(nltk.corpus.stopwords.words(language))
        except OSError:
            return set(_stopwords)
    if language == "german":
        try:
            import spacy
            stop |= spacy.load("de_core_news_sm").Defaults.stop_words
        except OSError:
            pass
        stop |= _EXTRA_STOP_DE
    if language == "french":
        # Same rationale as German above: NLTK's French list (157 words) is
        # missing most common discourse words spaCy's own list catches
        # (notamment, egalement, voici, tel, ...) - merge both rather than
        # rely on NLTK alone, verified directly (157 vs. spaCy's 507).
        try:
            import spacy
            stop |= spacy.load("fr_core_news_sm").Defaults.stop_words
        except OSError:
            pass
    return stop


_SPACY_MODEL = {"english": "en_core_web_sm", "german": "de_core_news_sm", "french": "fr_core_news_sm"}


# Experimental (see build_noun_final_vocab's exclude_abstract_suffixes):
# German abstract-nominalization suffixes. -ung/-heit/-keit/-nis/-schaft/
# -tum are the classic, highly productive deverbal/deadjectival
# abstract-noun formants (Verwunderung, Dankbarkeit, Einsamkeit, Verständnis,
# Freundschaft, Reichtum) - grammatically NOUN, so they pass the noun-final
# filter above untouched, but semantically closer to an adjective/verb than
# a concrete referent, which this project's own qualitative classification
# has repeatedly scored as the "abstract emotion/cognition soup" BAD
# pattern rather than a real subject-matter topic. Known false positives
# exist (Zeitung "newspaper", Mannschaft "team/crew", Bistum "diocese" are
# concrete nouns that happen to carry these suffixes) - this is a blunt,
# suffix-only heuristic, not a semantic classifier, kept opt-in and
# exploratory for exactly that reason.
_GERMAN_ABSTRACT_SUFFIXES = ("ung", "heit", "keit", "nis", "schaft", "tum")


def _ends_in_abstract_suffix(word: str) -> bool:
    w = word.lower()
    return any(w.endswith(suf) and len(w) > len(suf) + 2 for suf in _GERMAN_ABSTRACT_SUFFIXES)


_NOUN_FINAL_NLP: dict = {}


def build_noun_final_vocab(documents, language: str, ngram_range: tuple,
                            exclude_abstract_suffixes: bool = False) -> set:
    """Shared candidate-restriction vocabulary: n-grams that are right-headed
    noun phrases (last token NOUN/PROPN), stopword/punct/non-alpha tokens
    excluded from any span. This is the exact mechanism keyphrase_extractor
    .KeyphraseExtractor._noun_final_candidate_vectorizer uses for
    keyphrase_scope="per_document" (see that method's docstring for the
    full diagnosis/motivation - grammatical/register-class topics like "all
    speech verbs" instead of subject-matter ones). Factored out here so
    keyphrase_scope="global" (global_keyphrases.py) can apply the identical
    restriction to its own corpus-wide candidate vectorizer, tested opt-in
    via KATM's existing keybert_candidate_pos_filter parameter (see
    topic_model.py's fit(), "global" branch) - one shared implementation,
    not two driftable copies, matching this module's existing convention
    (see global_keyphrases.py's own docstring on the same point).

    exclude_abstract_suffixes: False (default, unchanged behavior). If
    True, additionally drops any span whose head (last token) ends in a
    German abstract-nominalization suffix (see _GERMAN_ABSTRACT_SUFFIXES) -
    experimental, testing whether German's especially productive abstract-
    noun morphology (see project notes: German keyphrase candidates skew
    more toward abstract emotion/cognition nouns than English's, even on
    matched "weak" novels) is itself a lever on top of the plain noun-final
    restriction. No-op for language != "german" regardless of this flag
    (the suffix list is German-specific).

    Returns a set of lowercased, space-joined n-gram strings. Empty set if
    spaCy finds nothing (caller should fall back to an unrestricted
    vectorizer, not silently produce zero candidates).
    """
    import spacy

    model_name = _SPACY_MODEL.get(language, language)
    nlp = _NOUN_FINAL_NLP.get(model_name)
    if nlp is None:   # cached: streaming calls this once per batch of documents
        nlp = _NOUN_FINAL_NLP[model_name] = spacy.load(model_name, disable=["ner", "parser"])
    # see _extract_content_words_spacy's identical fix: parser/ner already
    # disabled, so raising the default 1_000_000-char cap is safe and
    # necessary for full-length documents (e.g. full novels).
    longest = max((len(d) for d in documents), default=0)
    if longest >= nlp.max_length:
        nlp.max_length = longest + 1
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


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


def strip_email_addresses(text: str) -> str:
    """Remove email addresses from raw text before they reach any
    vectorizer/keyphrase extractor.

    Found via the 20 Newsgroups sci.med investigation
    (gemini/sdtm/newsgroups_gsc_autos_junk_diagnostic.py): sklearn's
    fetch_20newsgroups(remove=("headers", "footers", "quotes")) strips real
    email headers and quoted-reply lines, but NOT inline signature blocks or
    "who's who"/contact-list-style content embedded in a post's body (e.g.
    "Ryan, Sean    FSSPR@acad3.alaska.edu"). Content-word extraction's
    token_pattern splits an un-stripped email address into fragments on
    non-letter characters - "fsspr@acad3.alaska.edu" becomes candidate
    tokens "fsspr", "acad", "alaska", "edu" - and while "edu"/"com"/"org"
    are already in _EXTRA_STOP, a personal handle like "fsspr" can't be
    caught by any fixed stopword list. It survived a min_df=3 doc-frequency
    filter (appeared in 3 of 4000 sampled documents purely because it's one
    person's own address, reused across their own posts) and was then
    projected into an otherwise-unrelated GMM cluster as an opaque,
    meaningless topic word. Stripping the whole address BEFORE tokenization
    (rather than trying to blacklist fragments after the fact) removes the
    candidate at its source, so no downstream stopword list has to guess
    every person's email handle in advance.

    Matches the whole address (local-part@domain, requiring at least one
    dot in the domain, e.g. user@host.edu or user@sub.host.co.uk) and
    removes it entirely rather than replacing it with a placeholder token -
    an email address carries no topical content once identified as one, so
    there is nothing worth keeping in its place.
    """
    return _EMAIL_RE.sub(" ", text)


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
# tag_ values (verified directly: tag_ == pos_ always for this model, unlike
# German/English) nor a usable coarse VERB/AUX split fine enough on its own
# for finiteness - that detail lives in token.morph's VerbForm feature
# instead (Fin vs. Part/Inf, also verified directly), so the French branch
# below checks morph rather than tag_. Motivation for adding this as its own
# filter (as
# opposed to the broader, longstanding _CONTENT_POS default): KATM runs on
# 1870-1920 literary German (gius/orderofthings.pdf replication, see project
# memory) produced several topics that were tight, low-overlap clusters of
# verb forms sharing a grammatical/register class (all speech-act verbs,
# all reproach/annoyance words, all perception verbs, ...) rather than a
# shared subject matter - a cohesion/register signal, not a thematic one,
# in exactly the sense the paper itself distinguishes. Restricting the
# vocabulary finite verbs CAN be drawn from (this filter) is a direct,
# targeted response to that failure mode.
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


def extract_content_words(texts: List[str], min_df: int = 3,
                          method: str = "tfidf", language: str = "english",
                          pos_filter: str = None, max_df: float = 1.0) -> List[str]:
    """Extract content words from a corpus.

    Args:
        texts:  List of raw text strings.
        min_df: Minimum document frequency threshold.
        method: ``"tfidf"`` (default), ``"spacy"`` or ``"sampled_pos"`` (
                only: POS-tags a few sampled occurrence windows per word instead of the
                whole corpus, ~14x faster than "spacy" on ELTeC-100 with ~97% precision /
                89% recall against its vocabulary; honours pos_filter - see
                fast_content_words.py).

                ``"tfidf"`` — fits a CountVectorizer and returns alphabetic
                unigrams that pass stopword and sanity filters.  ~300× faster
                than spaCy on large corpora; ~84% vocabulary overlap.
                `pos_filter` has no effect with this method (no POS tags
                are available) and is silently ignored.

                ``"spacy"`` — POS-tags every document with a spaCy pipeline
                (en_core_web_sm / de_core_news_sm, selected by `language`) and
                keeps content words per `pos_filter` (lemmatised).  More
                linguistically precise but very slow on corpora > ~5,000 docs.
        language: ``"english"`` (default) or ``"german"``. Selects the
                stopword list (both methods) and the spaCy model (``"spacy"``
                method only).
        pos_filter: ``None`` (default) — keep the broad, longstanding set
                {NOUN, VERB, ADJ, ADV, PROPN} (spaCy universal POS), same as
                always. ``"paper_restricted"`` — keep only NN/ADJA/ADJD/
                VVFIN (German, exact) or its English equivalent (nouns,
                adjectives, and *finite lexical* verbs only - see
                _passes_paper_restricted_filter) per Uglanova & Gius 2020.
                Only affects ``method="spacy"``.
        max_df: Maximum document-frequency *ratio* (0-1) - words appearing
                in more than this fraction of documents are dropped, same
                sense as sklearn's CountVectorizer max_df. 1.0 (default) is
                a no-op, preserving prior behavior byte-for-byte. Uglanova
                & Gius's own high-frequency cleaning step used 0.5 (drop
                words in >50% of documents) - the counterpart to min_df's
                low-frequency cleaning. Only affects ``method="spacy"``;
                ``"tfidf"`` already hardcodes its own max_df=0.98 internally
                and ignores this argument.

    Returns:
        Deduplicated sorted list of content words.
    """
    if method == "spacy":
        return _extract_content_words_spacy(texts, min_df, language, pos_filter, max_df)
    if method == "sampled_pos":   #  see fast_content_words.py
        from .fast_content_words import extract_content_words_sampled
        return extract_content_words_sampled(texts, min_df=min_df, language=language,
                                             pos_filter=pos_filter, max_df=max_df)
    if method in ("type_pos", "type_pos_templates"):   #  see fast_content_words.py
        from .fast_content_words import extract_content_words_typed
        return extract_content_words_typed(texts, min_df=min_df, language=language, pos_filter=pos_filter,
                                           max_df=max_df,
                                           context="templates" if method == "type_pos_templates" else "none")
    if method != "tfidf":
        raise ValueError(f"content_words_method must be 'tfidf', 'spacy', 'sampled_pos', 'type_pos' "
                         f"or 'type_pos_templates' - got {method!r}")
    return _extract_content_words_tfidf(texts, min_df, language)


def _extract_content_words_tfidf(texts: List[str], min_df: int, language: str = "english") -> List[str]:
    import re as _re
    from sklearn.feature_extraction.text import CountVectorizer

    stop = _stopwords_for(language) | (_EXTRA_STOP if language == "english" else set())
    _bad_token = _re.compile(r"(.)\1{2,}|.{26,}")
    n = len(texts)
    vec = CountVectorizer(
        ngram_range=(1, 1),
        stop_words=list(stop),
        min_df=min(min_df, n),
        max_df=0.98,
        # Unicode-aware: [^\W\d_] matches any letter (incl. ä/ö/ü/ß etc.),
        # not just a-z — the old [a-zA-Z] pattern silently mangled non-ASCII
        # scripts (e.g. German umlauts split mid-word).
        token_pattern=r"(?u)\b[^\W\d_]{4,}\b",
    )
    vec.fit(texts)
    vocab = vec.get_feature_names_out()
    return sorted(
        w for w in vocab
        if w.lower() not in stop and not _bad_token.search(w)
        and (language != "german" or _is_german_alphabet(w))
    )


def extract_content_words_streaming(corpus, min_df: int = 3, max_df: float = 0.98,
                                    language: str = "english"):
    """Streaming counterpart of extract_content_words(method="tfidf") - same
    output semantics (document-frequency-filtered alphabetic unigram
    vocabulary, same stopword list and malformed-token guard), but built
    from ONE streaming pass over the corpus instead of an in-memory
    CountVectorizer.fit() - for corpora too large to hold as a Python list
    (SPGC, PG-19; see flake/stream.py, whose
    anchor-selection passes this mirrors).

    What this does NOT need to solve, measured directly: the final
    vocabulary's own size. flake_stream.py's pass 1a already streams a
    2,488-book PG-19 sample once and finds 664,903 raw word types, with
    only 74,052 surviving df >= 20 - a content-word vocabulary at full
    PG-19 scale would plausibly land in the low hundreds of thousands,
    the same order of magnitude as FLaKE's own candidate vocabulary
    (already embedded fine at that scale: 157,490 phrases in 50s on this
    sample). The actual blocker is that a plain CountVectorizer.fit()
    needs every document's raw text in memory at once to compute document
    frequencies - this function replaces that with a streaming Counter
    over per-document word SETS (so a word occurring 50 times in one
    document still only counts once toward its document frequency, the
    same semantics sklearn's own min_df/max_df use), exactly as
    flake_stream.py's pass 1a already does for FLaKE's own candidate
    vocabulary - no new mechanism, just the same one applied to content
    words instead of anchor candidates.

    Args:
        corpus: same Corpus type streaming.pairs_factory's
            functions take: a callable returning a fresh (id, text)
            iterator, or iter_text_files/iter_parquet_texts from that
            module.
        min_df: minimum document count (same meaning as
            extract_content_words's own min_df).
        max_df: maximum document-frequency RATIO (same meaning/default as
            _extract_content_words_tfidf's hardcoded 0.98).
        language: "english" only for now - the in-memory version's German
            alphabet filter isn't ported here yet.

    Returns:
        (vocab, info) - vocab: sorted list of content words. info:
        {"n_docs": documents streamed, "n_raw_types": vocabulary size
        before df filtering, "timings": {"count": seconds}}.
    """
    import time as _time

    if language != "english":
        raise NotImplementedError("extract_content_words_streaming currently supports English only")

    from .streaming import pairs_factory as _as_factory

    t0 = _time.perf_counter()
    stop = _stopwords_for(language) | _EXTRA_STOP
    token_re = re.compile(r"\b[^\W\d_]{4,}\b", re.UNICODE)
    bad_token = re.compile(r"(.)\1{2,}|.{26,}")

    doc_freq: Counter = Counter()
    factory = _as_factory(corpus)
    n_docs = 0
    for _, text in factory():
        n_docs += 1
        doc_freq.update({w.lower() for w in token_re.findall(text)})
    t_count = _time.perf_counter() - t0

    n_raw = len(doc_freq)
    min_df_eff = min(min_df, n_docs)
    max_df_count = max_df * n_docs
    vocab = sorted(
        w for w, c in doc_freq.items()
        if min_df_eff <= c <= max_df_count and w not in stop and not bad_token.search(w)
    )
    return vocab, {"n_docs": n_docs, "n_raw_types": n_raw, "timings": {"count": t_count}}


def extract_content_words_spacy_streaming(
    corpus, min_df: int = 3, max_df: float = 1.0, language: str = "english",
    pos_filter: str = None, batch_size: int = 64, return_info: bool = False,
    progress_every: int = 0,
):
    """Streaming counterpart of extract_content_words(method="spacy") - tags
    EVERY token of the corpus (not a sample, not isolated word types - this is
    the full, slow, ground-truth method every other content-word extractor in
    this project was built to approximate cheaply). Built here mainly as an
    EXPERIMENTAL baseline: to measure, on a corpus too large to comfortably
    hold in memory, what it actually costs to do this the slow way, and what
    (if anything) is lost by using the fast approximations instead.

    Structurally simpler to stream than sampled_pos/type_pos: those need a
    first pass to build a frequency-filtered CANDIDATE vocabulary before any
    tagging happens (so only a bounded, vocabulary-sized amount of tagging
    work is ever done). Full spaCy tags unconditionally - every token of
    every document, with no pre-filter - so there is no candidate step to
    build first. The only thing that has to accumulate across the whole
    corpus is PER-LEMMA statistics (document count, surface-form counts),
    which is bounded by vocabulary size regardless of corpus size - so this
    needs only ONE streaming pass: feed documents to spaCy's own nlp.pipe()
    (which already processes its input lazily, one batch at a time - it was
    never given the texts as an in-memory list in the first place, just an
    iterator from the same corpus factory every other function in this
    module takes) and accumulate lemma statistics as results come back.

    Args: same meaning as extract_content_words(method="spacy"), except
        `corpus` replaces `texts` - the same Corpus type
        streaming.pairs_factory accepts (a callable
        returning a fresh (id, text) iterator, or iter_text_files /
        iter_parquet_texts). German is not yet supported here (the in-memory
        version's extra alphabet filter isn't ported).

    Returns: (vocab, info) if return_info, else vocab - same convention as
        the other streaming content-word functions. info: {"n_docs":
        documents streamed, "n_lemmas_seen": vocabulary size before df
        filtering, "timings": {"tag": seconds}}.
    """
    import time as _time
    import spacy
    from collections import defaultdict
    from .streaming import pairs_factory as _as_factory

    if language != "english":
        raise NotImplementedError("extract_content_words_spacy_streaming currently supports English only")

    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    model_name = _SPACY_MODEL.get(language, language)
    nlp = spacy.load(model_name, disable=["parser", "ner", "senter"])
    nlp.max_length = 20_000_000  # parser/ner disabled, so long documents are safe to tag
    stop = _stopwords_for(language)

    factory = _as_factory(corpus)
    texts = (text for _, text in factory())

    t0 = _time.perf_counter()
    lemma_doc_count: Counter = Counter()
    lemma_surface_count: dict = defaultdict(Counter)
    n_docs = 0
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=1):
        n_docs += 1
        if progress_every and n_docs % progress_every == 0:
            elapsed = _time.perf_counter() - t0
            print(f"  [spacy_streaming] {n_docs} docs tagged, {len(lemma_doc_count)} lemmas so far, "
                 f"{elapsed:.0f}s elapsed ({n_docs/elapsed:.1f} docs/s)", flush=True)
        doc_lemmas: set = set()
        for token in doc:
            keep = (_passes_paper_restricted_filter(token, language) if pos_filter == "paper_restricted"
                   else token.pos_ in _CONTENT_POS)
            if keep:
                lemma = token.lemma_.lower()
                surface = token.text.lower()
                if (surface.isalpha() and len(surface) >= 2
                        and lemma not in stop and surface not in stop):
                    doc_lemmas.add(lemma)
                    lemma_surface_count[lemma][surface] += 1
        lemma_doc_count.update(doc_lemmas)
    t_tag = _time.perf_counter() - t0

    max_df_count = max_df * n_docs
    vocab = sorted(
        lemma_surface_count[lemma].most_common(1)[0][0]
        for lemma, cnt in lemma_doc_count.items()
        if min_df <= cnt <= max_df_count
    )
    info = {"n_docs": n_docs, "n_lemmas_seen": len(lemma_doc_count), "timings": {"tag": t_tag}}
    return (vocab, info) if return_info else vocab


def _extract_content_words_spacy(texts: List[str], min_df: int, language: str = "english",
                                  pos_filter: str = None, max_df: float = 1.0) -> List[str]:
    """Lemma groups document-frequency (so e.g. "sang"/"singen"/"gesungen"
    all pool their counts under one lemma, same as before) - but the lemma
    string itself is never returned as a vocabulary entry. spaCy's German
    lemmatizer is imperfect enough to occasionally produce a form that isn't
    a real word at all (e.g. "zugebrachen", "mußtesen" - neither is valid
    German; both are the lemmatizer over-regularizing a strong/prefixed verb
    it couldn't resolve correctly), and those malformed strings would
    otherwise end up as literal "topic words". Instead, each lemma group's
    most frequent genuine surface form (the actual inflected token as it
    appears in the text) is returned. Any surviving variant is equally
    acceptable - lemma is used here purely to group/count, exactly as
    lemma_dedup uses it purely to compare in topic_assignment.project_vocabulary;
    neither ever emits the lemma itself as output.
    """
    import spacy
    from collections import defaultdict
    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    model_name = _SPACY_MODEL.get(language, language)  # allow passing a raw model name too
    nlp = spacy.load(model_name, disable=["parser", "ner", "senter"])
    # spaCy's default nlp.max_length=1_000_000 chars exists to guard parser/NER
    # memory (~1GB/100k chars per its own docs) - both already disabled above,
    # so raising it here is safe and necessary for genuinely long documents
    # (e.g. full novels: a 1,084,263-character ELTeC novel crashed this with
    # spaCy's E088 before this fix; tagger+lemmatizer-only has no comparable
    # per-char memory cost).
    longest = max((len(t) for t in texts), default=0)
    if longest >= nlp.max_length:
        nlp.max_length = longest + 1
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