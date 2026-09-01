"""Keyphrase extraction module for KATM."""

import re
from collections import Counter
from typing import List

import nltk

# Ensure stopwords and punkt are available
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

# Common structural/boilerplate tokens that YAKE and RAKE lift from headers,
# footers, and thread metadata. These carry no topical signal.
_STRUCTURAL = {
    "subject", "lines", "organization", "organisation", "writes", "article",
    "reply", "wrote", "said", "date", "from", "re", "cc", "newsgroups",
    "path", "distribution", "keywords", "summary", "references", "sender",
    "nntp", "posting", "host", "xref", "approved", "followup",
}
_EXTENDED_STOP = _stopwords | _STRUCTURAL


def _rake_extract(text: str, max_phrase_len: int = 3) -> List[str]:
    """Minimal RAKE implementation (no external dependency).

    Splits text on stopwords/punctuation, scores candidate phrases by
    word degree / frequency, returns phrases sorted by score descending.
    """
    # Sentence splitting on punctuation + stopwords
    sentence_delimiters = re.compile(
        r"[\s\t,\.!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~\n]+"
    )
    phrase_enders = re.compile(r"[.!?,;:\n]")

    stop_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in sorted(_EXTENDED_STOP, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    # Split into candidate phrases by stopwords and punctuation
    sentences = phrase_enders.split(text.lower())
    candidates = []
    for sent in sentences:
        parts = stop_pattern.split(sent)
        for part in parts:
            words = [w for w in sentence_delimiters.split(part.strip()) if w.isalpha() and len(w) >= 2]
            if 1 <= len(words) <= max_phrase_len:
                candidates.append(words)

    # Count word frequency and degree (co-occurrence within phrases)
    freq: Counter = Counter()
    degree: Counter = Counter()
    for phrase_words in candidates:
        for w in phrase_words:
            freq[w] += 1
            degree[w] += len(phrase_words) - 1

    # Score = (freq + degree) / freq for each word; phrase score = sum of word scores
    scored: List[tuple] = []
    seen_phrases: set = set()
    for phrase_words in candidates:
        phrase_str = " ".join(phrase_words)
        if phrase_str in seen_phrases:
            continue
        seen_phrases.add(phrase_str)
        score = sum((freq[w] + degree[w]) / freq[w] for w in phrase_words)
        scored.append((phrase_str, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored]


def _is_clean_phrase(phrase: str) -> bool:
    """Return True if phrase has at least one real alpha word that is not a stopword."""
    words = phrase.lower().split()
    return any(
        w.isalpha() and len(w) >= 2 and w not in _EXTENDED_STOP
        for w in words
    )


def _all_stop(phrase: str) -> bool:
    """Return True if every token in phrase is a stopword/structural word or a digit."""
    words = phrase.lower().split()
    return all(w in _EXTENDED_STOP or w.isdigit() for w in words)


def _chunk_text_for_embedding(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Splits text into sentence-packed chunks that each fit within
    max_tokens, measured with the embedding model's own tokenizer (not a
    word-count approximation). Greedily fills each chunk with whole
    sentences until the next sentence would exceed the budget, then starts
    a new chunk — so sentence boundaries are never broken except in the
    rare case of a single sentence alone exceeding max_tokens, which is
    hard-split by raw token windows as a fallback.

    A document that already fits within max_tokens returns a single chunk
    (the whole text unchanged) — this is what makes
    long_document_strategy="chunk" a strict fix rather than a behavior
    change for documents already under the limit. Also used by
    global_keyphrases.extract_global_keyphrases (keyphrase_scope="global")
    for the same reason.
    """
    n_tokens = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    if n_tokens <= max_tokens:
        return [text]

    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False, truncation=False))
        if sent_len > max_tokens:
            # A single sentence alone exceeds the budget - flush whatever's
            # pending, then hard-split this one sentence by raw token count.
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            ids = tokenizer.encode(sent, add_special_tokens=False, truncation=False)
            for i in range(0, len(ids), max_tokens):
                chunks.append(tokenizer.decode(ids[i:i + max_tokens]))
            continue
        # +2 for the [CLS]/[SEP] special tokens the real embedding call adds
        if current and current_len + sent_len + 2 > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _max_relevance_mmr(
    relevance,
    word_embeddings,
    words: List[str],
    top_n: int,
    diversity: float,
):
    """Same greedy algorithm as keybert's own MMR, adapted to take a
    precomputed per-candidate relevance score (here: max cosine similarity
    across a document's own chunks) instead of similarity to a single
    doc_embedding — KeyBERT's own use_mmr path only supports the latter, so
    this is a light adaptation for long_document_strategy="chunk"."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [int(np.argmax(relevance))]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        candidate_relevance = relevance[candidates_idx]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        mmr_scores = (1 - diversity) * candidate_relevance - diversity * target_similarities
        best_local = int(np.argmax(mmr_scores))
        best_idx = candidates_idx[best_local]
        keywords_idx.append(best_idx)
        candidates_idx.remove(best_idx)

    keywords_idx.sort(key=lambda i: -relevance[i])
    return [words[i] for i in keywords_idx]


class KeyphraseExtractor:
    """Extracts keyphrases from documents using KeyBERT, RAKE, YAKE, or TF-IDF."""

    def __init__(self, algorithm: str = "keybert", n_keyphrases: int = 10, pretrained_model=None,
                 yake_use_position: bool = False, tfidf_ngram_range: tuple = (1, 2),
                 keybert_ngram_range: tuple = (1, 2), keybert_use_mmr: bool = False,
                 keybert_diversity: float = 0.5, keybert_stop_words="english",
                 long_document_strategy: str = "truncate"):
        """Initialize KeyphraseExtractor.

        Args:
            algorithm: "keybert", "rake", "yake", or "tfidf".
            n_keyphrases: Number of top keyphrases to extract per document.
            pretrained_model: Optional pre-loaded SentenceTransformer to pass to KeyBERT,
                avoiding a second model load when the caller already has one.
            tfidf_ngram_range: n-gram range for TF-IDF mode (default (1, 2)).
            keybert_ngram_range: n-gram range for KeyBERT candidate generation (default
                (1, 2)). (1, 3) captures longer phrases but encodes ~2× more candidates
                and roughly doubles extraction time with minimal anchor quality gain.
            keybert_use_mmr: If True and algorithm="keybert", diversifies KeyBERT's
                top-n_keyphrases candidates per document with Maximal Marginal
                Relevance instead of plain top-N-by-relevance. Default False
                (KeyBERT's own default) — without it, a document's top-N tends to
                be dominated by redundant n-gram variants of whichever word is
                most central to it (e.g. "wickham", "mr wickham", "wickham
                cluster" all in one document's top 10), which is a candidate-
                diversity problem, not a keyphrase-count problem.
            keybert_diversity: MMR lambda passed to KeyBERT when
                keybert_use_mmr=True (0 = pure relevance, 1 = pure diversity).
                Default 0.5, KeyBERT's own default.
            keybert_stop_words: Stop-word list passed to KeyBERT's own
                extract_keywords call. Either the string "english" (KeyBERT's
                built-in list, default) or an explicit list of stopwords for
                another language — KeyBERT/sklearn's CountVectorizer accepts
                either. KATM passes its own utils._stopwords_for(language)
                result here when language != "english".
            long_document_strategy: "truncate" (default, unchanged behavior)
                or "chunk". Only affects algorithm="keybert". KeyBERT ranks
                candidate phrases by cosine similarity to a single embedding
                of the whole document, but SentenceTransformer.encode()
                silently truncates that embedding at the model's
                max_seq_length (256 tokens for the common all-MiniLM-L6-v2
                default) — so on a document longer than that, everything past
                the limit is invisible to the ranking step. "chunk" fixes this
                by splitting any document exceeding max_seq_length into
                sentence-packed chunks that each fit within it, embedding
                every chunk, and scoring each candidate phrase by its MAXIMUM
                cosine similarity across the document's own chunks instead of
                one (possibly truncated) whole-document embedding. A no-op
                (identical output to "truncate") for any document that
                already fits within max_seq_length, so switching this on is a
                strict fix, never a regression, for short documents.

        Raises:
            ValueError: If algorithm is not recognized.
        """
        valid_algorithms = {"keybert", "rake", "yake", "tfidf"}
        if algorithm not in valid_algorithms:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Must be one of {valid_algorithms}")
        if long_document_strategy not in {"truncate", "chunk"}:
            raise ValueError(
                f"long_document_strategy must be 'truncate' or 'chunk' - got {long_document_strategy!r}"
            )

        self.algorithm = algorithm
        self.n_keyphrases = n_keyphrases
        self._pretrained_model = pretrained_model
        self._yake_use_position = yake_use_position
        self._tfidf_ngram_range = tfidf_ngram_range
        self._keybert_ngram_range = keybert_ngram_range
        self._keybert_use_mmr = keybert_use_mmr
        self._keybert_diversity = keybert_diversity
        self._keybert_stop_words = keybert_stop_words
        self._long_document_strategy = long_document_strategy
        self._keybert_model = None
        self._yake_extractor = None

    def extract(self, documents: List[str]) -> List[List[str]]:
        """Extract keyphrases from documents.

        Args:
            documents: List of document strings.

        Returns:
            List of keyphrase lists, one per document.
        """
        if not documents:
            return []

        if self.algorithm == "keybert":
            return self._extract_keybert(documents)
        elif self.algorithm == "rake":
            return self._extract_rake(documents)
        elif self.algorithm == "yake":
            return self._extract_yake(documents)
        elif self.algorithm == "tfidf":
            return self._extract_tfidf(documents)

        return []

    def _extract_keybert(self, documents: List[str]) -> List[List[str]]:
        """Extract keyphrases using KeyBERT."""
        try:
            from keybert import KeyBERT
        except ImportError:
            raise ImportError("KeyBERT is not installed. Install with: pip install keybert")

        if self._keybert_model is None:
            self._keybert_model = KeyBERT(model=self._pretrained_model)

        if self._long_document_strategy == "chunk":
            return self._extract_keybert_chunked(documents)

        # Pass the full corpus in one call so KeyBERT encodes all candidates
        # in a single batched model.encode() rather than one call per document.
        extract_kwargs = dict(
            keyphrase_ngram_range=self._keybert_ngram_range,
            top_n=self.n_keyphrases,
            stop_words=self._keybert_stop_words,
        )
        if self._keybert_use_mmr:
            extract_kwargs["use_mmr"] = True
            extract_kwargs["diversity"] = self._keybert_diversity
        all_keywords = self._keybert_model.extract_keywords(documents, **extract_kwargs)
        # KeyBERT silently unwraps its own return value for a batch of exactly
        # one document ([[(kw, score), ...]] -> [(kw, score), ...]) - re-wrap
        # so downstream code can always assume one list per document
        # regardless of how many documents were passed in. Not a behavior
        # change for KATM's normal usage (always called with the full
        # document batch), but a real bug for a single-document call.
        if len(documents) == 1 and all_keywords and isinstance(all_keywords[0], tuple):
            all_keywords = [all_keywords]

        return [[kp for kp, _ in doc_kws] for doc_kws in all_keywords]

    def _extract_keybert_chunked(self, documents: List[str]) -> List[List[str]]:
        """long_document_strategy="chunk" — see __init__'s docstring for the
        full motivation. For each document: split into token-budget-limited,
        sentence-packed chunks (a no-op single chunk when the document already
        fits), embed every chunk, and score each candidate phrase by its
        MAXIMUM cosine similarity across the document's own chunks rather than
        similarity to one (possibly truncated) whole-document embedding."""
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # The actual SentenceTransformer backing this KeyBERT instance -
        # retrieved from KeyBERT itself (not self._pretrained_model directly)
        # so this also works when self._pretrained_model was None and
        # KeyBERT lazily loaded its own default model.
        st_model = self._keybert_model.model.embedding_model
        max_tokens = getattr(st_model, "max_seq_length", 256) or 256

        results: List[List[str]] = []
        for doc in documents:
            if not doc.strip():
                results.append([])
                continue

            chunks = _chunk_text_for_embedding(doc, st_model.tokenizer, max_tokens)

            try:
                cv = CountVectorizer(
                    ngram_range=self._keybert_ngram_range,
                    stop_words=self._keybert_stop_words,
                    min_df=1,
                ).fit([doc])
            except ValueError:
                results.append([])  # no candidates survive (e.g. all-stopword document)
                continue
            candidates = list(cv.get_feature_names_out())
            if not candidates:
                results.append([])
                continue

            chunk_embeddings = st_model.encode(chunks, batch_size=32, show_progress_bar=False)
            candidate_embeddings = st_model.encode(candidates, batch_size=64, show_progress_bar=False)

            sim_to_chunks = cosine_similarity(candidate_embeddings, chunk_embeddings)  # (n_candidates, n_chunks)
            relevance = sim_to_chunks.max(axis=1)

            if self._keybert_use_mmr:
                keywords = _max_relevance_mmr(
                    relevance, candidate_embeddings, candidates,
                    self.n_keyphrases, self._keybert_diversity,
                )
            else:
                top_idx = np.argsort(relevance)[::-1][: self.n_keyphrases]
                keywords = [candidates[i] for i in top_idx]

            results.append(keywords)

        return results

    def _extract_rake(self, documents: List[str]) -> List[List[str]]:
        """Extract keyphrases using RAKE.

        Tries rake-nltk first; falls back to a built-in RAKE implementation
        when rake-nltk is not installable (e.g. Python ≥ 3.10).

        Changes vs vanilla RAKE:
        - max_length=3: prevents multi-sentence garbage phrases on noisy text
        - Post-filter: drops phrases with no real alphabetic non-stopword word
        """
        try:
            from rake_nltk import Rake as _RakeNltk

            def _rake_doc(doc):
                r = _RakeNltk(max_length=3)
                r.extract_keywords_from_text(doc)
                return r.get_ranked_phrases()

        except ImportError:
            # Built-in fallback: minimal RAKE (word co-occurrence scoring)
            def _rake_doc(doc):
                return _rake_extract(doc, max_phrase_len=3)

        results = []
        for doc in documents:
            if not doc.strip():
                results.append([])
                continue
            raw = _rake_doc(doc)
            phrases = [p for p in raw if _is_clean_phrase(p) and not _all_stop(p)][: self.n_keyphrases]
            results.append(phrases)

        return results

    def _extract_yake(self, documents: List[str]) -> List[List[str]]:
        """Extract keyphrases using YAKE.

        Changes vs vanilla YAKE:
        - Input lowercased: removes capitalization bias that lifts header tokens
        - stopwords=_EXTENDED_STOP: blocks structural tokens (subject, lines, …)
          at the candidate level rather than in a post-filter
        - dedup_lim=0.6: tighter deduplication for more diverse keyphrases
        - features: by default excludes "wpos" so early-document position does
          not boost structural words; pass yake_use_position=True to restore it
        """
        try:
            from yake import KeywordExtractor as YakeExtractor
        except ImportError:
            raise ImportError("YAKE is not installed. Install with: pip install yake")

        if self._yake_extractor is None:
            import inspect as _inspect
            _yake_params = _inspect.signature(YakeExtractor.__init__).parameters
            _dedup_kwarg = "dedup_lim" if "dedup_lim" in _yake_params else "dedupLim"

            _ALL_FEATURES = ["wrel", "wfreq", "wspread", "wcase", "wpos"]
            features = _ALL_FEATURES if self._yake_use_position else [f for f in _ALL_FEATURES if f != "wpos"]
            self._yake_extractor = YakeExtractor(
                lan="en",
                n=2,
                **{_dedup_kwarg: 0.6},
                top=self.n_keyphrases * 3,   # over-fetch so filter has room
                features=features,
                stopwords=_EXTENDED_STOP,
            )

        # Contraction-removal pattern: "don't" → "dont", "it's" → "its", etc.
        # YAKE's tokenizer splits on apostrophes producing "n't", "'s" fragments
        # that score highly (short, frequent) and flood topic word lists.
        _contraction_re = re.compile(r"'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
        _apostrophe_re  = re.compile(r"'")

        results = []
        for doc in documents:
            if not doc.strip():
                results.append([])
                continue
            # Remove contraction suffixes, then remaining apostrophes, then lowercase.
            cleaned = _contraction_re.sub("", doc)
            cleaned = _apostrophe_re.sub("", cleaned).lower()
            keywords = self._yake_extractor.extract_keywords(cleaned)
            keyphrases = [kw[0] for kw in keywords if _is_clean_phrase(kw[0])][: self.n_keyphrases]
            results.append(keyphrases)

        return results

    def _extract_tfidf(self, documents: List[str]) -> List[List[str]]:
        """Extract keyphrases using per-document TF-IDF ranking.

        Fits a TF-IDF vectorizer on the full corpus so IDF values reflect
        corpus-wide term rarity, then for each document returns the top-N
        terms by their TF-IDF score in that document.

        IDF naturally down-weights generic words (high df → low IDF) and
        up-weights domain-specific terms that are rare corpus-wide but prominent
        in individual documents.  This makes it much more reliable than YAKE or
        RAKE on short documents where co-occurrence statistics are unreliable.
        """
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = len(documents)
        vec = TfidfVectorizer(
            ngram_range=self._tfidf_ngram_range,
            stop_words="english",
            min_df=min(2, n),   # exclude hapax; clamp to 1 for tiny corpora
            max_df=0.95 if n >= 20 else 1.0,
            sublinear_tf=True,  # log(1+tf): dampens raw counts on short docs
        )
        X = vec.fit_transform(documents)
        vocab = vec.get_feature_names_out()

        results = []
        for i in range(X.shape[0]):
            row = X[i]
            scores = row.toarray().ravel() if hasattr(row, "toarray") else row.A.ravel()
            # argsort descending over non-zero entries only
            nz = scores.nonzero()[0]
            if len(nz) == 0:
                results.append([])
                continue
            nz_sorted = nz[scores[nz].argsort()[::-1]]
            terms = []
            for idx in nz_sorted:
                phrase = vocab[idx]
                if _is_clean_phrase(phrase):
                    terms.append(phrase)
                if len(terms) >= self.n_keyphrases:
                    break
            results.append(terms)

        return results

