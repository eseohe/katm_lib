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


class KeyphraseExtractor:
    """Extracts keyphrases from documents using KeyBERT, RAKE, YAKE, GSC, or TF-IDF."""

    def __init__(self, algorithm: str = "keybert", n_keyphrases: int = 10, pretrained_model=None,
                 yake_use_position: bool = False, tfidf_ngram_range: tuple = (1, 2),
                 keybert_ngram_range: tuple = (1, 2)):
        """Initialize KeyphraseExtractor.

        Args:
            algorithm: "keybert", "rake", "yake", "gsc", or "tfidf".
            n_keyphrases: Number of top keyphrases to extract per document.
            pretrained_model: Optional pre-loaded SentenceTransformer to pass to KeyBERT
                and GSC, avoiding a second model load when the caller already has one.
            tfidf_ngram_range: n-gram range for TF-IDF mode (default (1, 2)).
            keybert_ngram_range: n-gram range for KeyBERT candidate generation (default
                (1, 2)). (1, 3) captures longer phrases but encodes ~2× more candidates
                and roughly doubles extraction time with minimal anchor quality gain.

        Raises:
            ValueError: If algorithm is not recognized.
        """
        valid_algorithms = {"keybert", "rake", "yake", "gsc", "tfidf"}
        if algorithm not in valid_algorithms:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Must be one of {valid_algorithms}")

        self.algorithm = algorithm
        self.n_keyphrases = n_keyphrases
        self._pretrained_model = pretrained_model
        self._yake_use_position = yake_use_position
        self._tfidf_ngram_range = tfidf_ngram_range
        self._keybert_ngram_range = keybert_ngram_range
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
        elif self.algorithm == "gsc":
            return self._extract_gsc(documents)
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

        # Pass the full corpus in one call so KeyBERT encodes all candidates
        # in a single batched model.encode() rather than one call per document.
        # stop_words='english' reduces the candidate set before encoding.
        all_keywords = self._keybert_model.extract_keywords(
            documents,
            keyphrase_ngram_range=self._keybert_ngram_range,
            top_n=self.n_keyphrases,
            stop_words="english",
        )

        return [[kp for kp, _ in doc_kws] for doc_kws in all_keywords]

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

    def _extract_gsc(self, documents: List[str]) -> List[List[str]]:
        """Greedy Semantic Coverage (GSC) keyphrase extractor.

        For each document:
        1. Generate candidate n-gram phrases (1–3 words, content-word filtered).
        2. Split the document into sentences.
        3. Embed candidates and sentences with the shared sentence-transformer.
        4. Greedy selection: at each step pick the phrase that maximally increases
           total coverage, defined as the sum of per-sentence maximum cosine
           similarities to any already-selected phrase.  This forces selected
           phrases to cover different semantic regions of the document rather than
           all converging on the same dominant topic.

        Why this works better than RAKE/YAKE for semantically similar classes:
        GSC uses embedding-based selection so it picks semantically discriminative
        phrases (e.g. "manic episode", "suicidal ideation", "panic attack") rather
        than statistically frequent ones.  Unlike KeyBERT which ranks by similarity
        to the whole-document embedding, GSC distributes coverage across sentences,
        capturing sub-topics that KeyBERT's single-vector target misses.
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        if self._pretrained_model is None:
            from sentence_transformers import SentenceTransformer
            self._pretrained_model = SentenceTransformer("all-MiniLM-L6-v2")

        # ── sentence splitter ────────────────────────────────────────────────
        _sent_end = re.compile(r'(?<=[.!?])\s+')

        def _split_sentences(text: str) -> List[str]:
            sents = [s.strip() for s in _sent_end.split(text) if len(s.strip()) > 15]
            return sents if sents else [text]

        # ── candidate generator ──────────────────────────────────────────────
        _word_re   = re.compile(r"[a-z]{2,}")
        _stop_re   = re.compile(
            r"\b(?:" + "|".join(re.escape(w) for w in sorted(_EXTENDED_STOP, key=len, reverse=True)) + r")\b"
        )

        def _candidates(text: str) -> List[str]:
            """n-grams (1–3 words) from content-word runs between stopwords."""
            lowered = text.lower()
            parts   = _stop_re.split(lowered)
            cands: List[str] = []
            seen: set = set()
            for part in parts:
                words = _word_re.findall(part)
                for n in range(1, 4):
                    for i in range(len(words) - n + 1):
                        phrase = " ".join(words[i:i + n])
                        if phrase not in seen and _is_clean_phrase(phrase):
                            seen.add(phrase)
                            cands.append(phrase)
            return cands

        # ── per-document GSC ─────────────────────────────────────────────────
        results = []
        for doc in documents:
            if not doc.strip():
                results.append([])
                continue

            cands = _candidates(doc)
            if not cands:
                results.append([])
                continue

            sents = _split_sentences(doc)

            # Encode sentences + candidates in one call (no extra model loads)
            all_texts  = sents + cands
            all_embs   = self._pretrained_model.encode(
                all_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            sent_embs  = all_embs[:len(sents)]    # (n_sents, D)
            cand_embs  = all_embs[len(sents):]    # (n_cands, D)

            # sim_matrix[i, j] = cosine sim between candidate i and sentence j
            sim_mat = _cos_sim(cand_embs, sent_embs)  # (n_cands, n_sents)

            selected: List[int] = []
            # covered[j] = max cosine sim of sentence j to any selected phrase
            covered = np.zeros(len(sents))
            remaining = list(range(len(cands)))

            for _ in range(self.n_keyphrases):
                if not remaining:
                    break
                best_idx, best_gain = -1, -1.0
                rem_sims = sim_mat[remaining]          # (n_remaining, n_sents)
                # Vectorised marginal gain for all remaining candidates at once
                new_covered = np.maximum(covered, rem_sims)   # (n_remaining, n_sents)
                gains = new_covered.sum(axis=1) - covered.sum()
                best_local = int(np.argmax(gains))
                best_gain  = float(gains[best_local])
                best_idx   = remaining[best_local]

                if best_gain <= 0:
                    break

                selected.append(best_idx)
                covered = np.maximum(covered, sim_mat[best_idx])
                remaining.pop(best_local)

            results.append([cands[i] for i in selected])

        return results