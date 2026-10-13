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
    a new chunk - so sentence boundaries are never broken except in the
    rare case of a single sentence alone exceeding max_tokens, which is
    hard-split by raw token windows as a fallback (garbage/no-punctuation
    text, not a realistic case for real prose or Reddit posts).

    A document that already fits within max_tokens returns a single chunk
    (the whole text unchanged) - this is what makes long_document_strategy
    ="chunk" a strict fix rather than a behavior change for the common
    case of documents already under the limit.
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


def _gsc_chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """Splits text into ~chunk_size-word chunks without breaking sentences
    (greedy sentence packing by raw word count) - ported from
    greedy_semantic_coverage/gsc_clean.py's chunk_text. Deliberately NOT
    the same as _chunk_text_for_embedding above: that one packs by the
    embedding model's own tokenizer against its max_seq_length (for
    KeyBERT's long_document_strategy="chunk" and keyphrase_scope="global",
    where a chunk must itself be encodable in one forward pass); this one
    uses gsc_clean.py's validated raw-word-count chunk_size (default 200)
    purely to build (a) the document's mean-pooled global vector and (b) a
    within-document mini-corpus for TF-IDF candidate ranking - both of
    which tolerate a chunk larger than max_seq_length just as well as one
    that fits (the embedding call still runs, SentenceTransformer just
    truncates that one chunk's own tail, not the whole document)."""
    words = text.split()
    if not words:
        return []
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        sent_words = sent.split()
        n = len(sent_words)
        if current and current_len + n > chunk_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.extend(sent_words)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


def _gsc_select_pool(scores: dict, candidate_pool) -> List[str]:
    """Reduce a {candidate: ranking_score} map to an ordered candidate
    list - ported from greedy_semantic_coverage/gsc_clean.py's
    _select_pool. This is the pre-embedding candidate-pool filter that
    keeps GSC fast on long documents: the default candidate_pool=0.9 keeps
    only the smallest score-ranked prefix whose cumulative score share
    reaches 90% of the total (typically a few hundred candidates even for
    a novel-length document with tens of thousands of raw n-grams) - the
    fix documented in greedy_semantic_coverage/paper.md §4.3 that brings
    GSC's runtime to the same order of magnitude as KeyBERT.

    candidate_pool: None (keep every candidate, ranked) | int (top-N) |
        float in (0, 1] (Zipf-style cumulative-coverage cutoff)."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return []
    if candidate_pool is None:
        return [w for w, _ in ranked]
    if isinstance(candidate_pool, int):
        return [w for w, _ in ranked[:candidate_pool]]
    total = sum(s for _, s in ranked)
    if total <= 0:
        return [w for w, _ in ranked]
    out: List[str] = []
    cum = 0.0
    for w, s in ranked:
        out.append(w)
        cum += s / total
        if cum >= candidate_pool:
            break
    return out


def _gsc_cluster_select(
    doc_candidates: List[str], cand_norms, d_norm, top_n: int,
) -> List[str]:
    """gsc_method="cluster": a non-greedy alternative to centroid/
    orthogonal's sequential round-by-round selection. Partitions a
    document's (already pool-filtered) candidates into top_n KMeans
    clusters over their embeddings, then takes the single most
    document-relevant candidate from each cluster - one vectorized
    clustering call instead of top_n sequential rounds, each of which
    (in the centroid/orthogonal path) requires its own numpy pass over
    every still-active candidate plus a Python-level substring dedup
    check against every already-selected phrase.

    Diversity here is structural rather than accumulated: two candidates
    can only both survive if KMeans placed them in different clusters at
    all, rather than centroid/orthogonal's approach of scoring against a
    running state that only reflects picks made so far. This also sidesteps
    orthogonal's selection-order sensitivity (an early pick permanently
    deflating the residual can starve a later, otherwise-good candidate
    that overlaps it) - cluster membership doesn't depend on selection
    order at all.

    Substring dedup (gsc_dedup) is NOT applied here - KMeans clustering
    already tends to place near-duplicate phrases (e.g. "wickham" and
    "mr wickham", which are cosine-close) in the same cluster, so at most
    one of them can be the cluster's own representative; a separate
    dedup pass was not found necessary in testing, but near-duplicates
    could still occur across clusters in principle.

    Trade-off vs the round-loop: KMeans has its own per-call overhead
    (init, iteration), run once per document - whether this beats
    centroid's per-round numpy passes depends on candidate-pool size and
    top_n; benchmark both on your own corpus rather than assuming.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n = len(doc_candidates)
    relevance = cand_norms @ d_norm

    if top_n >= n:
        order = np.argsort(-relevance)
        return [doc_candidates[i] for i in order]

    km = KMeans(n_clusters=top_n, n_init=1, random_state=42)
    cluster_labels = km.fit_predict(cand_norms)

    selected_idx = []
    for c in range(top_n):
        members = np.where(cluster_labels == c)[0]
        if members.size == 0:
            continue
        selected_idx.append(int(members[np.argmax(relevance[members])]))

    selected_idx.sort(key=lambda i: -relevance[i])
    return [doc_candidates[i] for i in selected_idx]


def _max_relevance_mmr(
    relevance,
    word_embeddings,
    words: List[str],
    top_n: int,
    diversity: float,
):
    """Same greedy algorithm as keybert._mmr.mmr, adapted to take a
    precomputed per-candidate relevance score (here: max cosine similarity
    across a document's own chunks) instead of computing similarity to a
    single doc_embedding internally - keybert's own mmr() only accepts the
    latter, so this is a light adaptation rather than a reuse."""
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
    """Extracts keyphrases from documents using KeyBERT, RAKE, YAKE, GSC, or TF-IDF."""

    def __init__(self, algorithm: str = "keybert", n_keyphrases: int = 10, pretrained_model=None,
                 yake_use_position: bool = False, tfidf_ngram_range: tuple = (1, 2),
                 keybert_ngram_range: tuple = (1, 2), keybert_use_mmr: bool = False,
                 keybert_diversity: float = 0.5, keybert_stop_words="english",
                 language: str = "english", long_document_strategy: str = "truncate",
                 gsc_candidate_ranking: str = "tfidf", gsc_candidate_pool=0.9,
                 gsc_chunk_size: int = 200, gsc_ngram_range: tuple = (1, 2),
                 gsc_dedup: bool = True, gsc_method: str = "centroid",
                 gsc_batch_size: int = 256, keybert_candidate_pos_filter: bool = False):
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
            keybert_use_mmr: If True, KeyBERT diversifies its top-N candidates with
                Maximal Marginal Relevance instead of plain top-N-by-relevance.
                Default False (KeyBERT's own default) — without it, KeyBERT's top-N
                for a given document tends to be dominated by redundant n-gram
                variants of whichever 1-2 words are most central to that document
                (e.g. "wickham", "mr wickham", "wickham cluster" all in one doc's
                top 10), which is a candidate-diversity problem, not a keyphrase-count
                problem — see the notebook's keyphrase-extraction diagnostic.
            keybert_diversity: MMR lambda passed to KeyBERT when keybert_use_mmr=True
                (0 = pure relevance, 1 = pure diversity). Only used if
                keybert_use_mmr=True. Default 0.5, KeyBERT's own default.
            keybert_stop_words: Stop-word list passed to KeyBERT's own
                extract_keywords call. Either the string "english" (KeyBERT's
                built-in list, default) or an explicit list of stopwords for
                another language (e.g. NLTK German stopwords) — KeyBERT/
                sklearn's CountVectorizer accepts either.
            language: "english" (default) or "german". When "german",
                extracted keyphrases containing any non-German-alphabet
                character (a-z, äöüß) are dropped after extraction — cheap
                and targeted fix for foreign-script tokens (e.g. a Swedish
                placename in one document) leaking into the anchor pool via
                a compressed-embedding-space false-similarity match, which
                raising similarity thresholds alone can't reliably fix since
                the token shouldn't be a candidate in the first place.
            long_document_strategy: "truncate" (default, unchanged behavior)
                or "chunk". Only affects algorithm="keybert". KeyBERT ranks
                candidate phrases by cosine similarity to a single embedding
                of the whole document - but SentenceTransformer.encode()
                silently truncates that embedding at the model's
                max_seq_length (256 tokens for the common all-MiniLM-L6-v2
                default), so on any document longer than that, only the
                first ~256 tokens ever influence which keyphrases get
                selected; the rest of the document is invisible to the
                ranking step. Confirmed directly on this project's own
                mental-health-Reddit notebooks (POSTS_PER_DOC=10 merges 10
                posts into one document): 100% of 300 sampled documents
                exceeded 256 tokens, median 1,686 tokens (6.6x the limit).
                "chunk" fixes this by splitting any document that exceeds
                the embedding model's max_seq_length into sentence-packed
                chunks that each fit within it, embedding every chunk, and
                scoring each candidate phrase by its MAXIMUM cosine
                similarity across the document's own chunks rather than
                similarity to one (possibly truncated) whole-document
                embedding - so a keyphrase from post 9 of 10 can still be
                selected even though it would never have been visible to
                the reference embedding under "truncate". Max (not top-k
                average, unlike the corpus-level keyphrase_scope="global"
                fix in global_keyphrases.py) is deliberate here: a merged
                multi-post document's chunks are typically small in number
                (single digits to a few dozen) and can be about genuinely
                different sub-topics, so a candidate should qualify if it
                strongly represents ANY ONE part of the document, not only
                if it's related to most of it - the opposite requirement
                from the corpus-level fix, which specifically needed to
                reward words related to only a SUBSET of many documents
                over words weakly related to all of them. When a document
                already fits within max_seq_length, chunking reduces to a
                single "chunk" (the whole document) and the max reduces to
                a single similarity value - "chunk" mode then produces
                identical rankings to "truncate" mode, so switching this on
                is a strict fix, never a regression, for short documents.
            gsc_candidate_ranking: "tfidf" (default) or "frequency". Only
                affects algorithm="gsc". How candidates are scored before
                the pre-embedding pool filter (gsc_candidate_pool) trims
                them: "tfidf" ranks each candidate by its max TF-IDF score
                across the document's own ~200-word chunks (used as a mini-
                corpus, so scores reflect within-document salience, not
                global corpus frequency); "frequency" ranks by raw count.
            gsc_candidate_pool: None (keep every candidate) | int (keep the
                top-N by gsc_candidate_ranking score) | float in (0, 1]
                (Zipf-style cumulative-score cutoff, default 0.9 — keep the
                smallest score-ranked prefix whose cumulative share reaches
                90%). This is the pre-embedding filter that keeps GSC fast
                on long documents: without it, every raw 1-2-gram in a
                document becomes an embedding call (tens of thousands for a
                full novel); with the default 0.9 cutoff it's typically a
                few hundred, the fix documented in
                greedy_semantic_coverage/paper.md §4.3 that brings GSC's
                runtime to the same order of magnitude as KeyBERT even on
                full-novel-length documents.
            gsc_chunk_size: Words per chunk (sentence-respecting greedy
                packing, not the embedding model's own tokenizer) used to
                build (a) the document's mean-pooled global vector and (b)
                the mini-corpus for gsc_candidate_ranking="tfidf". Default
                200, matching greedy_semantic_coverage/gsc_clean.py's
                validated default. A document that already fits in one
                chunk (~200 words) is unaffected; this only matters for
                documents exceeding that.
            gsc_ngram_range: n-gram range for GSC candidate generation
                (default (1, 2), independent of keybert_ngram_range).
            gsc_dedup: If True (default), substring-deduplicates GSC's
                greedy selection — a candidate is skipped if it's a
                substring of (or contains) an already-selected phrase, so
                e.g. "wickham" doesn't also get picked after "mr wickham".
            gsc_method: "centroid" (default), "orthogonal", or "cluster" -
                which objective drives selection. "cluster" is not from
                gsc_clean.py - it's a non-greedy alternative added here:
                KMeans-partitions a document's candidates into n_keyphrases
                clusters, then takes the most relevant candidate from each
                cluster, in one vectorized pass instead of n_keyphrases
                sequential rounds (see _gsc_cluster_select's own docstring).
                "centroid" (gsc_clean.py's own default, and the
                method paper.md's headline results are reported for) picks
                the candidate that most increases cosine similarity between
                the document vector and the mean of the selected set so
                far. "orthogonal" instead does Gram-Schmidt residual
                coverage: each pick's own direction is subtracted out of a
                running document residual, so every subsequent pick is
                scored against what's LEFT uncovered, not against the whole
                document again - a structurally stronger facet-diversity
                pressure than centroid's shared running average, at the
                cost of being selection-order-sensitive (an early pick
                permanently deflates the residual in its own direction,
                which can starve a later, otherwise-good candidate that
                overlaps it). gsc_clean.py's "volume" method exists too but
                is not exposed here yet.
            gsc_batch_size: batch_size passed to both of GSC's model.
                encode() calls (chunks, candidates). Default 256, raised
                from the original hardcoded 32/64 after profiling showed
                ~46% of a real 4000-document run's wall time going to
                per-minibatch device-transfer sync (.cpu() calls) on an
                MPS/GPU backend - many small batches each pay a fixed
                sync cost, so fewer/larger batches reduce total sync
                overhead. Measured effect is real but uneven: ~33% faster
                for short candidate phrases at this batch-size range, only
                ~5% faster for longer document chunks (chunk encoding is
                compute-bound by real transformer FLOPs per item, not by
                per-call sync overhead, so batching matters less there).
                Higher values trade GPU/MPS memory for fewer sync round
                trips - lower this if you hit an out-of-memory error on
                very large corpora or long chunks.

        Raises:
            ValueError: If algorithm is not recognized.
        """
        valid_algorithms = {"keybert", "rake", "yake", "gsc", "tfidf"}
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
        self._language = language
        self._long_document_strategy = long_document_strategy
        self._gsc_candidate_ranking = gsc_candidate_ranking
        self._gsc_candidate_pool = gsc_candidate_pool
        self._gsc_chunk_size = gsc_chunk_size
        self._gsc_ngram_range = gsc_ngram_range
        self._gsc_dedup = gsc_dedup
        self._gsc_method = gsc_method
        self._gsc_batch_size = gsc_batch_size
        self._keybert_candidate_pos_filter = keybert_candidate_pos_filter
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

    def _noun_final_candidate_vectorizer(self, documents: List[str]):
        """Build a CountVectorizer whose fixed vocabulary is restricted to
        n-grams that are right-headed noun phrases (last token NOUN/PROPN) -
        the English equivalent of the paper-restricted NN/ADJA/ADJD/VVFIN
        idea (see utils.py's content_pos_filter), applied here at the
        candidate-GENERATION stage instead of the final display-vocabulary
        stage. Diagnosed directly (see project memory, orderofthings
        anchor-pool investigation) as the actually-effective place to apply
        this: filtering the display vocabulary afterward left the anchor
        pool - and therefore the GMM cluster centroids themselves - full of
        pure-verb/adverb candidates ("tripped doubtless", "escape clouded"),
        which is what produced register/grammatical-class clusters (all
        speech verbs, all reproach words) instead of subject-matter ones.
        Excluding non-noun-final n-grams before they're ever scored should
        stop those clusters from forming in the first place, not just hide
        them from the final word list.

        Right-headedness (last token, not "any token") is deliberate: it
        rules out "tripped doubtless" and "escape clouded" (last token an
        adverb/participle) while keeping "mill wall", "aeroplane", "captain
        cuttle" (last token a concrete noun) - the same heuristic that makes
        "the tall man" a noun phrase and "very tall" not.
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from .utils import build_noun_final_vocab

        fixed = getattr(self, "_fixed_candidate_vocab", None)   # set by streaming: corpus-wide vocab
        if fixed is not None:
            # KeyBERT embeds the WHOLE vectorizer vocabulary on every call, so restrict the corpus-wide
            # vocabulary to the n-grams that occur in this batch. A document's candidates are exactly the
            # vocabulary n-grams present in it, so this changes nothing but the embedding cost.
            analyzer = CountVectorizer(ngram_range=self._keybert_ngram_range, token_pattern=r"(?u)\b\w+\b",
                                       lowercase=True).build_analyzer()
            present = set()
            for d in documents:
                present.update(analyzer(d))
            vocab = fixed & present if len(present) < len(fixed) else present & fixed
            if not vocab:
                vocab = {"\x00"}   # no candidate in this batch; do NOT fall back to the unrestricted vectorizer
        else:
            vocab = build_noun_final_vocab(documents, self._language, self._keybert_ngram_range)

        if not vocab:
            return None  # caller falls back to KeyBERT's own default vectorizer
        return CountVectorizer(vocabulary=sorted(vocab), ngram_range=self._keybert_ngram_range,
                                token_pattern=r"(?u)\b\w+\b")

    def _extract_keybert(self, documents: List[str]) -> List[List[str]]:
        """Extract keyphrases using KeyBERT."""
        try:
            from keybert import KeyBERT
        except ImportError:
            raise ImportError("KeyBERT is not installed. Install with: pip install keybert")

        if self._keybert_model is None:
            self._keybert_model = KeyBERT(model=self._pretrained_model)

        if self._long_document_strategy == "chunk":
            phrases = self._extract_keybert_chunked(documents)
        else:
            # Pass the full corpus in one call so KeyBERT encodes all candidates
            # in a single batched model.encode() rather than one call per document.
            # stop_words='english' reduces the candidate set before encoding.
            extract_kwargs = dict(
                keyphrase_ngram_range=self._keybert_ngram_range,
                top_n=self.n_keyphrases,
                stop_words=self._keybert_stop_words,
            )
            if self._keybert_candidate_pos_filter:
                custom_vectorizer = self._noun_final_candidate_vectorizer(documents)
                if custom_vectorizer is not None:
                    extract_kwargs["vectorizer"] = custom_vectorizer
                    extract_kwargs.pop("stop_words", None)  # already applied via is_stop above
            if self._keybert_use_mmr:
                extract_kwargs["use_mmr"] = True
                extract_kwargs["diversity"] = self._keybert_diversity
            all_keywords = self._keybert_model.extract_keywords(documents, **extract_kwargs)
            # KeyBERT silently unwraps its own return value for a batch of
            # exactly one document ([[(kw, score), ...]] -> [(kw, score),
            # ...], see keybert._model.KeyBERT.extract_keywords: "if
            # len(all_keywords) == 1: ... all_keywords = all_keywords[0]") -
            # re-wrap so downstream code can always assume one list per
            # document regardless of how many documents were passed in.
            # Not a change in behavior for KATM's normal usage (always
            # called with the full document batch), but a real bug for
            # anyone calling extract() with a single-document list, as
            # found while testing long_document_strategy="chunk" below.
            if len(documents) == 1 and all_keywords and isinstance(all_keywords[0], tuple):
                all_keywords = [all_keywords]
            phrases = [[kp for kp, _ in doc_kws] for doc_kws in all_keywords]

        if self._language == "german":
            from .utils import _is_german_alphabet
            phrases = [[p for p in doc_phrases if _is_german_alphabet(p)] for doc_phrases in phrases]
        return phrases

    def _extract_keybert_chunked(self, documents: List[str]) -> List[List[str]]:
        """long_document_strategy="chunk" - see __init__'s docstring for the
        full motivation. For each document: split into token-budget-limited,
        sentence-packed chunks (a no-op single chunk when the document
        already fits), embed every chunk, and score each candidate phrase by
        its MAXIMUM cosine similarity across the document's own chunks
        rather than similarity to one (possibly truncated) whole-document
        embedding."""
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # The actual SentenceTransformer backing this KeyBERT instance -
        # retrieved from KeyBERT itself (not self._pretrained_model
        # directly) so this also works when self._pretrained_model was None
        # and KeyBERT lazily loaded its own default model.
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

    def _extract_gsc(self, documents: List[str]) -> List[List[str]]:
        """Greedy Semantic Coverage (GSC) keyphrase extractor.

        Ported from the validated, optimized implementation in
        greedy_semantic_coverage/gsc_clean.py (see that module's paper.md
        for the full method and cross-domain benchmarks: literary novels,
        clinical Reddit, political speeches). This REPLACES the older,
        unoptimized GSC that used to live here - unbounded per-document
        n-gram candidates (every raw 1-3-gram embedded, no pre-filter) and
        a full candidate x sentence coverage matrix recomputed from scratch
        every greedy round (see greedy_semantic_coverage/improvements.md's
        S1-S8 for the exact costs that version paid). Two changes:

        1. Document representation: ONE mean-pooled vector over sentence-
           packed ~200-word chunks (gsc_chunk_size), not a per-sentence
           coverage matrix - avoids the O(candidates x sentences) cost the
           old version paid on every greedy round.
        2. Candidate pre-filtering: candidates are ranked by within-
           document TF-IDF (gsc_candidate_ranking, using the document's own
           chunks as a mini-corpus) and trimmed to the smallest cumulative-
           score prefix reaching gsc_candidate_pool share (default 0.9)
           BEFORE any embedding call - the fix documented in paper.md §4.3
           that keeps GSC's runtime in the same order of magnitude as
           KeyBERT even on full novels, instead of embedding every raw
           n-gram in the document.

        Also batches BOTH embedding calls (chunks, candidates) across the
        WHOLE document corpus in two calls total, rather than looping
        doc-by-doc as gsc_clean.py itself does - the same "one batched
        model.encode() instead of N small ones" optimization already used
        for algorithm="keybert" (non-chunked) and keyphrase_scope="global"
        (see global_keyphrases.py).

        Greedy selection method is gsc_method ("centroid", default, or
        "orthogonal" - see gsc_method's own docstring above for the
        mechanism of each), with substring dedup between rounds (gsc_dedup)
        so e.g. "wickham" isn't also picked after "mr wickham" already was.
        Either way this distributes coverage across a document's semantic
        facets rather than converging on redundant variants of one dominant
        topic, the way ranking candidates independently (KeyBERT, TF-IDF)
        tends to. gsc_clean.py's "volume" method exists too but is not
        exposed here yet.
        """
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        if self._pretrained_model is None:
            from sentence_transformers import SentenceTransformer
            self._pretrained_model = SentenceTransformer("all-MiniLM-L6-v2")
        model = self._pretrained_model
        embed_dim = model.get_sentence_embedding_dimension()

        stop_list = sorted(_EXTENDED_STOP)
        chunk_size = self._gsc_chunk_size

        # Opt-in test of the same candidate-generation POS restriction
        # keybert_candidate_pos_filter applies to algorithm="keybert" (see
        # _noun_final_candidate_vectorizer's docstring) and
        # keyphrase_scope="global" applies via global_keyphrases.py's
        # candidate_pos_filter - one shared flag, same meaning, now ported
        # to GSC too: computed ONCE over the whole corpus (one batched
        # spaCy pass via utils.build_noun_final_vocab, not one per
        # document) and used below to filter each document's own raw
        # CountVectorizer candidates down to right-headed noun phrases
        # before TF-IDF ranking ever sees them.
        noun_final_vocab = None
        if self._keybert_candidate_pos_filter:
            from .utils import build_noun_final_vocab
            noun_final_vocab = build_noun_final_vocab(documents, self._language, self._gsc_ngram_range)

        # ── Step 1: per-document candidate generation + pre-filtering
        # (cheap sklearn only, no embedding calls yet) ──────────────────
        per_doc_chunks: List[List[str]] = []
        per_doc_candidates: List[List[str]] = []
        for doc in documents:
            if not doc.strip():
                per_doc_chunks.append([])
                per_doc_candidates.append([])
                continue

            chunks = _gsc_chunk_text(doc, chunk_size) or [doc]
            per_doc_chunks.append(chunks)

            try:
                cvec = CountVectorizer(ngram_range=self._gsc_ngram_range, stop_words=stop_list)
                Xc = cvec.fit_transform([doc])
            except ValueError:
                per_doc_candidates.append([])
                continue
            counts = dict(zip(cvec.get_feature_names_out(), Xc.toarray()[0].tolist()))
            if noun_final_vocab is not None:
                counts = {w: c for w, c in counts.items() if w in noun_final_vocab}
            if not counts:
                per_doc_candidates.append([])
                continue

            if self._gsc_candidate_ranking == "tfidf":
                corpus = chunks if len(chunks) >= 2 else (chunks or [""])
                try:
                    tvec = TfidfVectorizer(
                        vocabulary=list(counts), ngram_range=self._gsc_ngram_range,
                        stop_words=stop_list, sublinear_tf=True,
                    )
                    Xt = tvec.fit_transform(corpus)
                    tfidf_scores = np.asarray(Xt.max(axis=0).todense()).ravel()
                    scores = dict(zip(tvec.get_feature_names_out(), tfidf_scores.tolist()))
                except ValueError:
                    scores = counts
            else:
                scores = counts

            per_doc_candidates.append(_gsc_select_pool(scores, self._gsc_candidate_pool))

        # ── Step 2: batch-encode every document's chunks in ONE call, and
        # every document's (already pre-filtered) candidates in another ONE
        # call - not N small per-document calls. ────────────────────────
        chunk_counts = [len(c) for c in per_doc_chunks]
        all_chunks = [c for chunks in per_doc_chunks for c in chunks]
        cand_counts = [len(c) for c in per_doc_candidates]
        all_candidates = [c for cands in per_doc_candidates for c in cands]

        chunk_embeddings = (
            model.encode(all_chunks, batch_size=self._gsc_batch_size, show_progress_bar=False)
            if all_chunks else np.zeros((0, embed_dim))
        )
        candidate_embeddings = (
            model.encode(all_candidates, batch_size=self._gsc_batch_size, show_progress_bar=False)
            if all_candidates else np.zeros((0, embed_dim))
        )

        chunk_starts = np.cumsum([0] + chunk_counts[:-1]) if chunk_counts else np.array([], dtype=int)
        cand_starts = np.cumsum([0] + cand_counts[:-1]) if cand_counts else np.array([], dtype=int)

        # ── Step 3: per-document greedy centroid selection - pure numpy,
        # no more model calls (the expensive step is already done). ────
        results: List[List[str]] = []
        for i, doc_candidates in enumerate(per_doc_candidates):
            if not doc_candidates or chunk_counts[i] == 0:
                results.append([])
                continue

            c0, c1 = int(cand_starts[i]), int(cand_starts[i]) + cand_counts[i]
            h0, h1 = int(chunk_starts[i]), int(chunk_starts[i]) + chunk_counts[i]
            cand_emb = candidate_embeddings[c0:c1]
            doc_global = chunk_embeddings[h0:h1].mean(axis=0)

            cand_norms = cand_emb / (np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9)
            d_norm = doc_global / (np.linalg.norm(doc_global) + 1e-9)

            n = len(doc_candidates)
            top_n = min(self.n_keyphrases, n)

            if self._gsc_method == "cluster":
                results.append(_gsc_cluster_select(doc_candidates, cand_norms, d_norm, top_n))
                continue

            selected_mask = np.zeros(n, dtype=bool)
            selected_strs: List[str] = []
            centroid_sum = np.zeros_like(d_norm)
            doc_residual = d_norm.copy()

            for _ in range(top_n):
                active = np.where(~selected_mask)[0]
                if self._gsc_dedup and selected_strs:
                    active = np.array(
                        [j for j in active if not any(
                            doc_candidates[j] in s or s in doc_candidates[j] for s in selected_strs
                        )],
                        dtype=int,
                    )
                if active.size == 0:
                    break

                if self._gsc_method == "orthogonal":
                    # Gram-Schmidt residual coverage: score against what's
                    # LEFT uncovered by already-selected picks, not the
                    # whole document again.
                    scores_arr = cand_norms[active] @ doc_residual
                elif self._gsc_method == "centroid":
                    trial = centroid_sum + cand_norms[active]
                    trial_norms = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-9)
                    scores_arr = trial_norms @ d_norm
                else:
                    raise ValueError(
                        f"gsc_method must be 'centroid', 'orthogonal', or 'cluster' - got {self._gsc_method!r}"
                    )
                best_local = int(np.argmax(scores_arr))
                best = int(active[best_local])

                selected_mask[best] = True
                selected_strs.append(doc_candidates[best])
                if self._gsc_method == "orthogonal":
                    doc_residual = doc_residual - np.dot(doc_residual, cand_norms[best]) * cand_norms[best]
                    if np.linalg.norm(doc_residual) < 1e-6:
                        break
                else:
                    centroid_sum = centroid_sum + cand_norms[best]

            results.append(selected_strs)

        if self._language == "german":
            from .utils import _is_german_alphabet
            results = [[p for p in doc_phrases if _is_german_alphabet(p)] for doc_phrases in results]
        return results
