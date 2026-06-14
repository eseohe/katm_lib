"""Main topic model module for KATM."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import GMMTopicClusterer
from .embedding import SentenceEmbedder
from .keyphrase_extractor import KeyphraseExtractor
from .topic_assignment import WordTopicProjector
from .utils import extract_content_words


class KATM:
    """Keyphrase Anchored Topic Modeling (KATM) main class."""

    def __init__(
        self,
        kp_algorithm: str = "keybert",
        n_keyphrases: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_topics: int = 10,
        soft_threshold: float = 0.15,
        top_n_words: int = 20,
        min_df: int = 3,
        yake_use_position: bool = False,
        tfidf_ngram_range: tuple = (1, 2),
        keybert_ngram_range: tuple = (1, 2),
        content_words_method: str = "tfidf",
        normalize_embeddings: bool = False,
        anchor_dedup_threshold: float = 0.85,
        min_anchor_df: int = 2,
        max_anchor_df_ratio: float = 0.4,
        mmr_diversity: float = 0.2,
        mmr_max_sim: float = 0.85,
    ):
        """Initialize KATM model.

        Args:
            kp_algorithm: Keyphrase extraction algorithm ("keybert", "rake", or "yake").
            n_keyphrases: Number of keyphrases to extract per document.
            embedding_model: Sentence-transformer model name.
            n_topics: Number of topics to discover.
            soft_threshold: Probability threshold for word-topic assignments.
            top_n_words: Number of top words to store per topic.
            min_df: Minimum document frequency for vocabulary.
            yake_use_position: If True, YAKE includes the wpos feature (words
                appearing earlier in a document score higher). Defaults to False
                because position systematically lifts header/boilerplate tokens.
            anchor_dedup_threshold: Cosine similarity threshold for greedy
                semantic deduplication of the anchor pool. Set to None to disable.
            min_anchor_df: Minimum number of documents a keyphrase must appear
                in to be included in the anchor pool. Removes hapax phrases that
                are too specific to one document. Default 2.
            max_anchor_df_ratio: Maximum fraction of documents a keyphrase may
                appear in. Phrases above this threshold are class-agnostic noise
                (e.g. "feel like" in 73% of MH posts). Default 0.4.
            mmr_diversity: MMR lambda for word-list construction. Controls the
                relevance/diversity trade-off: 0.0 = pure relevance (top-k by
                cosine sim), 1.0 = pure diversity. Default 0.2 (80% relevance).
            mmr_max_sim: Hard dedup cap for word-list MMR. Any candidate with
                cosine similarity above this threshold to any already-selected
                word is excluded before scoring. Prevents near-morphological
                duplicates (e.g. anxious/anxiousness, sim=0.93) regardless of
                lambda. Default 0.85.
            tfidf_ngram_range: n-gram range for TF-IDF keyphrase mode.
                Default (1, 2) extracts unigrams and bigrams.
            keybert_ngram_range: n-gram range for KeyBERT candidate generation.
                Default (1, 2). Use (1, 3) if you need longer anchor phrases but
                expect ~2× longer extraction time.
            content_words_method: ``"tfidf"`` (default) or ``"spacy"``. Controls
                how the vocabulary for topic word projection is built. TF-IDF is
                ~300× faster; spaCy is more linguistically precise (POS-filtered,
                lemmatised) but very slow on corpora > ~5,000 docs.
            normalize_embeddings: If True, L2-normalise all embeddings before
                GMM fitting and downstream assignment. This projects keyphrases,
                words, and documents onto the unit sphere so the GMM covariance
                matrices capture directional (angular) variance rather than
                magnitude variance. Makes ``GMM.predict_proba`` valid for
                out-of-distribution inputs (documents vs keyphrases). Default
                False to preserve backward compatibility.
        """
        self.kp_algorithm = kp_algorithm
        self.n_keyphrases = n_keyphrases
        self.embedding_model = embedding_model
        self.n_topics = n_topics
        self.soft_threshold = soft_threshold
        self.top_n_words = top_n_words
        self.min_df = min_df
        self.yake_use_position = yake_use_position
        self.tfidf_ngram_range = tfidf_ngram_range
        self.keybert_ngram_range = keybert_ngram_range
        self.content_words_method = content_words_method
        self.normalize_embeddings = normalize_embeddings
        self.anchor_dedup_threshold = anchor_dedup_threshold
        self.min_anchor_df = min_anchor_df
        self.max_anchor_df_ratio = max_anchor_df_ratio
        self.mmr_diversity = mmr_diversity
        self.mmr_max_sim = mmr_max_sim

        # State set by fit()
        self.topics_: Optional[Dict[int, List[Tuple[str, float]]]] = None
        self.doc_topic_probs_: Optional[List[np.ndarray]] = None
        self._n_fitted_docs = 0
        self._embedder: Optional[SentenceEmbedder] = None
        self._clusterer: Optional[GMMTopicClusterer] = None

    def fit(self, documents: List[str]) -> "KATM":
        """Run full KATM pipeline.

        Args:
            documents: List of document strings. If your raw data is posts,
                group them into documents before calling fit().

        Returns:
            self for chaining.
        """
        if not documents:
            self.topics_ = {}
            self.doc_topic_probs_ = []
            return self

        # Step 1: Load the shared embedder first so KeyBERT can reuse it
        embedder = SentenceEmbedder(model_name=self.embedding_model)

        # Step 2: Extract keyphrases (pass embedder._model to avoid a second load)
        kp_extractor = KeyphraseExtractor(
            algorithm=self.kp_algorithm,
            n_keyphrases=self.n_keyphrases,
            pretrained_model=embedder._model,
            yake_use_position=self.yake_use_position,
            tfidf_ngram_range=self.tfidf_ngram_range,
            keybert_ngram_range=self.keybert_ngram_range,
        )
        doc_keyphrases = kp_extractor.extract(documents)

        # Step 3: Flatten keyphrases, track doc-frequency per phrase.
        # Sorting by doc-frequency before semantic dedup ensures the most
        # representative (common) phrases survive as GMM anchors — rare
        # one-off phrases are removed as near-duplicates of common ones.
        from collections import Counter as _Counter
        doc_freq: _Counter = _Counter()
        for kps in doc_keyphrases:
            for kp in set(kps):   # count each phrase once per document
                doc_freq[kp] += 1

        # Apply document-frequency filter before sorting.
        # min_anchor_df removes hapax phrases (too specific to one doc).
        # max_anchor_df_ratio removes ubiquitous phrases that appear in so many
        # docs they carry no class signal (e.g. "feel like" in 73 % of MH posts).
        n_docs = len(documents)
        max_df = int(self.max_anchor_df_ratio * n_docs) if self.max_anchor_df_ratio else n_docs
        all_keyphrases = [
            kp for kp, freq in sorted(
                doc_freq.items(), key=lambda x: (-x[1], x[0])
            )
            if self.min_anchor_df <= freq <= max_df
        ]

        if not all_keyphrases:
            self.topics_ = {}
            self.doc_topic_probs_ = []
            return self

        # Step 4: Embed keyphrases
        keyphrase_embeddings = self._maybe_normalize(
            embedder.encode(all_keyphrases, batch_size=32)
        )

        # Step 4b: Greedy semantic deduplication of the anchor pool.
        # Processing in doc-frequency order means common phrases are kept as
        # seeds; near-synonyms that are rarer get dropped. This removes the
        # tight synonym clouds that RAKE/YAKE produce (e.g. "panic attack",
        # "panic attacks", "panicking") and forces GMM components to spread
        # across the actual topic space.
        # S3: for n ≤ _DEDUP_MATRIX_MAX_N, one cosine_similarity call covers
        # all pairwise sims in a single BLAS op; above that threshold the
        # original per-row loop runs to avoid an oversized n×n allocation.
        if self.anchor_dedup_threshold is not None:
            keyphrase_embeddings, all_keyphrases = _greedy_dedup(
                keyphrase_embeddings, all_keyphrases, self.anchor_dedup_threshold
            )

        # Step 5: Cluster keyphrase embeddings
        clusterer = GMMTopicClusterer(
            n_topics=self.n_topics,
            soft_threshold=self.soft_threshold,
        )
        clusterer.fit(keyphrase_embeddings)
        self._embedder  = embedder
        self._clusterer = clusterer

        # Step 6: Extract content words from all documents
        content_words = extract_content_words(documents, min_df=self.min_df,
                                               method=self.content_words_method)

        # Step 7: Project vocabulary onto clusters
        if content_words:
            projector = WordTopicProjector(clusterer=clusterer, embedder=embedder)
            self.topics_ = projector.project_vocabulary(
                content_words,
                min_prob=self.soft_threshold,
                mmr_diversity=self.mmr_diversity,
                mmr_max_sim=self.mmr_max_sim,
                normalize_embeddings=self.normalize_embeddings,
            )
            # Limit to top_n_words per topic
            for tid in self.topics_:
                self.topics_[tid] = self.topics_[tid][: self.top_n_words]
        else:
            self.topics_ = {tid: [] for tid in range(self.n_topics)}

        # Step 8: Compute document-level topic distributions by encoding full
        # document texts and computing cosine similarity to each GMM centroid,
        # then normalising to a probability distribution via softmax.
        # Using cosine similarity to centroids (rather than GMM predict_proba
        # or averaged keyphrase posteriors) is robust across text lengths and
        # keyphrase extractor quality: a document finds its natural position
        # relative to topic anchor regions regardless of what phrases were
        # extracted from it. GMM posteriors are unreliable when training
        # (short phrases) and prediction (long documents) distributions differ.
        doc_embeddings = self._maybe_normalize(embedder.encode(documents, batch_size=32))
        centroids = clusterer.cluster_centers_
        # Shape: (n_docs, n_topics)
        sim_to_centroids = cosine_similarity(doc_embeddings, centroids)
        # Softmax to convert similarities to a probability distribution
        exp_sim = np.exp(sim_to_centroids - sim_to_centroids.max(axis=1, keepdims=True))
        doc_probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)
        self.doc_topic_probs_ = list(doc_probs)

        self._n_fitted_docs = len(documents)

        return self

    def _maybe_normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x / norms

    def transform(self, documents: List[str]) -> np.ndarray:
        """Return topic distributions for new documents without re-fitting.

        Encodes each document with the fitted embedder, computes cosine
        similarity to the fitted GMM centroids, and applies softmax to
        produce a proper probability distribution over topics.

        Args:
            documents: List of document strings.

        Returns:
            numpy.ndarray of shape (n_docs, n_topics).

        Raises:
            RuntimeError: If called before fit().
        """
        if self._clusterer is None or not self._clusterer.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        if not documents:
            return np.zeros((0, self.n_topics))

        doc_embeddings = self._maybe_normalize(
            self._embedder.encode(documents, batch_size=32)
        )
        centroids = self._clusterer.cluster_centers_
        sim = cosine_similarity(doc_embeddings, centroids)
        exp_sim = np.exp(sim - sim.max(axis=1, keepdims=True))
        return exp_sim / exp_sim.sum(axis=1, keepdims=True)

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the model and return topic distributions for the training documents.

        Equivalent to calling fit(documents) then retrieving doc_topic_probs_
        as a numpy array. Provided for sklearn-style API compatibility.

        Args:
            documents: List of document strings.

        Returns:
            numpy.ndarray of shape (n_docs, n_topics).
        """
        self.fit(documents)
        if not self.doc_topic_probs_:
            return np.zeros((0, self.n_topics))
        return np.array(self.doc_topic_probs_)

    def get_topic_words(self, topic_id: int, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get top words for a specific topic.

        Args:
            topic_id: Topic ID.
            n: Number of words to return (None for all).

        Returns:
            List of (word, probability) tuples.
        """
        if self.topics_ is None:
            return []

        if topic_id not in self.topics_:
            return []

        words = self.topics_[topic_id]
        if n is not None:
            return words[:n]
        return words

    def get_document_topics(self, doc_idx: int) -> Dict[int, float]:
        """Get topic distribution for a document by index.

        Args:
            doc_idx: Index of the document in the list passed to fit().

        Returns:
            Dictionary mapping topic_id to probability.
        """
        if self.doc_topic_probs_ is None or doc_idx < 0 or doc_idx >= len(self.doc_topic_probs_):
            return {}

        probs = self.doc_topic_probs_[doc_idx]

        # Convert to dict, filtering out zero probabilities
        return {i: float(probs[i]) for i in range(len(probs)) if probs[i] > 0}

    def print_topics(self, n_words: int = 10):
        """Pretty-print top words per topic.

        Args:
            n_words: Number of top words to print per topic.
        """
        if self.topics_ is None:
            print("Model has not been fitted.")
            return

        print(f"\n{'='*60}")
        print(f"KATM Topics (n_topics={self.n_topics})")
        print(f"{'='*60}")

        for topic_id in sorted(self.topics_.keys()):
            words = self.get_topic_words(topic_id, n=n_words)
            if words:
                word_str = ", ".join([f"{w} ({p:.3f})" for w, p in words])
                print(f"\nTopic {topic_id}:")
                print(f"  {word_str}")
            else:
                print(f"\nTopic {topic_id}: (no words above threshold)")

        print(f"\n{'='*60}\n")


# ── S3: vectorized anchor deduplication ──────────────────────────────────────

_DEDUP_MATRIX_MAX_N = 8000  # n×n float32 at this size ≈ 244 MB


def _greedy_dedup(
    embeddings: np.ndarray,
    phrases: List[str],
    threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    """Greedy semantic dedup of the anchor pool.

    Processes candidates in doc-frequency order (already sorted by caller);
    keeps a phrase iff its max cosine similarity to all previously kept phrases
    is below `threshold`. Logic is identical to the original per-row loop;
    only the implementation changes:

    - n ≤ _DEDUP_MATRIX_MAX_N: one cosine_similarity call for the full n×n
      pairwise matrix (single BLAS op), then a greedy pass over boolean-indexed
      rows — replaces n separate sklearn calls with one.
    - n > _DEDUP_MATRIX_MAX_N: falls back to the original per-row loop to avoid
      allocating an oversized matrix.
    """
    n = len(embeddings)
    if n == 0:
        return embeddings, phrases

    if n > _DEDUP_MATRIX_MAX_N:
        kept_indices = [0]
        kept_embs = [embeddings[0]]
        for i in range(1, n):
            if float(np.max(cosine_similarity([embeddings[i]], kept_embs)[0])) < threshold:
                kept_indices.append(i)
                kept_embs.append(embeddings[i])
        return np.array(kept_embs), [phrases[i] for i in kept_indices]

    sim_matrix = cosine_similarity(embeddings)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[0] = True
    for i in range(1, n):
        if float(sim_matrix[i, kept_mask].max()) < threshold:
            kept_mask[i] = True

    idx = np.where(kept_mask)[0]
    return embeddings[idx], [phrases[i] for i in idx]