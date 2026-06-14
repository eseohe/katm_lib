"""S3 + S4 experimental KATM variant.

S3 — Vectorized anchor deduplication:
  The original topic_model.py calls cosine_similarity([emb], kept_embs) inside
  a Python loop (O(n²) separate sklearn calls). This replaces that with a single
  cosine_similarity(all_embeddings, all_embeddings) call (one BLAS op), then runs
  the greedy keep/drop pass using numpy boolean indexing over the precomputed
  matrix. The deduplication logic is identical; only the number of sklearn calls
  changes from O(n) to O(1).

  Memory note: the full pairwise matrix is (n × n) float32. For n = 5 000 anchors
  that is ~100 MB. If memory is a concern, reduce max_anchor_df_ratio or
  min_anchor_df to keep n small; block-based processing is a future option.

S4 — Incremental MMR:
  Delegates to WordTopicProjectorFast (topic_assignment_fast_exp.py), which
  maintains a running max_sim_to_selected vector updated by one cosine_similarity
  call per selection step (O(N·K)) rather than recomputing the full selected
  matrix each iteration (O(N·K²)).

This file is experimental — compare against the original KATM class on your
corpus before deciding whether to merge. If topic word sets change, coherence
scores or downstream purity may differ; only adopt if timing improves and quality
is preserved.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import GMMTopicClusterer
from .embedding import SentenceEmbedder
from .keyphrase_extractor import KeyphraseExtractor
from .topic_assignment_fast import WordTopicProjectorFast
from .utils import extract_content_words


class KATMFast:
    """KATM with S3 (vectorized anchor dedup) + S4 (incremental MMR).

    Drop-in replacement for KATM — identical constructor signature and public API.
    """

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

        self.topics_: Optional[Dict[int, List[Tuple[str, float]]]] = None
        self.doc_topic_probs_: Optional[List[np.ndarray]] = None
        self._n_fitted_docs = 0
        self._embedder: Optional[SentenceEmbedder] = None
        self._clusterer: Optional[GMMTopicClusterer] = None

    def fit(self, documents: List[str]) -> "KATMFast":
        if not documents:
            self.topics_ = {}
            self.doc_topic_probs_ = []
            return self

        embedder = SentenceEmbedder(model_name=self.embedding_model)

        kp_extractor = KeyphraseExtractor(
            algorithm=self.kp_algorithm,
            n_keyphrases=self.n_keyphrases,
            pretrained_model=embedder._model,
            yake_use_position=self.yake_use_position,
            tfidf_ngram_range=self.tfidf_ngram_range,
            keybert_ngram_range=self.keybert_ngram_range,
        )
        doc_keyphrases = kp_extractor.extract(documents)

        from collections import Counter as _Counter
        doc_freq: _Counter = _Counter()
        for kps in doc_keyphrases:
            for kp in set(kps):
                doc_freq[kp] += 1

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

        keyphrase_embeddings = self._maybe_normalize(
            embedder.encode(all_keyphrases, batch_size=32)
        )

        # ── S3: Vectorized anchor deduplication ──────────────────────────────
        # Original: O(n) separate cosine_similarity([emb], kept_embs) calls.
        # New: one cosine_similarity call for the full pairwise matrix, then a
        # greedy pass using boolean indexing over the precomputed rows.
        if self.anchor_dedup_threshold is not None:
            keyphrase_embeddings, all_keyphrases = _vectorized_dedup(
                keyphrase_embeddings, all_keyphrases, self.anchor_dedup_threshold
            )
        # ── end S3 ───────────────────────────────────────────────────────────

        clusterer = GMMTopicClusterer(
            n_topics=self.n_topics,
            soft_threshold=self.soft_threshold,
        )
        clusterer.fit(keyphrase_embeddings)
        self._embedder = embedder
        self._clusterer = clusterer

        content_words = extract_content_words(documents, min_df=self.min_df,
                                               method=self.content_words_method)

        # ── S4: incremental MMR via WordTopicProjectorFast ───────────────────
        if content_words:
            projector = WordTopicProjectorFast(clusterer=clusterer, embedder=embedder)
            self.topics_ = projector.project_vocabulary(
                content_words,
                min_prob=self.soft_threshold,
                mmr_diversity=self.mmr_diversity,
                mmr_max_sim=self.mmr_max_sim,
                normalize_embeddings=self.normalize_embeddings,
            )
            for tid in self.topics_:
                self.topics_[tid] = self.topics_[tid][: self.top_n_words]
        else:
            self.topics_ = {tid: [] for tid in range(self.n_topics)}
        # ── end S4 ───────────────────────────────────────────────────────────

        doc_embeddings = self._maybe_normalize(embedder.encode(documents, batch_size=32))
        centroids = clusterer.cluster_centers_
        sim_to_centroids = cosine_similarity(doc_embeddings, centroids)
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
        self.fit(documents)
        if not self.doc_topic_probs_:
            return np.zeros((0, self.n_topics))
        return np.array(self.doc_topic_probs_)

    def get_topic_words(self, topic_id: int, n: Optional[int] = None) -> List[Tuple[str, float]]:
        if self.topics_ is None or topic_id not in self.topics_:
            return []
        words = self.topics_[topic_id]
        return words[:n] if n is not None else words

    def get_document_topics(self, doc_idx: int) -> Dict[int, float]:
        if self.doc_topic_probs_ is None or doc_idx < 0 or doc_idx >= len(self.doc_topic_probs_):
            return {}
        probs = self.doc_topic_probs_[doc_idx]
        return {i: float(probs[i]) for i in range(len(probs)) if probs[i] > 0}

    def print_topics(self, n_words: int = 10):
        if self.topics_ is None:
            print("Model has not been fitted.")
            return
        print(f"\n{'='*60}\nKATMFast Topics (n_topics={self.n_topics})\n{'='*60}")
        for topic_id in sorted(self.topics_.keys()):
            words = self.get_topic_words(topic_id, n=n_words)
            if words:
                word_str = ", ".join([f"{w} ({p:.3f})" for w, p in words])
                print(f"\nTopic {topic_id}:\n  {word_str}")
            else:
                print(f"\nTopic {topic_id}: (no words above threshold)")
        print(f"\n{'='*60}\n")


_DEDUP_MATRIX_MAX_N = 8000  # above this, full n×n matrix would exceed ~250 MB


def _vectorized_dedup(
    embeddings: np.ndarray,
    phrases: List[str],
    threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    """Greedy semantic deduplication using a precomputed pairwise similarity matrix.

    Semantically identical to the original loop in topic_model.py — processes
    candidates in order, keeps a candidate iff its max cosine similarity to all
    previously kept candidates is below `threshold`. The difference is that all
    pairwise similarities are computed in a single BLAS call rather than n
    separate sklearn calls.

    Falls back to the original per-row approach when n > _DEDUP_MATRIX_MAX_N to
    avoid allocating an (n × n) matrix that would exhaust RAM.

    Args:
        embeddings: (n, d) float32 array, already in doc-frequency order.
        phrases:    Parallel list of phrase strings.
        threshold:  Maximum cosine similarity to any kept phrase.

    Returns:
        Tuple of (filtered_embeddings, filtered_phrases).
    """
    n = len(embeddings)
    if n == 0:
        return embeddings, phrases

    if n > _DEDUP_MATRIX_MAX_N:
        # Original per-row loop — avoids n×n memory allocation.
        kept_indices = [0]
        kept_embs = [embeddings[0]]
        for i in range(1, n):
            sims = cosine_similarity([embeddings[i]], kept_embs)[0]
            if float(np.max(sims)) < threshold:
                kept_indices.append(i)
                kept_embs.append(embeddings[i])
        return np.array(kept_embs), [phrases[i] for i in kept_indices]

    # One BLAS call for the full pairwise matrix.
    sim_matrix = cosine_similarity(embeddings)  # (n, n)

    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[0] = True

    for i in range(1, n):
        # Max similarity of candidate i to all previously kept candidates.
        max_sim = float(sim_matrix[i, kept_mask].max())
        if max_sim < threshold:
            kept_mask[i] = True

    kept_indices = np.where(kept_mask)[0]
    return embeddings[kept_indices], [phrases[i] for i in kept_indices]