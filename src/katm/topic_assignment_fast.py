"""S4 experiment: incremental MMR for WordTopicProjector.

Replaces the O(N·K²) MMR loop in WordTopicProjector.project_vocabulary with an
O(N·K) incremental variant that maintains a running max_sim_to_selected vector.
Instead of recomputing cosine_similarity(unselected, selected) from scratch each
iteration (which grows by one column each time), we do one extra cosine_similarity
call against the single newly-added word and take the element-wise maximum.

Output is numerically identical to the original; only runtime changes.
This file is experimental — only merge into topic_assignment.py if benchmarks
confirm a speedup without quality loss.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import GMMTopicClusterer
from .embedding import SentenceEmbedder


class WordTopicProjectorFast:
    """Drop-in replacement for WordTopicProjector using incremental MMR (S4)."""

    def __init__(self, clusterer: GMMTopicClusterer, embedder: SentenceEmbedder):
        self.clusterer = clusterer
        self.embedder = embedder

    def project_vocabulary(
        self,
        vocab: List[str],
        min_prob: float = 0.15,
        mmr_diversity: float = 0.2,
        mmr_max_sim: float = 0.85,
        normalize_embeddings: bool = False,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Same interface as WordTopicProjector.project_vocabulary."""
        if not vocab:
            return {}

        word_embeddings = self.embedder.encode(vocab, batch_size=32)
        if normalize_embeddings:
            norms = np.linalg.norm(word_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            word_embeddings = word_embeddings / norms

        centroids = self.clusterer.cluster_centers_
        sim_matrix = cosine_similarity(word_embeddings, centroids)

        n_topics = self.clusterer.n_topics
        topic_words: Dict[int, List[Tuple[str, float]]] = {
            tid: [] for tid in range(n_topics)
        }

        for topic_id in range(n_topics):
            sim_to_centroid = sim_matrix[:, topic_id]

            candidate_indices = np.where(sim_to_centroid >= min_prob)[0]
            if len(candidate_indices) == 0:
                continue

            ranked = candidate_indices[
                np.argsort(sim_to_centroid[candidate_indices])[::-1]
            ]
            top_indices = ranked[:100].tolist()

            if len(top_indices) == 1:
                topic_words[topic_id] = [
                    (vocab[top_indices[0]], float(sim_to_centroid[top_indices[0]]))
                ]
                continue

            # ── Incremental MMR (S4) ────────────────────────────────────────
            # Seed: best candidate by centroid similarity.
            selected = [top_indices[0]]
            unselected = top_indices[1:]

            # Initialise running max_sim to sims of all unselected vs the seed.
            max_sim = cosine_similarity(
                word_embeddings[unselected],
                word_embeddings[[selected[0]]],
            )[:, 0].copy()

            while len(selected) < 50 and unselected:
                eligible_local = np.where(max_sim < mmr_max_sim)[0]
                if len(eligible_local) == 0:
                    break

                eligible_global = [unselected[i] for i in eligible_local]
                relevance = sim_to_centroid[eligible_global]
                mmr_scores = (
                    (1 - mmr_diversity) * relevance
                    - mmr_diversity * max_sim[eligible_local]
                )

                best_in_eligible = int(np.argmax(mmr_scores))
                best_local_idx = int(eligible_local[best_in_eligible])

                new_sel = unselected.pop(best_local_idx)
                selected.append(new_sel)
                max_sim = np.delete(max_sim, best_local_idx)

                # Extend max_sim incrementally — one BLAS call, then element-wise max.
                if unselected:
                    new_sims = cosine_similarity(
                        word_embeddings[unselected],
                        word_embeddings[[new_sel]],
                    )[:, 0]
                    np.maximum(max_sim, new_sims, out=max_sim)
            # ── end incremental MMR ─────────────────────────────────────────

            topic_words[topic_id] = [
                (vocab[idx], float(sim_to_centroid[idx]))
                for idx in selected
            ]

        return topic_words