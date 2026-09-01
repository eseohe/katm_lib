"""Topic assignment module for KATM."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import GMMTopicClusterer
from .embedding import SentenceEmbedder


class WordTopicProjector:
    """Projects vocabulary words onto topic clusters via cosine similarity to centroids."""

    def __init__(self, clusterer: GMMTopicClusterer, embedder: SentenceEmbedder):
        """Initialize WordTopicProjector.

        Args:
            clusterer: Fitted GMMTopicClusterer instance.
            embedder: SentenceEmbedder instance for encoding words.
        """
        self.clusterer = clusterer
        self.embedder = embedder

    def project_vocabulary(
        self,
        vocab: List[str],
        min_prob: float = 0.15,
        mmr_diversity: float = 0.2,
        mmr_max_sim: float = 0.85,
        normalize_embeddings: bool = False,
        exclusive_assignment: bool = False,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Project vocabulary words onto topics via cosine similarity + MMR.

        For each topic:
        1. Score all vocab words by cosine similarity to the GMM centroid.
        2. Filter by min_prob, then keep the top-100 candidates.
        3. Apply MMR (Maximal Marginal Relevance) to select up to 50 words
           that are both relevant to the centroid and diverse from each other.
           Any candidate whose max cosine similarity to already-selected words
           exceeds mmr_max_sim is hard-excluded before scoring, preventing
           near-morphological duplicates (e.g. anxious/anxiousness, sim=0.93)
           from slipping through when MMR lambda is low.

        Args:
            vocab: List of vocabulary words.
            min_prob: Minimum cosine similarity to the centroid for inclusion.
            mmr_diversity: MMR lambda — fraction penalising inter-word
                similarity. 0.2 means 80 % relevance / 20 % diversity.
            mmr_max_sim: Hard dedup cap. Any candidate with cosine similarity
                above this threshold to any already-selected word is excluded
                entirely, regardless of its MMR score. Default 0.85.
            exclusive_assignment: If True, a word only competes for a topic
                if that topic is its single best (argmax) match across all
                n_topics centroid similarities. Default False (original
                behavior): each topic is filled independently, so a word
                above min_prob for several centroids at once can be listed
                in all of their word lists — on some corpora adjacent topics
                can then share a large fraction of their displayed words,
                wasting slots a more distinctive word could have filled.
                This is a different mechanism from mmr_max_sim/mmr_diversity
                (which only dedup *within* one topic's own list) — this is
                what makes topics compete with each other for the same word.

        Returns:
            Dictionary mapping topic_id to list of (word, similarity) tuples,
            ordered by MMR selection order (best first).
        """
        if not vocab:
            return {}

        # Embed all vocabulary words
        word_embeddings = self.embedder.encode(vocab, batch_size=32)
        if normalize_embeddings:
            norms = np.linalg.norm(word_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            word_embeddings = word_embeddings / norms

        # Get cluster centroids (means) from fitted GMM
        centroids = self.clusterer.cluster_centers_

        # Cosine similarity matrix: shape (len(vocab), n_topics)
        sim_matrix = cosine_similarity(word_embeddings, centroids)

        n_topics = self.clusterer.n_topics
        topic_words: Dict[int, List[Tuple[str, float]]] = {
            tid: [] for tid in range(n_topics)
        }

        # Cross-topic exclusivity gate (see exclusive_assignment's docstring
        # above) — computed once, against the same sim_matrix every topic
        # already uses, so every topic judges "who owns this word" the same way.
        best_topic_per_word = np.argmax(sim_matrix, axis=1) if exclusive_assignment else None

        for topic_id in range(n_topics):
            sim_to_centroid = sim_matrix[:, topic_id]

            # Filter candidates above threshold and rank by similarity
            candidate_indices = np.where(sim_to_centroid >= min_prob)[0]
            if exclusive_assignment:
                candidate_indices = candidate_indices[
                    best_topic_per_word[candidate_indices] == topic_id
                ]
            if len(candidate_indices) == 0:
                continue

            ranked = candidate_indices[
                np.argsort(sim_to_centroid[candidate_indices])[::-1]
            ]
            top_indices = ranked[:100].tolist()

            # S4 — incremental MMR: maintain a running max_sim_to_selected
            # vector updated by one cosine_similarity call per selection step
            # (O(N·K)) rather than recomputing the full selected matrix each
            # iteration (O(N·K²)).
            selected = [top_indices[0]]
            unselected = top_indices[1:]

            # Seed: sims of all unselected candidates to the first selected word.
            # Guard the degenerate case of exactly one candidate above min_prob
            # (unselected == []) — cosine_similarity rejects a 0-sample array,
            # and the while loop below already no-ops on an empty `unselected`,
            # so an empty float array here is the correct thing to reach it with.
            if unselected:
                max_sim_to_selected = cosine_similarity(
                    word_embeddings[unselected],
                    word_embeddings[[selected[0]]],
                )[:, 0].copy()
            else:
                max_sim_to_selected = np.array([], dtype=float)

            while len(selected) < 50 and unselected:
                # Hard dedup: drop near-duplicates of already-selected words
                eligible_local = np.where(max_sim_to_selected < mmr_max_sim)[0]
                if len(eligible_local) == 0:
                    break

                eligible_global = [unselected[i] for i in eligible_local]
                relevance = sim_to_centroid[eligible_global]
                mmr_scores = (
                    (1 - mmr_diversity) * relevance
                    - mmr_diversity * max_sim_to_selected[eligible_local]
                )

                best_in_eligible = int(np.argmax(mmr_scores))
                best_local_idx = int(eligible_local[best_in_eligible])
                new_sel = unselected.pop(best_local_idx)
                selected.append(new_sel)
                max_sim_to_selected = np.delete(max_sim_to_selected, best_local_idx)

                # Extend max_sim incrementally — one BLAS call, element-wise max.
                if unselected:
                    new_sims = cosine_similarity(
                        word_embeddings[unselected],
                        word_embeddings[[new_sel]],
                    )[:, 0]
                    np.maximum(max_sim_to_selected, new_sims, out=max_sim_to_selected)

            topic_words[topic_id] = [
                (vocab[idx], float(sim_to_centroid[idx]))
                for idx in selected
            ]

        return topic_words