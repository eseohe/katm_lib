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
        mmr_diversity: float = 0.3,
        mmr_max_sim: float = 0.75,
        normalize_embeddings: bool = False,
        lemma_dedup: bool = False,
        language: str = "english",
        centroid_mode: str = "mean",
        anchor_embeddings: np.ndarray = None,
        anchor_labels: np.ndarray = None,
        centroid_blend_alpha: float = 0.5,
        centroid_top_k: int = 3,
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
           from slipping through when MMR lambda is low. Empirically this
           alone doesn't fully solve the problem in a compressed/miscalibrated
           embedding space (unrelated words can already sit above mmr_max_sim,
           so morphological variants of one word don't stand out as "more
           similar than usual") - see lemma_dedup below for a second,
           independent guard against the same failure mode.

        Args:
            vocab: List of vocabulary words.
            min_prob: Minimum cosine similarity to the centroid for inclusion.
            mmr_diversity: MMR lambda — fraction penalising inter-word
                similarity. Default 0.3 means 70 % relevance / 30 % diversity
                (raised from an original 0.2 - see topic_model.py's KATM
                docstring for the sweep that motivated the change).
            mmr_max_sim: Hard dedup cap. Any candidate with cosine similarity
                above this threshold to any already-selected word is excluded
                entirely, regardless of its MMR score. Default 0.75 (lowered
                from an original 0.85 - see topic_model.py's KATM docstring).
            lemma_dedup: If True, also exclude any candidate whose lemma
                (spaCy, isolated-token — see utils.lemma_key_map) matches an
                already-selected word's lemma *within that topic*. This is a
                within-topic guard only: it stops one topic from listing
                e.g. both "gebückt" and "bücken", but does not stop the same
                word from being independently selected into several different
                topics (a different phenomenon — see mmr_max_sim's docstring
                caveat above; that's the embedding space treating the word as
                relevant to multiple centroids, not a morphology problem).
                Default False, fully backward-compatible.
            language: "english" (default) or "german". Only used when
                lemma_dedup=True, to select spaCy's lemmatizer model.
            centroid_mode: How each topic's relevance score is computed for
                the vocabulary projection. "mean" (default, unchanged
                behavior) - cosine similarity to the GMM centroid, the plain
                arithmetic mean of that cluster's anchor embeddings.

                Diagnosed directly on the bipolar per-class mental-health
                corpus: when a cluster mixes a tight, lexically-repetitive
                subgroup of anchors (generic condition-identity language,
                e.g. "bipolar disorder"/"bipolar type"/"im bipolar" - all
                close to each other) with a scattered subgroup (many
                different medication names, each in its own neighborhood),
                the mean is pulled toward whichever subgroup is internally
                tighter, not whichever is more numerous - a cluster that is
                genuinely 35% medication anchors produced a word list only
                4/20 medication-related, because the centroid sat 0.96
                cosine-similar to the non-medication sub-centroid but only
                0.87 to the medication sub-centroid. "mean" alone
                systematically underrepresents any scattered subgroup
                regardless of its share of the cluster.

                Other modes require anchor_embeddings/anchor_labels (below)
                and use each vocabulary word's similarity to individual
                anchors in the cluster instead of/alongside the mean:
                  - "max": relevance = max cosine similarity to any single
                    anchor in the cluster. Stops penalising scattered
                    subgroups entirely (measured: bipolar medication share
                    in the top-20 rose from 4/20 to 6/20) but can admit
                    words tied to just one unusual/outlier anchor rather
                    than the cluster as a whole (noisier).
                  - "top_k_anchors": relevance = mean of the top-k highest
                    per-anchor similarities (k=centroid_top_k, default 3) -
                    a middle ground between "mean" (diluted by the whole
                    cluster) and "max" (only one anchor, noisy); requires
                    the word to be genuinely close to several anchors, not
                    just one.
                  - "blend": relevance = centroid_blend_alpha * mean_sim +
                    (1 - centroid_blend_alpha) * max_sim (default alpha=0.5)
                    - a direct weighted combination of "mean" and "max".
            anchor_embeddings: (n_anchors, dim) array of the anchor pool's
                own embeddings - required (and used only) when
                centroid_mode != "mean". KATM already computes this as
                self.anchor_embeddings_ (Step 4/4d output, the exact
                embeddings clusterer.fit() saw).
            anchor_labels: (n_anchors,) array of each anchor's hard GMM
                cluster assignment - required (and used only) when
                centroid_mode != "mean". KATM already computes this as
                self.anchor_labels_.
            centroid_blend_alpha: Weight on the mean-centroid term when
                centroid_mode="blend" (0 = pure "max", 1 = pure "mean").
                Default 0.5. Unused for other modes.
            centroid_top_k: Number of nearest anchors averaged when
                centroid_mode="top_k_anchors". Default 3. Unused for other
                modes.
            exclusive_assignment: If True, a word only competes for a topic
                if that topic is its single best (argmax) match across all
                n_topics mean-centroid similarities - independent of
                centroid_mode, always judged against the plain mean
                centroids, since exclusivity needs one consistent
                cross-topic comparison. Default False (original behavior):
                each topic is filled independently, so a word above
                min_prob for several centroids at once can be listed in all
                of their word lists. Diagnosed directly on bipolar per-class
                data: with exclusive_assignment=False, adjacent topics
                shared 4-7 of their 15 displayed words (e.g. "bipolars",
                "psychiatric", "seroquel" all appearing in two or three
                topics at once) - wasted slots that a genuinely distinctive
                word (e.g. a specific medication name) could have filled
                instead. This is a different phenomenon from mmr_max_sim/
                mmr_diversity (which only dedup *within* one topic's own
                list) and from centroid_mode (which only changes *how*
                similarity to one topic is scored) - this is the first
                mechanism in the pipeline that makes topics compete with
                each other for the same word.

        Returns:
            Dictionary mapping topic_id to list of (word, similarity) tuples,
            ordered by MMR selection order (best first).
        """
        if not vocab:
            return {}

        word_lemma = None
        if lemma_dedup:
            from .utils import lemma_key_map
            word_lemma = lemma_key_map(vocab, language)

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

        if centroid_mode != "mean" and (anchor_embeddings is None or anchor_labels is None):
            raise ValueError(
                "centroid_mode != 'mean' requires anchor_embeddings and anchor_labels "
                "(KATM already computes these as self.anchor_embeddings_/self.anchor_labels_)"
            )

        n_topics = self.clusterer.n_topics
        topic_words: Dict[int, List[Tuple[str, float]]] = {
            tid: [] for tid in range(n_topics)
        }

        # Cross-topic exclusivity gate (see exclusive_assignment's docstring
        # above): always judged against the plain mean centroids, regardless
        # of centroid_mode, so every topic uses the same consistent signal
        # to decide which one "owns" a given word.
        best_topic_per_word = np.argmax(sim_matrix, axis=1) if exclusive_assignment else None

        for topic_id in range(n_topics):
            if centroid_mode == "mean":
                sim_to_centroid = sim_matrix[:, topic_id]
            else:
                # Per-anchor similarity for this topic's own anchors only -
                # computed once per topic (n_vocab x n_anchors_in_topic is
                # cheap: a few thousand vocab words x a few hundred anchors).
                topic_anchor_mask = anchor_labels == topic_id
                topic_anchor_embs = anchor_embeddings[topic_anchor_mask]
                if len(topic_anchor_embs) == 0:
                    sim_to_centroid = sim_matrix[:, topic_id]  # degenerate: fall back to mean
                else:
                    sim_to_anchors = cosine_similarity(word_embeddings, topic_anchor_embs)
                    if centroid_mode == "max":
                        sim_to_centroid = sim_to_anchors.max(axis=1)
                    elif centroid_mode == "top_k_anchors":
                        k_eff = max(1, min(centroid_top_k, sim_to_anchors.shape[1]))
                        top_k = np.sort(sim_to_anchors, axis=1)[:, -k_eff:]
                        sim_to_centroid = top_k.mean(axis=1)
                    elif centroid_mode == "blend":
                        max_sim = sim_to_anchors.max(axis=1)
                        sim_to_centroid = (
                            centroid_blend_alpha * sim_matrix[:, topic_id]
                            + (1 - centroid_blend_alpha) * max_sim
                        )
                    else:
                        raise ValueError(
                            f"Unknown centroid_mode {centroid_mode!r} - must be one of "
                            "'mean', 'max', 'top_k_anchors', 'blend'"
                        )

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
            # Within-topic lemma dedup (see lemma_dedup docstring above) -
            # reset per topic, since the same word/lemma legitimately being
            # relevant to *different* topics is a separate phenomenon this
            # is not meant to suppress.
            selected_lemmas = {word_lemma[vocab[selected[0]]]} if word_lemma else None

            # Seed: sims of all unselected candidates to the first selected word.
            # Guard the degenerate case of exactly one candidate above min_prob
            # (unselected == []) — cosine_similarity rejects a 0-sample array,
            # and there's nothing to seed against anyway; the while loop below
            # already no-ops on an empty `unselected`, so an empty float array
            # here is the correct (and only) thing needed to reach it safely.
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
                if word_lemma is not None and len(eligible_local) > 0:
                    lemma_ok = np.array([
                        word_lemma[vocab[unselected[i]]] not in selected_lemmas
                        for i in eligible_local
                    ])
                    eligible_local = eligible_local[lemma_ok]
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
                if word_lemma is not None:
                    selected_lemmas.add(word_lemma[vocab[new_sel]])
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
