"""Clustering module for KATM using Gaussian Mixture Models."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def select_n_topics(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 20,
    method: str = "bic",
    random_state: int = 42,
    covariance_type: str = "diag",
) -> Tuple[int, Dict[str, list]]:
    """Diagnostic helper for choosing n_topics — not wired into KATM itself.

    Fits a GMM at every k in [k_min, k_max] (inclusive) on `embeddings` and
    picks the best k according to `method`. Call this yourself on the same
    anchor/keyphrase embeddings you'd pass to GMMTopicClusterer.fit (e.g.
    KeyphraseExtractor's output, embedded), inspect the returned scores, and
    pass the k you choose as KATM(n_topics=...). There is no n_topics="auto"
    mode in KATM — this only helps you pick the number up front:

    - "bic" (default): minimize Bayesian Information Criterion. This is the
      standard, principled criterion for choosing the number of GMM
      components — unlike a k-means inertia elbow, BIC already includes a
      penalty for model complexity (number of parameters), so it doesn't
      just monotonically improve as k grows and require a subjective
      "elbow" judgment call. Preferred over "elbow" here for that reason.
    - "aic": minimize Akaike Information Criterion. Penalizes complexity
      more weakly than BIC, so tends to prefer larger k. Offered as an
      alternative when BIC's preference for fewer, larger topics feels too
      conservative for a given corpus.
    - "elbow": classic knee-of-the-curve heuristic applied to the BIC curve
      itself (rather than raw log-likelihood/inertia) — picks the k where
      the marginal BIC improvement drops off most sharply, via maximum
      absolute discrete second derivative. Provided for parity with the
      "elbow method" request, but "bic" is the recommended default: it
      picks a k directly rather than inferring one from curve geometry,
      which is more robust on the small/noisy k-ranges typical here.
    - "silhouette": maximize mean silhouette score (sklearn.metrics.
      silhouette_score) of each k's hard cluster assignment (gmm.predict()).
      Unlike bic/aic (likelihood-based - "does this k explain the data
      well as a generative model") this is purely geometric - "how well
      separated are the resulting clusters from each other, relative to
      their own internal spread" - so it answers a different question and
      can disagree with bic/aic on which k is "best". A degenerate k where
      the fit only produces one non-empty cluster scores -1.0 (worst
      possible), so it can't silently win by collapsing. Only computed when
      method="silhouette" is actually requested (unlike bic/aic, which are
      cheap log-likelihood evaluations always kept for diagnostics) -
      silhouette_score is O(n^2) in the number of embeddings, materially
      more expensive per k than bic/aic, so it isn't computed unconditionally.

    covariance_type="diag" (not "full") for these trial fits specifically —
    found necessary empirically. With "full" covariance, a GMM component in
    D=768 (typical sentence-embedding dimensionality) has ~D(D+1)/2 ≈ 295k
    free parameters; BIC's complexity penalty on that is so large it swamps
    the likelihood term almost immediately; on a real 200-doc German test
    corpus this made BIC decrease monotonically across the *entire*
    k=5..20 range with no interior minimum, so "best k" was silently just
    whatever k_min happened to be — the range bound was deciding the
    answer, not the data. "diag" costs only D parameters/component, so the
    penalty scales with k roughly linearly instead of quadratically with
    dimensionality, which is enough for an interior minimum to actually
    emerge. This only affects this k-selection scan; GMMTopicClusterer's own
    covariance_type for the final production fit is independent (see its
    own docstring, default "full").

    Args:
        embeddings: numpy.ndarray of shape (N, D) — the anchor/keyphrase
            embeddings that will be clustered (i.e. call this on the same
            array you'll pass to GMMTopicClusterer.fit).
        k_min: Smallest number of components to try. Default 2.
        k_max: Largest number of components to try (inclusive). Default 20.
            Capped down to N-1 automatically if the corpus is too small to
            support it (GaussianMixture requires n_components <= n_samples).
        method: "bic" (default), "aic", "elbow", or "silhouette".
        random_state: Passed to every trial GMM fit, for reproducibility.
        covariance_type: Covariance type for the *trial* GMMs used only to
            score each k. Default "diag" — see rationale above. Exposed
            (rather than hardcoded) so "full" can still be requested if a
            corpus's embedding dimensionality is small enough that the
            quadratic penalty isn't a problem, or for direct comparison.

    Returns:
        (best_k, scores) where scores = {"k": [...], "bic": [...], "aic": [...]}
        for every k actually tried — kept for diagnostics/plotting even
        when method="elbow" only used the bic column to pick best_k.
    """
    n_samples = embeddings.shape[0]
    k_max = min(k_max, n_samples - 1)
    if k_max < k_min:
        # Corpus too small for the requested range at all; fall back to the
        # smallest sensible value rather than raising.
        k_min = k_max = max(1, min(k_min, n_samples))

    compute_silhouette = method == "silhouette"
    ks, bics, aics, silhouettes = [], [], [], []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type=covariance_type,
            random_state=random_state, max_iter=200,
        )
        gmm.fit(embeddings)
        ks.append(k)
        bics.append(gmm.bic(embeddings))
        aics.append(gmm.aic(embeddings))
        if compute_silhouette:
            labels = gmm.predict(embeddings)
            # A degenerate fit (all points in one component, or a component
            # so small every remaining point lands in the others) can leave
            # fewer than 2 non-empty clusters, which silhouette_score can't
            # score at all - treat as the worst possible score rather than
            # letting the k loop crash on a k that happens to collapse.
            if len(np.unique(labels)) < 2:
                silhouettes.append(-1.0)
            else:
                silhouettes.append(silhouette_score(embeddings, labels))

    scores = {"k": ks, "bic": bics, "aic": aics}
    if compute_silhouette:
        scores["silhouette"] = silhouettes

    if method == "bic":
        best_k = ks[int(np.argmin(bics))]
    elif method == "aic":
        best_k = ks[int(np.argmin(aics))]
    elif method == "elbow":
        if len(ks) < 3:
            best_k = ks[int(np.argmin(bics))]
        else:
            curve = np.array(bics)
            # Discrete second derivative — the "elbow" is where the BIC
            # curve's rate of improvement changes most sharply.
            second_deriv = np.diff(curve, 2)
            best_k = ks[int(np.argmax(np.abs(second_deriv))) + 1]
    elif method == "silhouette":
        best_k = ks[int(np.argmax(silhouettes))]
    else:
        raise ValueError(
            f"Unknown method: {method!r} (expected 'bic', 'aic', 'elbow', or 'silhouette')"
        )

    return best_k, scores


class GMMTopicClusterer:
    """Clusters embeddings using Gaussian Mixture Model."""

    def __init__(
        self,
        n_topics: int,
        soft_threshold: float = 0.15,
        random_state: int = 42,
        covariance_type: str = "full",
        discard_hub_clusters: bool = False,
        hub_overcluster_k: int = 5,
    ):
        """Initialize GMMTopicClusterer.

        Args:
            n_topics: Number of GMM components (topics) in the FINAL output -
                unchanged meaning regardless of discard_hub_clusters below;
                that flag changes how many components are fit internally,
                not how many topics come out.
            soft_threshold: Not currently used, kept for API compatibility.
            random_state: Random seed for reproducibility.
            covariance_type: sklearn GaussianMixture covariance type. Default
                "full" (unchanged behavior). "full" gives each component
                D(D+1)/2 free covariance parameters — with a typical
                sentence-embedding dimensionality (D=384-1024) that can be
                tens to hundreds of thousands of parameters per component,
                wildly underdetermined against a modest anchor pool (a few
                hundred to a couple thousand points after dedup). "diag" (D
                parameters/component) is much better conditioned in that
                regime and worth trying whenever the anchor pool is large
                relative to embedding dimensionality — e.g.
                KATM(keyphrase_scope="global"), whose anchor pool is the
                n_keyphrases_total budget (default 500) rather than the
                per-document-dedup pool "full" was validated against.
            discard_hub_clusters: False (default, unchanged behavior - fits
                exactly n_topics components). If True, fits
                n_topics + hub_overcluster_k components instead, then keeps
                only the n_topics whose centroid sits FURTHEST (lowest
                cosine similarity) from the global mean of the fitted
                embeddings, discarding the rest.

                Motivated by a direct empirical finding on real literary
                prose: with a fixed n_topics, GMM is sometimes forced to
                spend one component on a catch-all of leftover,
                individually-generic anchors (thing, way, good, course...)
                that have nowhere thematic to go - that component's
                centroid sat at cosine similarity 0.91 to the whole anchor
                pool's global mean (vs. 0.47-0.84 for every genuinely
                thematic cluster in the same run), and the pattern
                replicated on a second novel (highest-hubness cluster =
                filler both times; lowest-hubness clusters were reliably
                among the most distinctive topics). A component whose
                centroid is unusually close to "the average of everything"
                isn't really a topic - it's the generic residue every
                fixed-k clustering has nowhere else to put. Overclustering
                first gives that residue its own component(s) to absorb
                into WITHOUT taking one of the n_topics real slots, and
                discarding by global-mean proximity afterward removes it
                while still delivering exactly n_topics topics.

                Anchors originally assigned to a discarded component are
                reassigned to their nearest surviving centroid (by cosine
                similarity) rather than dropped, so every anchor still
                counts toward some topic's centroid/tightness - see fit()'s
                mechanism.

                Not yet validated as a net improvement on real topic
                quality (only the discovery signal itself was validated,
                two novels) - opt-in and exploratory, but particularly
                worth trying on literary/narrative prose, where this
                failure mode was diagnosed.
            hub_overcluster_k: Only used when discard_hub_clusters=True.
                How many extra components to fit beyond n_topics before
                discarding the highest-hubness ones back down to n_topics.
                Default 5 - a starting guess, not independently validated;
                exposed specifically so it can be swept per-corpus (a
                larger k gives the residue more room to spread into
                multiple components and get discarded cleanly, at the cost
                of a bigger/slower GMM fit and less certainty about which
                of the extra components are actually "real" topics being
                wrongly discarded rather than residue).
        """
        self.n_topics = n_topics
        self.soft_threshold = soft_threshold
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.discard_hub_clusters = discard_hub_clusters
        self.hub_overcluster_k = hub_overcluster_k
        self._gmm = None
        self._is_fitted = False
        # Only set when discard_hub_clusters=True - see fit()/cluster_centers_/predict().
        self._kept_component_indices: Optional[np.ndarray] = None
        self._kept_means: Optional[np.ndarray] = None
        # Diagnostics from the discard step, kept for inspection regardless
        # of whether the caller looks at them - not used internally beyond fit().
        self.hub_scores_: Optional[np.ndarray] = None  # (n_topics+k,) cosine sim to global mean, all fit components
        self.discarded_component_indices_: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> "GMMTopicClusterer":
        """Fit the GMM to embeddings.

        Args:
            embeddings: numpy.ndarray of shape (N, D).

        Returns:
            self for chaining.
        """
        n_components = self.n_topics
        if self.discard_hub_clusters:
            # Never request more components than samples - GaussianMixture
            # requires n_components <= n_samples.
            n_components = min(self.n_topics + self.hub_overcluster_k, len(embeddings))

        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
        )
        self._gmm.fit(embeddings)
        self._is_fitted = True

        if self.discard_hub_clusters and n_components > self.n_topics:
            global_mean = embeddings.mean(axis=0)
            hub_scores = cosine_similarity(self._gmm.means_, [global_mean])[:, 0]
            self.hub_scores_ = hub_scores
            # Keep the n_topics components FURTHEST from the global mean
            # (lowest hub score = most distinctive); discard the rest.
            order = np.argsort(hub_scores)  # ascending: most distinctive first
            self._kept_component_indices = np.sort(order[: self.n_topics])
            self.discarded_component_indices_ = np.sort(order[self.n_topics:])
            self._kept_means = self._gmm.means_[self._kept_component_indices]
        else:
            self._kept_component_indices = None
            self._kept_means = None
            self.hub_scores_ = None
            self.discarded_component_indices_ = None

        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Hard cluster assignment, remapped to 0..n_topics-1.

        Without discard_hub_clusters, this is exactly self._gmm.predict().
        With it: points whose raw GMM assignment is a KEPT component are
        remapped to that component's new compact index; points assigned to
        a DISCARDED component are reassigned to their nearest surviving
        centroid (by cosine similarity) instead of being dropped, so every
        input point still counts toward some topic.

        Args:
            embeddings: numpy.ndarray of shape (N, D).

        Returns:
            numpy.ndarray of shape (N,) with each embedding's component index.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict")

        raw = self._gmm.predict(embeddings)
        if self._kept_component_indices is None:
            return raw

        # Map original component index -> compact 0..n_topics-1 index, for
        # points that already landed on a kept component.
        remap = {orig: new for new, orig in enumerate(self._kept_component_indices)}
        discarded = set(self.discarded_component_indices_.tolist())

        out = np.empty(len(raw), dtype=int)
        needs_reassign = np.array([r in discarded for r in raw])
        for i, r in enumerate(raw):
            if not needs_reassign[i]:
                out[i] = remap[r]
        if needs_reassign.any():
            sims = cosine_similarity(embeddings[needs_reassign], self._kept_means)
            out[needs_reassign] = np.argmax(sims, axis=1)
        return out

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities for each component.

        Args:
            embeddings: numpy.ndarray of shape (N, D).

        Returns:
            numpy.ndarray of shape (N, n_topics) with probability values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba")

        proba = self._gmm.predict_proba(embeddings)
        if self._kept_component_indices is None:
            return proba
        # Restrict to kept components' columns and renormalize - points with
        # most of their mass on discarded components will look low-confidence
        # everywhere, which is an honest reflection of "this doesn't clearly
        # belong to any of the surviving topics" rather than silently forcing
        # a confident-looking distribution.
        kept = proba[:, self._kept_component_indices]
        row_sums = kept.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return kept / row_sums

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Mean vectors of each surviving GMM component.

        Returns:
            numpy.ndarray of shape (n_topics, D).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing cluster_centers_")

        if self._kept_means is not None:
            return self._kept_means
        return self._gmm.means_

    @property
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        return self._is_fitted