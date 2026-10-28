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
    covariance_type: str = "full",
) -> Tuple[int, Dict[str, list]]:
    """Automatically choose the number of GMM components (topics).

    Fits a GMM at every k in [k_min, k_max] (inclusive) on `embeddings` and
    picks the best k according to `method`:

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

    Fitting one GMM per k in the range is O((k_max - k_min) x GMM-fit-cost);
    for the anchor-embedding counts KATM works with (hundreds to a few
    thousand keyphrases after dedup) this is fast enough to run inline
    during fit() and is not a separate offline step.

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
    emerge. This only affects the k-selection scan; GMMTopicClusterer's own
    covariance_type for the final production fit is unchanged ("full").

    Args:
        embeddings: numpy.ndarray of shape (N, D) — the anchor/keyphrase
            embeddings that will be clustered (i.e. call this on the same
            array passed to GMMTopicClusterer.fit).
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


def anchor_isolation_scores(embeddings: np.ndarray, k_neighbors: int = 10, method: str = "concentration") -> np.ndarray:
    """Per-anchor "does this belong to any lexical field at all" score
    (exploratory). Two different, competing hypotheses
    about what a catch-all anchor looks like in embedding space - see
    `method` below; higher always means "safer to cluster", lower always
    means "candidate to drop" (filter_isolated_anchors), regardless of
    method.

    Diagnosis this targets: the clustering-init experiments
    (coverage_seed_embeddings, above) kept landing one of K seeds on a
    thematically-generic anchor - ag_news's Sci/Tech slot collapsed to
    "name, boss, scientist, body, movie, father, clean, patient, character,
    answer" under coverage seeding, and the ORIGINAL (pre-FLaKE) partition
    diagnosis found the same thing under plain GMM: a catch-all component
    absorbing 400-517 anchors with centroid words like "computer, movie,
    business, phone, something, name". Both are symptoms of the same
    upstream fact: FLaKE's anchor-selection objective (document coverage)
    and "is this a good GMM/k-means point" are different questions.

    Args:
        embeddings: (n_anchors, D) anchor embeddings (any order).
        k_neighbors: how many nearest other anchors to average over. Default
            10 - not independently tuned, sweep per corpus.
        method:
          - "density" (the first thing tried): mean cosine similarity to the
            anchor's own k_neighbors nearest OTHER anchors. Hypothesis: a
            genuine topic word has many close relatives sharing its lexical
            field (e.g. "championship" near "tournament, league, goal,
            opponent, ...") so it scores high; a catch-all word has no such
            neighborhood anywhere in the pool, so it scores low.
            MEASURED AND REJECTED on 20NewsGroup's actual catch-all topic
            (body, water, matter, brain, material, science, smoke, light,
            money, black): these words' "density" scores landed at the
            30th-74th percentile of the whole 1000-anchor pool, overlapping
            heavily with genuine sport-topic words (51st-92nd percentile) -
            not reliably separated, and dropping the bottom 30% by this
            score on the real pipeline left the same flavor of catch-all
            topic standing, with NMI flat-to-slightly-down.
          - "concentration" (default - the fix for what "density" got
            backwards): top_k_neighbor_mean MINUS mean-similarity-to-the-
            REST of the pool, i.e. how much more similar an anchor is to its
            own best friends than to a random/typical other anchor. A
            genuine topic word is similar to its own ~100-200-anchor lexical
            field and comparatively unrelated to the other 800+ anchors
            (big gap = sharply peaked profile = high concentration). A
            catch-all word turns out to have a nearly FLAT similarity
            profile instead - its "nearest neighbors" are barely more
            similar than a typical anchor, because it has diffuse, roughly
            uniform affinity spread across many unrelated topics rather
            than one specific home (small/negative gap = low concentration)
            - this is the "weirdly close to everything" signal: not
            uniquely isolated, and not uniquely hub-close to everything in
            absolute terms either (plain mean-similarity-to-all-anchors and
            similarity-to-the-pool-centroid were ALSO tried and tested
            directly against the same catch-all-vs-clean word sets; both
            put catch-all words at the 93rd percentile but clean words
            right behind at the 87th - correctly directional but too small
            a gap, ~6 points, to be useful alone). VALIDATED directly on the
            same 20NewsGroup catch-all-vs-clean word sets: catch-all words
            averaged the 29th percentile on concentration, clean sport words
            the 50th - a 20-point gap, the clearest separation of every
            signal tried.

    Returns:
        (n_anchors,) float array, one score per row of `embeddings`.
    """
    if method not in ("density", "concentration"):
        raise ValueError("method must be 'density' or 'concentration'")
    Xn = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    sims = Xn @ Xn.T
    n = len(embeddings)
    np.fill_diagonal(sims, np.nan)
    k = max(1, min(k_neighbors, n - 1)) if n > 1 else 0
    if k <= 0:
        return np.zeros(n)
    top_k = np.partition(np.nan_to_num(sims, nan=-1.0), n - k, axis=1)[:, n - k:].mean(axis=1)
    if method == "density":
        return top_k
    mean_all = np.nanmean(sims, axis=1)  # similarity to the REST of the pool (excludes self)
    return top_k - mean_all


def filter_isolated_anchors(
    embeddings: np.ndarray, k_neighbors: int = 10, drop_frac: float = 0.1, method: str = "concentration",
) -> np.ndarray:
    """Boolean keep-mask (True = keep) dropping the `drop_frac` fraction of
    anchors with the lowest anchor_isolation_scores (see its own docstring
    for the two methods and which one was actually validated).
    Exploratory.

    drop_frac=0 (any falsy value) is a no-op: returns an all-True mask, so
    this is always safe to call unconditionally and only changes anything
    when actually requested with drop_frac > 0.

    Args:
        embeddings: (n_anchors, D) anchor embeddings.
        k_neighbors: passed to anchor_isolation_scores.
        drop_frac: fraction in [0, 1) of the LOWEST-scoring anchors to drop.
            Not independently tuned - sweep per corpus; dropping too much
            risks thinning out a real but genuinely small/niche topic's own
            anchor pool along with the actual catch-all words (a small
            theme's anchors are still each other's closest neighbors, just
            fewer of them, so a very aggressive drop_frac could start
            hollowing out real minority themes too, not only noise).
        method: "density" or "concentration" (default) - see
            anchor_isolation_scores's docstring.

    Returns:
        (n_anchors,) bool array.
    """
    n = len(embeddings)
    if not drop_frac or n == 0:
        return np.ones(n, dtype=bool)
    n_drop = int(round(drop_frac * n))
    if n_drop <= 0:
        return np.ones(n, dtype=bool)
    scores = anchor_isolation_scores(embeddings, k_neighbors, method=method)
    drop_idx = np.argsort(scores)[:n_drop]
    mask = np.ones(n, dtype=bool)
    mask[drop_idx] = False
    return mask


def _farthest_point_subset(X: np.ndarray, k: int, start: int = 0) -> list:
    """Deterministic k-means++-style farthest-point sampling (max-min cosine
    distance), no randomness. Used by coverage_seed_embeddings's over_factor
    reduction - see that function's docstring for why."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    n = len(X)
    k = min(k, n)
    chosen = [start]
    min_sim = Xn @ Xn[start]  # cosine sim to nearest chosen so far
    for _ in range(k - 1):
        nxt = int(np.argmin(min_sim))
        chosen.append(nxt)
        min_sim = np.minimum(min_sim, Xn @ Xn[nxt])
    return chosen


def coverage_seed_embeddings(
    relevance, embeddings: np.ndarray, k: int, objective: str = "facility",
    over_factor: int = 1,
) -> Optional[np.ndarray]:
    """Coverage-aware seeds for GMM/k-means initialization (
    exploratory — see clustering_compare's "the real lever is the clustering
    step" follow-up and the semantic-relevance-graph exploration that
    preceded this).

    The diagnosed failure mode (FLaKE experiments, "Follow-up: does
    splitting the corpus before FLaKE help weak topics?"): GMM/k-means's own
    init (kmeans++ in embedding space) sometimes merges two or three
    genuinely distinct document themes into one catch-all component, purely
    because their anchors happen to sit close together in embedding space -
    one AG seed put 400 of 1,000 anchors (tech + drug + court + oil + gold +
    airline) in a single component. That failure is about which DOCUMENTS
    each cluster ends up representing, not about embedding geometry - so an
    init that is chosen by document coverage, rather than embedding
    distance, should be less likely to start two seeds in the same
    document-theme and starve a third.

    Mechanism: FLaKE's own anchor selection already built, and KATM already
    has in hand, the sparse (docs x anchors) relevance graph used to pick
    the anchor pool (R[d, c] = how well anchor c represents document d's own
    content). Reselecting from that SAME graph at budget k (instead of the
    original ~500-2000) with the same facility-location objective
    (sum_d max_{c in seeds} R[d, c]) picks k anchors that are maximally
    complementary in DOCUMENT COVERAGE - each successive pick is the anchor
    that best serves documents the previous picks under-serve. Embeddings of
    those k anchors become the initial means (GMM) / centroids (k-means);
    the normal EM / Lloyd iterations run unchanged afterward - this only
    changes where they start, not the objective they optimize.

    This is reusing flake.core.select_facility_location
    (already generic over any non-negative docs x candidates matrix) rather
    than reimplementing greedy coverage selection - zero new selection
    logic, and the relevance graph is already computed as a side effect of
    FLaKE's own anchor selection, so this costs one extra lazy-greedy pass
    over an already-tiny sparse matrix (milliseconds - see README's
    "Clustering time is under 1 s for every method").

    Args:
        relevance: sparse (n_docs, n_anchors) matrix, columns in the SAME
            order as `embeddings`'s rows (e.g. FLaKE's R[:, selected_indices]).
            None (no coverage graph available - e.g. keyphrase_scope is not
            "flake", or FLaKE ran partitioned) means this function is a
            no-op and returns None, so callers can always try this and fall
            back to the clusterer's own default init.
        embeddings: (n_anchors, D) anchor embeddings, row-aligned with
            `relevance`'s columns.
        k: number of seeds to pick (the clusterer's actual n_components -
            for GMMTopicClusterer with discard_hub_clusters=True, that's
            n_topics + hub_overcluster_k, not n_topics).
        objective: passed straight through to select_facility_location
            ("facility", the default, or "concave").
        over_factor: 1 (default, unchanged behavior) picks exactly k seeds by
            coverage alone. An int > 1 instead facility-location-selects
            over_factor*k candidate anchors first, then reduces to k by
            deterministic farthest-point sampling (max-min cosine distance)
            IN EMBEDDING SPACE among just those candidates.

            Why: plain coverage selection (over_factor=1) was found to
            sometimes pick a seed that is itself a poor topic representative
            - not a near-duplicate of another seed, but a single anchor that
            happens to have moderate relevance to the largest number of
            still-uncovered documents precisely because it's thematically
            generic (the same "broad-but-moderate relevance beats narrow-
            but-strong relevance" failure already diagnosed for top-k-
            average ranking in global_keyphrases.py's module docstring -
            raw total/incremental coverage doesn't distinguish "this anchor
            is the one true center of a real theme" from "this anchor is
            weakly related to many documents spanning several themes").
            Measured directly on ag_news (ambiguous/overlapping anchors
            concentrated in Sci/Tech, matching the exact catch-all failure
            already diagnosed in README's partition-diagnosis section): with
            over_factor=1, one of K=4 coverage seeds for spherical_kmeans
            landed on a generic catch-all anchor and the resulting topic
            read "name, boss, scientist, body, movie, father, clean,
            patient, character, answer" (NMI 0.495 vs the default init's
            0.543 - a regression, not an improvement). Over-selecting a
            richer candidate pool first and then choosing the final k by
            embedding-space spread gives a bad-but-high-coverage candidate
            room to be passed over in favor of a more distinctive
            alternative that still has decent (if not maximal) coverage.
            Not yet validated as a net win - exploratory, opt-in, sweep it
            per corpus rather than assuming a value.

    Returns:
        (k, D) ndarray of seed embeddings, or None if `relevance` is None or
        k <= 0. select_facility_location always fills to exactly k seeds
        (padding by total relevance if coverage saturates before k picks -
        see its own docstring), so the caller never has to handle a
        short array.
    """
    if relevance is None or k <= 0:
        return None
    import os as _os, sys as _sys
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from ._flake import flake_core
    select_facility_location = flake_core().select_facility_location

    n_anchors = embeddings.shape[0]
    k = min(k, n_anchors)
    pool_k = min(max(k * max(1, over_factor), k), n_anchors)
    idx, _gains = select_facility_location(relevance, pool_k, objective=objective)
    if not idx:
        return None
    if over_factor <= 1 or pool_k <= k:
        return embeddings[idx[:k]]
    pool_emb = embeddings[idx]
    local = _farthest_point_subset(pool_emb, k)
    return pool_emb[local]


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
        means_init: Optional[np.ndarray] = None,
    ):
        """Initialize GMMTopicClusterer.

        Args:
            n_topics: Number of GMM components (topics) in the FINAL output -
                unchanged meaning regardless of discard_hub_clusters below;
                that flag changes how many components are fit internally,
                not how many topics come out.
            soft_threshold: Not currently used, kept for API compatibility.
            random_state: Random seed for reproducibility.
            covariance_type: sklearn GaussianMixture covariance type.
                Default "full". "full" gives each component D(D+1)/2 free
                covariance parameters, which can be wildly underdetermined
                against a modest anchor pool when D is large (e.g. D=1024);
                "diag" (D parameters/component) is far better conditioned in
                that regime and worth trying (it is what the alternative
                clustering methods and n_topics="auto" scan use).
            discard_hub_clusters: False (default, unchanged behavior - fits
                exactly n_topics components). If True, fits
                n_topics + hub_overcluster_k components instead, then keeps
                only the n_topics whose centroid sits FURTHEST (lowest
                cosine similarity) from the global mean of the fitted
                embeddings, discarding the rest.

                Motivated by a direct empirical finding (see project notes:
                gemini/subtopic_discovery's "T12 filler topic" investigation
                on The Thirty-Nine Steps and The Sign of Four): with a fixed
                n_topics, GMM is sometimes forced to spend one component on
                a catch-all of leftover, individually-generic anchors
                (thing, way, good, course...) that have nowhere thematic to
                go - that component's centroid was measured to sit at
                cosine similarity 0.91 to the whole anchor pool's global
                mean (vs. 0.47-0.84 for every genuinely thematic cluster in
                the same run), and the pattern replicated on a second novel
                (highest-hubness cluster = filler both times; lowest-hubness
                clusters were reliably among the most distinctive topics,
                e.g. a coastal cluster at 0.47). A component whose centroid
                is unusually close to "the average of everything" isn't
                really a topic - it's the generic residue every fixed-k
                clustering has nowhere else to put. Overclustering first
                gives that residue its own component(s) to absorb into
                WITHOUT taking one of the n_topics real slots, and discarding
                by global-mean proximity afterward removes it while still
                delivering exactly n_topics topics.

                Anchors originally assigned to a discarded component are
                reassigned to their nearest surviving centroid (by cosine
                similarity) rather than dropped, so every anchor still
                counts toward some topic's centroid/tightness - see fit()'s
                mechanism.

                Not yet validated as a net improvement on real topic
                quality (only the discovery signal itself was validated,
                two novels) - opt-in and exploratory.
            hub_overcluster_k: Only used when discard_hub_clusters=True.
                How many extra components to fit beyond n_topics before
                discarding the highest-hubness ones back down to n_topics.
                Default 5 - a starting guess, not independently validated;
                expose as a parameter specifically so it can be swept
                per-corpus (a large k gives the residue more room to spread
                into multiple components and get discarded cleanly, at the
                cost of a bigger/slower GMM fit and less certainty about
                which of the extra components are actually "real" topics
                being wrongly discarded rather than residue).
            means_init: None (default, unchanged behavior - sklearn's own
                "kmeans" init_params). Optional (n_components, D) array of
                initial component means, e.g. from coverage_seed_embeddings
                above - exploratory. Passed straight to
                GaussianMixture(means_init=...); sklearn then only uses its
                own init_params to seed weights/precisions, not means. Must
                have exactly n_components rows (n_topics, or
                n_topics + hub_overcluster_k when discard_hub_clusters=True)
                - a mismatched shape is treated as "no seed available" and
                silently falls back to default init, rather than raising,
                since the caller (KATM.fit) may not always know in advance
                whether a coverage graph was available at the right size.
        """
        self.n_topics = n_topics
        self.soft_threshold = soft_threshold
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.discard_hub_clusters = discard_hub_clusters
        self.hub_overcluster_k = hub_overcluster_k
        self.means_init = means_init
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

        means_init = self.means_init
        if means_init is not None and (len(means_init) != n_components or means_init.shape[1] != embeddings.shape[1]):
            # Shape mismatch (e.g. caller sized seeds for n_topics but
            # discard_hub_clusters bumped n_components) - fall back to
            # default init rather than letting GaussianMixture raise.
            means_init = None

        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=200,
            means_init=means_init,
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
            self.discarded_component_indices_ = np.sort(order[self.n_topics :])
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
            numpy.ndarray of shape (N,), values in [0, n_topics).
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