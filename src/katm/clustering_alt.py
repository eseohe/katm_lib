"""Alternative anchor-clustering paradigms for KATM.

All expose the interface KATM uses from GMMTopicClusterer:
fit(X), predict(X), predict_proba(X), cluster_centers_, n_topics, is_fitted, _gmm.

methods
  "gmm"              GaussianMixture(diag), 1 init            (= original GMMTopicClusterer)
  "gmm_ninit"        GaussianMixture(diag), n_init inits (best likelihood kept)
  "gmm_hub"          original discard_hub_clusters: fit K+extra components, drop the
                     ones closest to the global mean (catch-all), reassign their points
  "spherical_kmeans" k-means on L2-normalised vectors (cosine geometry), n_init inits
  "overcluster_merge" spherical k-means with over_factor*K clusters, then the
                     centroids are merged to K groups by average-linkage on cosine
                     distance (size-weighted centroids) - lets a catch-all region
                     split first and recombine along real semantic boundaries
  "agglomerative"    Ward agglomerative clustering on L2-normalised vectors
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

METHODS = ("gmm", "gmm_ninit", "gmm_hub", "spherical_kmeans", "overcluster_merge", "agglomerative")


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


class AltTopicClusterer:
    def __init__(self, n_topics: int, method: str = "spherical_kmeans", random_state: int = 42,
                 n_init: int = 10, over_factor: int = 3, hub_extra: int = 5, softmax_temp: float = 0.05):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.n_topics, self.method, self.random_state = n_topics, method, random_state
        self.n_init, self.over_factor, self.hub_extra, self.softmax_temp = n_init, over_factor, hub_extra, softmax_temp
        self._gmm = None
        self._inner = None
        self._centers = None
        self._is_fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "AltTopicClusterer":
        X = np.asarray(X, dtype=np.float64)
        K = min(self.n_topics, len(X))
        m = self.method
        if m in ("gmm", "gmm_hub"):
            from .clustering import GMMTopicClusterer
            self._inner = GMMTopicClusterer(n_topics=K, random_state=self.random_state, covariance_type="diag",
                                            discard_hub_clusters=(m == "gmm_hub"), hub_overcluster_k=self.hub_extra)
            self._inner.fit(X)
            self._gmm = self._inner._gmm
            self._centers = self._inner.cluster_centers_
        elif m == "gmm_ninit":
            self._gmm = GaussianMixture(n_components=K, covariance_type="diag", random_state=self.random_state,
                                        max_iter=200, n_init=self.n_init).fit(X)
            self._centers = self._gmm.means_
        elif m == "spherical_kmeans":
            Xn = _norm(X)
            km = KMeans(n_clusters=K, n_init=self.n_init, random_state=self.random_state).fit(Xn)
            self._centers = _norm(km.cluster_centers_)
        elif m == "overcluster_merge":
            Xn = _norm(X)
            M = min(len(X), self.over_factor * K)
            km = KMeans(n_clusters=M, n_init=self.n_init, random_state=self.random_state).fit(Xn)
            C = _norm(km.cluster_centers_)
            sizes = np.bincount(km.labels_, minlength=M).astype(float)
            agg = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit(C)
            cents = np.stack([(C[agg.labels_ == g] * sizes[agg.labels_ == g, None]).sum(axis=0) for g in range(K)])
            self._centers = _norm(cents)
        elif m == "agglomerative":
            Xn = _norm(X)
            lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xn)
            self._centers = _norm(np.stack([Xn[lab == g].mean(axis=0) for g in range(K)]))
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, X):
        if self._inner is not None:
            return self._inner.predict(X)
        if self.method == "gmm_ninit":
            return self._gmm.predict(X)
        return (_norm(np.asarray(X)) @ self._centers.T).argmax(axis=1)

    def predict_proba(self, X):
        if self._inner is not None:
            return self._inner.predict_proba(X)
        if self.method == "gmm_ninit":
            return self._gmm.predict_proba(X)
        s = (_norm(np.asarray(X)) @ self._centers.T) / self.softmax_temp
        s = np.exp(s - s.max(axis=1, keepdims=True))
        return s / s.sum(axis=1, keepdims=True)

    @property
    def cluster_centers_(self):
        return self._centers

    @property
    def is_fitted(self):
        return self._is_fitted
