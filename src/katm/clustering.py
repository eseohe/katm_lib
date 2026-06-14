"""Clustering module for KATM using Gaussian Mixture Models."""

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMTopicClusterer:
    """Clusters embeddings using Gaussian Mixture Model."""

    def __init__(
        self,
        n_topics: int,
        soft_threshold: float = 0.15,
        random_state: int = 42,
    ):
        """Initialize GMMTopicClusterer.

        Args:
            n_topics: Number of GMM components (topics).
            soft_threshold: Not currently used, kept for API compatibility.
            random_state: Random seed for reproducibility.
        """
        self.n_topics = n_topics
        self.soft_threshold = soft_threshold
        self.random_state = random_state
        self._gmm = None
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray) -> "GMMTopicClusterer":
        """Fit the GMM to embeddings.

        Args:
            embeddings: numpy.ndarray of shape (N, D).

        Returns:
            self for chaining.
        """
        self._gmm = GaussianMixture(
            n_components=self.n_topics,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200,
        )
        self._gmm.fit(embeddings)
        self._is_fitted = True
        return self

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities for each component.

        Args:
            embeddings: numpy.ndarray of shape (N, D).

        Returns:
            numpy.ndarray of shape (N, n_topics) with probability values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba")

        return self._gmm.predict_proba(embeddings)

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Mean vectors of each GMM component.

        Returns:
            numpy.ndarray of shape (n_topics, D).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing cluster_centers_")

        return self._gmm.means_

    @property
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        return self._is_fitted