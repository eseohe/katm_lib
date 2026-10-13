"""Shared dimensionality-reduction helpers for KATM's optional
dim_reduction parameter. Used by both topic_model.py (KATM) and
topic_model_global_exp.py (KATMGlobal) so the two never carry independent,
driftable copies of the same logic.

"umap" was implemented and tested here first, matching BERTopic/SDTM's own
settings (n_neighbors=15, min_dist=0.0, cosine metric) - and removed after
producing degenerate topics (mean coherence 0.0000, most topics reduced to
a single near-random word) on every dataset tested (mental-health Reddit,
TweetEval, 20 Newsgroups; see sdtm/KATM_GLOBAL_REPORT.md). The cause is
architectural: UMAP's out-of-sample .transform() has to place thousands of
content-word vocabulary items - a completely different distribution from
the few hundred keyphrase anchors it was fit on - into the same reduced
space, and does not preserve meaningful neighborhood structure for points
that different from its training set. PCA doesn't have this failure mode
because a linear projection generalizes to new points by construction.
"""
from typing import List, Optional

import numpy as np

from .embedding import SentenceEmbedder


class ReducedEmbedder:
    """Wraps a SentenceEmbedder + a fitted dimensionality reducer behind the
    same .encode() interface SentenceEmbedder exposes, so WordTopicProjector
    (which only ever calls self.embedder.encode(...)) can be reused as-is
    whether or not dim_reduction is active. The reducer is fit exactly once,
    on the anchor embeddings - every other call only ever transforms."""

    def __init__(self, embedder: SentenceEmbedder, reducer=None):
        self._embedder = embedder
        self._reducer = reducer

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        emb = self._embedder.encode(texts, batch_size=batch_size)
        if self._reducer is None or len(emb) == 0:
            return emb
        return self._reducer.transform(emb)


def fit_reducer(method: str, embeddings: np.ndarray, n_components: Optional[int], random_state: int):
    n = len(embeddings)
    dim = embeddings.shape[1] if n else 0
    if method == "pca":
        from sklearn.decomposition import PCA
        k = n_components or min(50, n - 1, dim)
        k = max(2, min(k, n - 1, dim))
        reducer = PCA(n_components=k, random_state=random_state)
        reducer.fit(embeddings)
        return reducer
    raise ValueError(f"dim_reduction must be None or 'pca' - got {method!r}")
