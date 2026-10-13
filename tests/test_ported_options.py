"""Tests for the options ported from katm_flake: clustering_method and the fast content-word methods.
Defaults must be unchanged (clustering_method="gmm", content_words_method="tfidf").

    python -m pytest tests/test_ported_options.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest

from katm.topic_model import KATM
from katm.utils import extract_content_words

DOCS = ([
    "The football team won the championship match after a late goal by the striker.",
    "The striker scored twice and the coach praised the defenders after the league game.",
    "Shares fell on the stock market as investors worried about company profits and inflation.",
    "The central bank raised interest rates to fight inflation, and bond markets reacted quickly.",
    "The new smartphone has a faster processor, a better camera and longer battery life.",
    "Software engineers released an update that fixes security bugs in the mobile operating system.",
] * 8)


def _fit(**kw):
    m = KATM(n_topics=3, n_keyphrases=5, min_df=2, min_anchor_df=2, compute_doc_topic_probs=True, **kw)
    return m, m.fit_transform(DOCS)


def test_default_equals_explicit_gmm():
    a, pa = _fit()
    b, pb = _fit(clustering_method="gmm")
    assert np.allclose(pa, pb)
    assert all(a.get_topic_words(t, n=10) == b.get_topic_words(t, n=10) for t in range(3))


def test_spherical_kmeans_runs():
    m, p = _fit(clustering_method="spherical_kmeans")
    assert p.shape == (len(DOCS), 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
    assert m.gmm_ is None


def test_bad_clustering_method():
    with pytest.raises(ValueError):
        KATM(clustering_method="nope")


@pytest.mark.parametrize("method", ["sampled_pos", "type_pos", "type_pos_templates"])
def test_fast_content_words(method):
    words = extract_content_words(DOCS, min_df=2, method=method, pos_filter="paper_restricted")
    assert "striker" in words and "inflation" in words
    assert not {"the", "and", "after"} & set(words)


def test_unknown_content_method():
    with pytest.raises(ValueError):
        extract_content_words(DOCS, method="nope")
