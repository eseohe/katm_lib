"""Tests for the options synced from the katm_flake testbed: word_scoring,
keyphrase_scope="flake" (needs the optional `flake` library) and the clear
error when that library is missing.

    python -m pytest tests/test_flake_sync.py -q
"""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
_FLAKE_SRC = os.path.join(HERE, '..', '..', 'flake', 'src')   # sibling checkout, if present
if os.path.isdir(_FLAKE_SRC):
    sys.path.append(_FLAKE_SRC)

import numpy as np
import pytest

from katm.topic_model import KATM
from test_ported_options import DOCS


def _fit(**kw):
    m = KATM(n_topics=3, n_keyphrases=5, min_df=2, min_anchor_df=2, compute_doc_topic_probs=True,
             clustering_method="spherical_kmeans", **kw)
    return m, m.fit_transform(DOCS)


@pytest.mark.parametrize("scoring", ["similarity", "margin", "posterior"])
def test_word_scoring_runs_and_keeps_document_assignments(scoring):
    base, p0 = _fit()
    m, p = _fit(word_scoring=scoring)
    assert len(m.topics_) == 3 and all(len(w) > 0 for w in m.topics_.values())
    assert np.allclose(p0, p)   # word scoring only changes the displayed words


def test_bad_word_scoring():
    with pytest.raises(ValueError):
        _fit(word_scoring="nope")


def test_flake_scope_runs_when_flake_installed():
    pytest.importorskip("flake")
    m = KATM(n_topics=3, min_df=2, keyphrase_scope="flake", n_keyphrases_total=30,
             flake_kwargs={"min_df": 2, "max_df": 0.9}, clustering_method="spherical_kmeans")
    m.fit(DOCS)
    assert len(m.topics_) == 3


def test_flake_scope_without_flake_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "flake", None)
    monkeypatch.delitem(sys.modules, "flake.core", raising=False)
    m = KATM(n_topics=3, min_df=2, keyphrase_scope="flake", n_keyphrases_total=30)
    with pytest.raises(ImportError, match="flake"):
        m.fit(DOCS)
