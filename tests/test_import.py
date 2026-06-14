import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import katm


def test_version():
    assert katm.__version__ == "0.1.0"


def test_all_public_symbols():
    assert hasattr(katm, "KATM")
    assert hasattr(katm, "KATMFast")
    assert hasattr(katm, "DocumentBuilder")
    assert hasattr(katm, "KeyphraseExtractor")
    assert hasattr(katm, "SentenceEmbedder")
    assert hasattr(katm, "GMMTopicClusterer")
    assert hasattr(katm, "WordTopicProjector")
    assert hasattr(katm, "WordTopicProjectorFast")
    assert hasattr(katm, "clean_text")
    assert hasattr(katm, "build_vocabulary")
    assert hasattr(katm, "extract_content_words")


def test_katm_fast_parameter():
    """fast=False returns standard KATM; fast=True returns KATMFast."""
    from katm.topic_model import KATM as _KATMBase

    m1 = katm.KATM(fast=False)
    assert type(m1) is _KATMBase

    m2 = katm.KATM(fast=True)
    assert type(m2) is katm.KATMFast