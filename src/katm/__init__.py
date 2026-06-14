"""Keyphrase Anchored Topic Modeling (KATM) package."""

from .topic_model import KATM as _KATMBase
from .topic_model_fast import KATMFast
from .document_builder import DocumentBuilder
from .keyphrase_extractor import KeyphraseExtractor
from .embedding import SentenceEmbedder
from .clustering import GMMTopicClusterer
from .topic_assignment import WordTopicProjector
from .topic_assignment_fast import WordTopicProjectorFast
from .utils import clean_text, build_vocabulary, extract_content_words


class KATM:
    """Unified KATM entry point.

    Pass ``fast=True`` to instantiate :class:`KATMFast` (vectorised
    anchor dedup + incremental MMR). Pass ``fast=False`` (default) for
    the standard implementation. All other arguments are forwarded
    unchanged.

    Examples
    --------
    >>> from katm import KATM
    >>> model = KATM(n_topics=5)            # standard KATM
    >>> model = KATM(n_topics=5, fast=True) # KATMFast
    """

    def __new__(cls, *args, fast: bool = False, **kwargs):
        if fast:
            return KATMFast(*args, **kwargs)
        return _KATMBase(*args, **kwargs)


__all__ = [
    "KATM",
    "KATMFast",
    "DocumentBuilder",
    "KeyphraseExtractor",
    "SentenceEmbedder",
    "GMMTopicClusterer",
    "WordTopicProjector",
    "WordTopicProjectorFast",
    "clean_text",
    "build_vocabulary",
    "extract_content_words",
]

__version__ = "0.1.0"