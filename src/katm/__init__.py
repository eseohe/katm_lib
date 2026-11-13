"""Keyphrase Anchored Topic Modeling (KATM) package."""

from .topic_model import KATM
from .document_builder import DocumentBuilder
from .keyphrase_extractor import KeyphraseExtractor
from .embedding import SentenceEmbedder
from .clustering import GMMTopicClusterer, select_n_topics
from .clustering_alt import AltTopicClusterer
from .topic_assignment import WordTopicProjector
from .utils import clean_text, build_vocabulary, extract_content_words, strip_email_addresses

__all__ = [
    "KATM",
    "DocumentBuilder",
    "KeyphraseExtractor",
    "SentenceEmbedder",
    "GMMTopicClusterer",
    "AltTopicClusterer",
    "select_n_topics",
    "WordTopicProjector",
    "clean_text",
    "build_vocabulary",
    "extract_content_words",
    "strip_email_addresses",
]

__version__ = "0.4.0"