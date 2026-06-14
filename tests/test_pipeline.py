import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import katm
from katm import (
    KATM,
    KATMFast,
    DocumentBuilder,
    KeyphraseExtractor,
    SentenceEmbedder,
    GMMTopicClusterer,
)


# Synthetic corpus: 5 documents on distinct topics.
# NOTE: TF-IDF keyphrase extractor uses min_df=2 internally (min(2, n_docs)),
# so at least some words must appear in 2+ documents. We ensure this by adding
# a common "report" word to all docs so TF-IDF keyphrase extraction succeeds.
CORPUS = [
    (
        "The solar system contains eight planets orbiting the sun. "
        "Mercury is the closest planet, while Neptune is the farthest. "
        "Earth is the only known planet with life and has one natural satellite, the moon. "
        "This report covers the basic facts about our solar system."
    ),
    (
        "Football is the most popular sport in the world. "
        "Players score goals by getting the ball into the opposing team's net. "
        "The FIFA World Cup is held every four years and attracts billions of viewers. "
        "This report discusses football and its global importance."
    ),
    (
        "Cooking pasta correctly requires boiling water with plenty of salt. "
        "Spaghetti should be cooked until al dente, which means firm to the bite. "
        "Italian cuisine emphasizes fresh ingredients and simple preparation methods. "
        "This report explores traditional Italian cooking techniques."
    ),
    (
        "Democratic elections allow citizens to vote for their preferred candidates. "
        "Voting rights have been fought for throughout history by many activist movements. "
        "Representatives in parliament discuss and pass laws on behalf of the people. "
        "This report examines democratic political systems around the world."
    ),
    (
        "Artificial intelligence enables computers to learn from data and make decisions. "
        "Machine learning algorithms power recommendation systems and image recognition. "
        "Large language models are trained on massive text corpora to generate human-like text. "
        "This report surveys the current state of artificial intelligence research."
    ),
]

N_TOPICS = 3
N_KEYPHRASES = 5
TOP_N_WORDS = 10


def test_katm_fit():
    # Use relaxed anchor params for this tiny corpus:
    # min_anchor_df=1 allows hapax anchors, anchor_dedup_threshold=None skips
    # semantic dedup (skipped on 5 docs), and max_anchor_df_ratio=1.0 keeps all.
    model = KATM(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
    )
    model.fit(CORPUS)

    # topics_ must be a dict with at least some topic entries
    assert model.topics_ is not None
    assert isinstance(model.topics_, dict)
    # With 5 very different documents and only 3 topics, we expect topics to be discovered
    assert len(model.topics_) > 0, "topics_ should not be empty"

    # doc_topic_probs_ must be a list with one entry per document
    assert model.doc_topic_probs_ is not None
    assert isinstance(model.doc_topic_probs_, list)
    assert len(model.doc_topic_probs_) == len(CORPUS)


def test_katm_transform():
    model = KATM(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
    )
    model.fit(CORPUS)

    result = model.transform(CORPUS)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(CORPUS), N_TOPICS)


def test_katm_fast_fit():
    model_fast = KATMFast(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
    )
    model_fast.fit(CORPUS)

    assert model_fast.topics_ is not None
    assert isinstance(model_fast.topics_, dict)
    assert len(model_fast.topics_) > 0

    assert model_fast.doc_topic_probs_ is not None
    assert isinstance(model_fast.doc_topic_probs_, list)
    assert len(model_fast.doc_topic_probs_) == len(CORPUS)


def test_document_builder():
    builder = DocumentBuilder(strategy="paragraph_group", chunk_size=2)
    docs = builder.build(CORPUS)
    assert isinstance(docs, list)
    # paragraph_group with chunk_size=2 groups 5 single-paragraph docs into ceil(5/2)=3 chunks
    assert len(docs) == 3
    for doc in docs:
        assert isinstance(doc, str)
        assert len(doc) > 0


def test_keyphrase_extractor_rake():
    extractor = KeyphraseExtractor(algorithm="rake", n_keyphrases=N_KEYPHRASES)
    keyphrases = extractor.extract(CORPUS)
    assert isinstance(keyphrases, list)
    assert len(keyphrases) == len(CORPUS)
    for doc_kps in keyphrases:
        assert isinstance(doc_kps, list)
        # Each document should produce some keyphrases
        assert len(doc_kps) > 0


def test_keyphrase_extractor_tfidf():
    # Use unigrams only (ngram_range=(1,1)) so that in a tiny 5-doc corpus
    # each term appears in multiple docs and survives min_df=2 pruning
    extractor = KeyphraseExtractor(
        algorithm="tfidf",
        n_keyphrases=N_KEYPHRASES,
        tfidf_ngram_range=(1, 1),
    )
    keyphrases = extractor.extract(CORPUS)
    assert isinstance(keyphrases, list)
    assert len(keyphrases) == len(CORPUS)
    for doc_kps in keyphrases:
        assert isinstance(doc_kps, list)
        assert len(doc_kps) > 0


def test_sentence_embedder_encode():
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(CORPUS)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(CORPUS)
    assert embeddings.shape[1] > 0  # embedding dimension


def test_gmm_topic_clusterer():
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(CORPUS)

    clusterer = GMMTopicClusterer(n_topics=N_TOPICS)
    clusterer.fit(embeddings)

    assert clusterer.is_fitted
    assert clusterer.cluster_centers_.shape == (N_TOPICS, embeddings.shape[1])