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
from katm.global_keyphrases import extract_global_keyphrases
from katm.keyphrase_extractor import _chunk_text_for_embedding
from katm.utils import _stopwords_for


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


def test_gmm_topic_clusterer_diag_covariance():
    # gmm_covariance_type="diag" - the recommended setting for the larger
    # anchor pools keyphrase_scope="global" tends to produce.
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(CORPUS)

    clusterer = GMMTopicClusterer(n_topics=N_TOPICS, covariance_type="diag")
    clusterer.fit(embeddings)

    assert clusterer.is_fitted
    assert clusterer.covariance_type == "diag"
    assert clusterer.cluster_centers_.shape == (N_TOPICS, embeddings.shape[1])


# ── keyphrase_scope="global" ─────────────────────────────────────────────


def test_katm_global_keyphrase_scope():
    model = KATM(
        n_topics=N_TOPICS,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
        keyphrase_scope="global",
        n_keyphrases_total=50,
    )
    model.fit(CORPUS)

    assert isinstance(model.topics_, dict)
    assert len(model.topics_) > 0
    assert len(model.doc_topic_probs_) == len(CORPUS)


def test_katm_global_keyphrase_scope_membership_ranking():
    model = KATM(
        n_topics=N_TOPICS,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
        keyphrase_scope="global",
        n_keyphrases_total=50,
        global_ranking_mode="membership",
        global_membership_n_per_doc=5,
        gmm_covariance_type="diag",
    )
    model.fit(CORPUS)

    assert isinstance(model.topics_, dict)
    assert len(model.doc_topic_probs_) == len(CORPUS)


def test_katm_global_keyphrase_scope_ignored_for_non_keybert():
    # keyphrase_scope="global" only applies to kp_algorithm="keybert" - other
    # algorithms should fit exactly as if keyphrase_scope had been left at
    # its "per_document" default, not raise or silently produce empty topics.
    model = KATM(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
        kp_algorithm="rake",
        keyphrase_scope="global",
    )
    model.fit(CORPUS)

    assert isinstance(model.topics_, dict)
    assert len(model.doc_topic_probs_) == len(CORPUS)


def test_extract_global_keyphrases_direct():
    embedder = SentenceEmbedder()
    keyphrases = extract_global_keyphrases(
        CORPUS, embedder, keybert_ngram_range=(1, 2), language="english",
        min_anchor_df=1, max_anchor_df_ratio=1.0, n_keyphrases_total=15,
    )
    assert isinstance(keyphrases, list)
    assert 0 < len(keyphrases) <= 15
    assert all(isinstance(kp, str) for kp in keyphrases)


def test_extract_global_keyphrases_unknown_ranking_mode():
    embedder = SentenceEmbedder()
    try:
        extract_global_keyphrases(
            CORPUS, embedder, keybert_ngram_range=(1, 2), language="english",
            min_anchor_df=1, max_anchor_df_ratio=1.0, n_keyphrases_total=15,
            ranking_mode="not_a_real_mode",
        )
        assert False, "expected ValueError for an unknown ranking_mode"
    except ValueError:
        pass


# ── exclusive_assignment ─────────────────────────────────────────────────


def test_katm_exclusive_assignment_no_cross_topic_duplicates():
    model = KATM(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
        exclusive_assignment=True,
    )
    model.fit(CORPUS)

    all_words = [w for words in model.topics_.values() for w, _ in words]
    assert len(all_words) == len(set(all_words))


# ── long_document_strategy="chunk" ───────────────────────────────────────

# A document well over any common embedding model's max_seq_length (256
# tokens for all-MiniLM-L6-v2), so long_document_strategy="chunk" actually
# has to split it rather than no-op.
LONG_DOCUMENT = (
    "The quantum computing chip broke new records in qubit stability "
    "and error correction rates. "
) * 60


def test_chunk_text_for_embedding_splits_long_document():
    embedder = SentenceEmbedder()
    tokenizer = embedder._model.tokenizer
    max_tokens = embedder._model.max_seq_length

    chunks = _chunk_text_for_embedding(LONG_DOCUMENT, tokenizer, max_tokens)
    assert len(chunks) > 1
    for chunk in chunks:
        n_tokens = len(tokenizer.encode(chunk, add_special_tokens=True, truncation=False))
        assert n_tokens <= max_tokens


def test_chunk_text_for_embedding_short_document_is_noop():
    embedder = SentenceEmbedder()
    tokenizer = embedder._model.tokenizer
    max_tokens = embedder._model.max_seq_length

    short_text = "A short sentence."
    assert _chunk_text_for_embedding(short_text, tokenizer, max_tokens) == [short_text]


def test_keyphrase_extractor_keybert_chunked_long_document():
    embedder = SentenceEmbedder()
    extractor = KeyphraseExtractor(
        algorithm="keybert", n_keyphrases=5, pretrained_model=embedder._model,
        long_document_strategy="chunk",
    )
    keyphrases = extractor.extract([LONG_DOCUMENT, CORPUS[0]])
    assert len(keyphrases) == 2
    for doc_kps in keyphrases:
        assert isinstance(doc_kps, list)
        assert len(doc_kps) > 0


def test_keyphrase_extractor_keybert_single_document_truncate_mode():
    # Regression test for a real KeyBERT quirk: extract_keywords() silently
    # unwraps its own return value for a batch of exactly one document
    # ([[(kw, score), ...]] -> [(kw, score), ...]); extract() must always
    # return one list per input document regardless of batch size.
    embedder = SentenceEmbedder()
    extractor = KeyphraseExtractor(
        algorithm="keybert", n_keyphrases=5, pretrained_model=embedder._model,
    )
    keyphrases = extractor.extract([CORPUS[0]])
    assert len(keyphrases) == 1
    assert isinstance(keyphrases[0], list)
    assert len(keyphrases[0]) > 0


def test_katm_long_document_strategy_chunk_end_to_end():
    corpus = CORPUS + [LONG_DOCUMENT]
    model = KATM(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
        long_document_strategy="chunk",
    )
    model.fit(corpus)

    assert isinstance(model.topics_, dict)
    assert len(model.doc_topic_probs_) == len(corpus)


# ── keybert_use_mmr ───────────────────────────────────────────────────────


def test_katm_keybert_use_mmr():
    model = KATM(
        n_topics=N_TOPICS,
        n_keyphrases=N_KEYPHRASES,
        top_n_words=TOP_N_WORDS,
        min_df=1,
        min_anchor_df=1,
        max_anchor_df_ratio=1.0,
        anchor_dedup_threshold=None,
        keybert_use_mmr=True,
        keybert_diversity=0.6,
    )
    model.fit(CORPUS)

    assert isinstance(model.topics_, dict)
    assert len(model.doc_topic_probs_) == len(CORPUS)


# ── language / stopwords ─────────────────────────────────────────────────


def test_stopwords_for_english():
    stop = _stopwords_for("english")
    assert isinstance(stop, set)
    assert "the" in stop and "and" in stop


def test_stopwords_for_unsupported_language_falls_back_to_english():
    # No NLTK stopword corpus exists for this code - should fall back to
    # the English list rather than raising.
    assert _stopwords_for("not-a-real-language") == _stopwords_for("english")