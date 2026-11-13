"""Tests for the proper-noun filter (proper_noun_filter and katm.proper_nouns).

    python -m pytest tests/test_proper_nouns.py -q
"""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

import pytest

from katm.proper_nouns import detect_proper_nouns, proper_noun_share, scrub_proper_nouns
from katm.topic_model import KATM

# Names appear capitalised mid-sentence; "wand"/"castle" appear capitalised only at sentence start.
FANDOM = [
    "I think Harry trusted Hermione more than Ron did and asked Dumbledore. The castle was cold. Wand work was hard.",
    "Then Hermione told Harry about the wand. Dumbledore smiled at Ron and Harry. Castle rules apply.",
    "Later Ron and Hermione read about Dumbledore. Harry kept the wand close in the castle.",
    "People say Dumbledore was wise. Ron saw Hermione and Harry leave the castle with a wand.",
]


def test_detect_capitalization_finds_names_not_common_words():
    names = detect_proper_nouns(FANDOM, min_count=3)
    assert {"harry", "hermione", "ron", "dumbledore"} <= names
    assert "wand" not in names and "castle" not in names


def test_scrub_removes_names_and_possessives_keeps_other_words():
    names = {"harry", "hermione"}
    out = scrub_proper_nouns("Harry's wand and Hermione read the castle book.", names)
    assert "harry" not in out.lower() and "hermione" not in out.lower()
    assert "wand" in out and "castle" in out


def test_scrub_handles_plural_stem():
    assert "dumbledores" not in scrub_proper_nouns("the Dumbledores came", {"dumbledore"}).lower()


def test_proper_noun_share_orders_corpora():
    names = detect_proper_nouns(FANDOM, min_count=3)
    plain = ["the wand and the castle were cold today and the books were old"]
    assert proper_noun_share(FANDOM, names) > proper_noun_share(plain, names)


def test_bad_method_raises():
    with pytest.raises(ValueError):
        detect_proper_nouns(FANDOM, method="nope")


def test_bad_filter_value_raises():
    with pytest.raises(ValueError):
        KATM(proper_noun_filter="sometimes")


def test_streaming_not_supported():
    with pytest.raises(NotImplementedError):
        KATM(streaming=True, proper_noun_filter=True).fit(FANDOM)


def test_katm_fit_removes_names_from_topics():
    docs = FANDOM * 4
    m = KATM(n_topics=2, n_keyphrases=4, min_df=2, min_anchor_df=2, max_anchor_df_ratio=1.0,
             anchor_dedup_threshold=None, proper_noun_filter=True, proper_noun_min_count=3,
             proper_nouns_keep=["ron"])
    m.fit(docs)
    assert {"harry", "hermione", "dumbledore"} <= m.proper_nouns_
    assert "ron" not in m.proper_nouns_            # kept by request
    assert m.proper_noun_report_["applied"] is True
    shown = {w.lower() for ws in m.topics_.values() for w in ([x if isinstance(x, str) else x[0] for x in ws])}
    assert not shown & {"harry", "hermione", "dumbledore"}


def test_auto_mode_skips_when_few_names():
    plain = ["the old library holds many books about medieval farming and tools",
             "farming tools were made from iron and wood in the village",
             "the village market sold bread cheese and wool every week",
             "wool and cheese were traded at the market for iron tools",
             "books on bread making are kept in the library",
             "the library opens early and the village market opens late"] * 2
    m = KATM(n_topics=2, n_keyphrases=3, min_df=1, min_anchor_df=1, max_anchor_df_ratio=1.0,
             anchor_dedup_threshold=None, proper_noun_filter="auto")
    m.fit(plain)
    assert m.proper_noun_report_["applied"] is False and m.proper_nouns_ is None


def test_spacy_method_english():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    names = detect_proper_nouns(FANDOM * 3, language="english", method="spacy", min_count=3)
    assert {"harry", "hermione"} <= names
    assert "wand" not in names and "castle" not in names


def _en_nlp_or_skip():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")


NER_DOCS = [
    "Barack Obama visited Paris on Monday and met Angela Merkel.",
    "Angela Merkel flew to Paris to meet Barack Obama at Google.",
    "Google announced a deal after Barack Obama spoke in Paris.",
    "In Paris, Angela Merkel and Barack Obama thanked Google.",
] * 3


def test_ner_person_only_keeps_places_and_orgs():
    _en_nlp_or_skip()
    people = detect_proper_nouns(NER_DOCS, method="ner", ner_labels="person", min_count=3)
    assert {"obama", "merkel"} <= people
    assert "paris" not in people and "google" not in people


def test_ner_combination_of_labels():
    _en_nlp_or_skip()
    names = detect_proper_nouns(NER_DOCS, method="ner", ner_labels=["PERSON", "gpe", "ORG"], min_count=3)
    assert {"obama", "merkel", "paris", "google"} <= names


def test_ner_unknown_label_raises():
    _en_nlp_or_skip()
    with pytest.raises(ValueError):
        detect_proper_nouns(NER_DOCS, method="ner", ner_labels=["PERSONN"])


def test_ner_empty_labels_raises():
    with pytest.raises(ValueError):
        KATM(proper_noun_filter=True, proper_noun_method="ner", proper_noun_ner_labels=[])


def test_bad_method_rejected_at_construction():
    with pytest.raises(ValueError):
        KATM(proper_noun_method="nope")


def test_katm_fit_with_ner_people_only():
    _en_nlp_or_skip()
    docs = NER_DOCS * 2
    m = KATM(n_topics=2, n_keyphrases=4, min_df=2, min_anchor_df=2, max_anchor_df_ratio=1.0,
             anchor_dedup_threshold=None, proper_noun_filter=True,
             proper_noun_method="ner", proper_noun_ner_labels="PERSON")
    m.fit(docs)
    assert {"obama", "merkel"} <= m.proper_nouns_ and "paris" not in m.proper_nouns_
