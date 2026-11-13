"""Proper-noun (name) detection and scrubbing for corpora dominated by names.

On fandom, fiction and news text, KeyBERT anchors and projected topic words
are often just character or place names, so every topic reads as a list of the
same names. This module finds the corpus's proper nouns from the text itself
(no hand-made list needed) and removes them from the documents before
keyphrase extraction and vocabulary building.

Two detectors:
  "capitalization" - a word is a name if it is capitalised in >= min_ratio of
      its mid-sentence occurrences (counting lower-case uses against it).
      Language-neutral for English; useless for German, where every noun is
      capitalised.
  "spacy" - a word is a name if spaCy tags it PROPN in >= min_ratio of its
      occurrences. Works for English too, and is the choice for German/French;
      noisier on lower-case web text.
  "ner" - a word is a name if spaCy's named-entity recogniser labels it with
      one of `ner_labels` (e.g. PERSON, GPE, ORG) in >= min_ratio of its
      occurrences. Lets you remove only some kinds of names (people but not
      places, say). Labels are those of the spaCy model, e.g. for English
      en_core_web_sm: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT,
      WORK_OF_ART, LAW, LANGUAGE.
"""
import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set, Union

DEFAULT_NER_LABELS = ("PERSON", "GPE", "ORG")
METHODS = ("capitalization", "spacy", "ner")

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# Capitalised word that does not start a sentence/line (prev char is not . ! ? or a newline).
_MID_CAP = re.compile(r"(?<![.!?\n\"“‘'(])\s([A-Z][a-z]{2,})\b")
_POSS = re.compile(r"['’]s\b")


def _stop(language: str) -> Set[str]:
    from .utils import _stopwords_for
    return _stopwords_for(language)


def detect_proper_nouns(documents: Iterable[str], language: str = "english",
                        method: str = "capitalization", min_count: int = 3,
                        min_ratio: float = 0.8,
                        ner_labels: Union[str, Sequence[str]] = DEFAULT_NER_LABELS) -> Set[str]:
    """Return the set of lower-cased proper-noun forms found in `documents`.

    `ner_labels` (a label or a list of labels, case-insensitive) is only used
    when method="ner"."""
    if method not in METHODS:
        raise ValueError("proper_noun_method must be one of %s" % (METHODS,))
    stop = _stop(language)
    docs = list(documents)
    if method == "capitalization":
        cap, low = Counter(), Counter()
        for d in docs:
            for m in _MID_CAP.finditer(d):
                cap[m.group(1).lower()] += 1
            for t in _WORD.findall(d):
                if t.islower():
                    low[t] += 1
        names = {w for w, c in cap.items()
                 if c >= min_count and c / (c + low[w]) >= min_ratio and w not in stop}
    elif method == "spacy":
        import spacy
        from .utils import _SPACY_MODEL
        nlp = spacy.load(_SPACY_MODEL.get(language, "en_core_web_sm"),
                         disable=["parser", "ner", "senter"])
        pn, tot = Counter(), Counter()
        for doc in nlp.pipe(docs, batch_size=64):
            for t in doc:
                if t.is_alpha and len(t) > 1:
                    w = t.text.lower()
                    tot[w] += 1
                    if t.pos_ == "PROPN":
                        pn[w] += 1
        names = {w for w, c in pn.items()
                 if c >= min_count and c / tot[w] >= min_ratio and w not in stop}
    else:  # "ner"
        import spacy
        from .utils import _SPACY_MODEL
        labels = normalize_ner_labels(ner_labels)
        nlp = spacy.load(_SPACY_MODEL.get(language, "en_core_web_sm"),
                         disable=["parser", "senter", "lemmatizer"])
        if "ner" not in nlp.pipe_names:
            raise ValueError("the spaCy model for %r has no NER component" % language)
        known = set(nlp.get_pipe("ner").labels)
        unknown = labels - known
        if unknown:
            raise ValueError("ner_labels %s not produced by this spaCy model; available: %s"
                             % (sorted(unknown), sorted(known)))
        ent, tot = Counter(), Counter()
        for doc in nlp.pipe(docs, batch_size=64):
            for t in doc:
                if t.is_alpha and len(t) > 1:
                    w = t.text.lower()
                    tot[w] += 1
                    if t.ent_type_ in labels:
                        ent[w] += 1
        names = {w for w, c in ent.items()
                 if c >= min_count and c / tot[w] >= min_ratio and w not in stop}
    # Plural/possessive stems: "dumbledores" counts as a name if "dumbledore" is one.
    return names


def normalize_ner_labels(ner_labels: Union[str, Sequence[str]]) -> Set[str]:
    """Upper-case a label or list of labels; reject an empty selection."""
    if isinstance(ner_labels, str):
        ner_labels = [ner_labels]
    labels = {str(l).strip().upper() for l in ner_labels if str(l).strip()}
    if not labels:
        raise ValueError("ner_labels must name at least one entity label, e.g. ['PERSON', 'GPE']")
    return labels


def _is_name(w: str, names: Set[str]) -> bool:
    return w in names or (w.endswith("s") and w[:-1] in names)


def proper_noun_share(documents: Iterable[str], names: Set[str], language: str = "english") -> float:
    """Fraction of non-stopword word tokens that are names."""
    stop = _stop(language)
    n_name = n_all = 0
    for d in documents:
        for t in _WORD.findall(_POSS.sub("", d)):
            w = t.lower()
            if w in stop:
                continue
            n_all += 1
            if _is_name(w, names):
                n_name += 1
    return n_name / n_all if n_all else 0.0


def scrub_proper_nouns(text: str, names: Set[str]) -> str:
    """Delete every occurrence (any case, with possessive 's) of a name."""
    text = _POSS.sub("", text)

    def rep(m):
        return " " if _is_name(m.group(0).lower(), names) else m.group(0)

    return _WORD.sub(rep, text)
