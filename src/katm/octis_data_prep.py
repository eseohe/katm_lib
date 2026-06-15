"""Dataset loading and preprocessing for OCTIS evaluation.

Supports:
  - '20NewsGroup' / 'BBC_News'  — fetched from OCTIS's preprocessed GitHub repo
  - 'ag_news', 'dbpedia_14', 'reuters_r8' — downloaded from HuggingFace and
    preprocessed with OCTIS's Preprocessing pipeline, then cached as OCTIS
    Dataset objects in ~/octis_data/<name>/

Usage::

    from katm.octis_data_prep import load_dataset
    dataset = load_dataset('20NewsGroup')   # returns octis.dataset.dataset.Dataset
    dataset = load_dataset('BBC_News')
    dataset = load_dataset('ag_news', n_docs=10000)
"""

import os
import pickle
import random
import tempfile
from pathlib import Path

from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing


_CACHE_DIR = Path.home() / "octis_data"

_R8_CATEGORIES = {"earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship"}

_HF_CONFIGS = {
    "ag_news": {
        "hf_name": "ag_news",
        "text_field": "text",
        "label_field": "label",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "dbpedia_14": {
        "hf_name": "dbpedia_14",
        "text_field": "content",
        "label_field": "label",
        "label_names": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
        ],
    },
}


def load_dataset(name: str, n_docs: int = None, seed: int = 42) -> Dataset:
    """Load an OCTIS Dataset by name, with optional document cap.

    Args:
        name: One of '20NewsGroup', 'BBC_News', 'ag_news', 'dbpedia_14', 'reuters_r8'.
        n_docs: If set, subsample to at most this many documents (class-balanced
                where possible). None = use full corpus.
        seed: Random seed for reproducible subsampling.

    Returns:
        octis.dataset.dataset.Dataset ready for OCTIS models and metrics.
    """
    if name in ("20NewsGroup", "BBC_News"):
        d = Dataset()
        d.fetch_dataset(name)
        return d

    if name == "reuters_r8":
        return _load_reuters_r8(n_docs=n_docs, seed=seed)

    if name in _HF_CONFIGS:
        return _load_hf_dataset(name, n_docs=n_docs, seed=seed)

    raise ValueError(
        f"Unknown dataset '{name}'. "
        "Choose from: 20NewsGroup, BBC_News, ag_news, dbpedia_14, reuters_r8."
    )


def _load_hf_dataset(name: str, n_docs: int = None, seed: int = 42) -> Dataset:
    cache_path = _CACHE_DIR / f"{name}_{n_docs or 'full'}_{seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import datasets as hf_datasets

    cfg = _HF_CONFIGS[name]
    raw = hf_datasets.load_dataset(cfg["hf_name"])
    all_texts, all_labels = [], []
    for split_name in ("train", "test"):
        if split_name not in raw:
            continue
        split = raw[split_name]
        for item in split:
            text = item[cfg["text_field"]].strip()
            label = cfg["label_names"][item[cfg["label_field"]]]
            if text:
                all_texts.append(text)
                all_labels.append(label)

    if n_docs is not None and n_docs < len(all_texts):
        all_texts, all_labels = _balanced_sample(all_texts, all_labels, n_docs, seed)

    dataset = _texts_to_octis_dataset(all_texts, all_labels, seed=seed)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def _load_reuters_r8(n_docs: int = None, seed: int = 42) -> Dataset:
    import ast

    cache_path = _CACHE_DIR / f"reuters_r8_{n_docs or 'full'}_{seed}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import datasets as hf_datasets

    raw = hf_datasets.load_dataset("reuters21578", "ModHayes", trust_remote_code=True)
    all_texts, all_labels = [], []
    for split_name in ("train", "test"):
        if split_name not in raw:
            continue
        for item in raw[split_name]:
            topics_raw = item.get("topics", "[]") or "[]"
            try:
                cats = ast.literal_eval(topics_raw) if isinstance(topics_raw, str) else topics_raw
            except (ValueError, SyntaxError):
                cats = []
            r8_cats = [c.lower() for c in cats if c.lower() in _R8_CATEGORIES]
            if len(r8_cats) == 1:
                text = (item.get("text") or "").strip()
                if text:
                    all_texts.append(text)
                    all_labels.append(r8_cats[0])

    if n_docs is not None and n_docs < len(all_texts):
        all_texts, all_labels = _balanced_sample(all_texts, all_labels, n_docs, seed)

    dataset = _texts_to_octis_dataset(all_texts, all_labels, seed=seed)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def _balanced_sample(texts, labels, n_docs, seed):
    """Return a class-balanced subsample of size <= n_docs."""
    rng = random.Random(seed)
    from collections import defaultdict
    by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[lbl].append(i)
    n_classes = len(by_class)
    per_class = max(1, n_docs // n_classes)
    selected = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    rng.shuffle(selected)
    return [texts[i] for i in selected], [labels[i] for i in selected]


def _texts_to_octis_dataset(texts, labels, seed: int = 42) -> Dataset:
    """Preprocess raw texts into an OCTIS Dataset.

    Uses OCTIS's Preprocessing pipeline (lowercasing, lemmatisation,
    stopword removal, vocabulary filtering) and an 85/7.5/7.5 split.
    """
    with tempfile.TemporaryDirectory() as tmp:
        docs_path = os.path.join(tmp, "docs.txt")
        labels_path = os.path.join(tmp, "labels.txt")
        with open(docs_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

        prep = Preprocessing(
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True,
            lemmatize=True,
            remove_stopwords_spacy=True,
            min_df=5,
            max_df=0.85,
            min_chars=3,
            min_words_docs=3,
            split=True,
            language="english",
            verbose=False,
        )
        dataset = prep.preprocess_dataset(
            documents_path=docs_path,
            labels_path=labels_path,
        )
    return dataset
