# KATM — Complete Usage Guide

This guide walks you through every step of using the **Keyphrase Anchored Topic Modeling (KATM)** library, from cloning the repository to producing evaluated topic models.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Clone & Install](#2-clone--install)
3. [Prepare Your Data](#3-prepare-your-data)
4. [Run KATM — Basic Workflow](#4-run-katm--basic-workflow)
5. [Run KATM — Advanced Options](#5-run-katm--advanced-options)
6. [Compare Keyphrase Extractors](#6-compare-keyphrase-extractors)
7. [Use KATMFast for Speed](#7-use-katmfast-for-speed)
8. [Inspect Results](#8-inspect-results)
9. [Evaluate with OCTIS Metrics](#9-evaluate-with-octis-metrics)
10. [Use Standalone Components](#10-use-standalone-components)
11. [Run the Notebooks](#11-run-the-notebooks)
12. [End-to-End Example Script](#12-end-to-end-example-script)

---

## 1. Prerequisites & Environment Setup

**Python version:** 3.9 or newer.

**Recommended:** Use a virtual environment so dependencies do not conflict with other projects.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Large dependencies** (PyTorch + transformer models) will download automatically on first use.

---

## 2. Clone & Install

```bash
# Clone from GitHub (replace with your actual repo URL)
git clone https://github.com/eseohe/katm_lib.git
cd katm_lib    # or whatever you named the repo locally

# Install all Python dependencies
pip install -r requirements.txt

# Install KATM itself in editable mode
pip install -e .
```

**What this does:**
- Installs core scientific libraries (`numpy`, `scikit-learn`, `scipy`)
- Installs deep-learning / NLP libraries (`torch`, `sentence-transformers`, `nltk`, `spacy`, `gensim`)
- Installs keyphrase extractors (`keybert`, `yake`, `rake-nltk`)
- Installs evaluation & notebook tools (`octis`, `jupyter`, `pandas`, `matplotlib`)
- Makes `katm` importable from anywhere once the environment is active

---

## 3. Prepare Your Data

### 3.1 Minimal input format

KATM expects a **list of raw text strings**, one per document:

```python
documents = [
    "The James Webb Space Telescope captured stunning infrared images of distant galaxies.",
    "NASA's Perseverance rover found organic molecules in Jezero Crater on Mars.",
    "A new quantum computing chip from IBM broke the 1000-qubit barrier.",
    "CRISPR gene editing was used to treat sickle cell disease in a landmark clinical trial.",
    "The WHO declared the end of the global health emergency for COVID-19.",
]
```

### 3.2 Loading from files

```python
import glob

# Load all .txt files from a directory
text_files = sorted(glob.glob("data/articles/*.txt"))
documents = [open(f, encoding="utf-8").read() for f in text_files]
```

### 3.3 Loading from sklearn datasets

```python
from sklearn.datasets import fetch_20newsgroups

# Fetch 20 newsgroups (no headers/footers for cleaner text)
newsgroups = fetch_20newsgroups(
    subset="train",
    categories=["sci.space", "sci.med", "comp.graphics", "rec.sport.baseball"],
    remove=("headers", "footers", "quotes"),
)
documents = newsgroups.data
labels = newsgroups.target  # ground-truth labels if you want to evaluate clustering
```

### 3.4 Grouping short posts with DocumentBuilder

If your raw data is very short (e.g., tweets, forum posts, chat messages), group them into longer documents before fitting:

```python
from katm import DocumentBuilder

# Suppose `posts` is a list of short strings
posts = ["Great match today!", "That goal was incredible", "Another win for the team"] * 100

builder = DocumentBuilder(strategy="post_group", chunk_size=5)
documents = builder.build(posts)
# Result: each "document" is 5 posts concatenated together
```

Strategies:
- `"paragraph_group"` — split on `\n\n`, group `chunk_size` paragraphs
- `"fixed_window"` — split into sentences, group `chunk_size` sentences
- `"post_group"` — group `chunk_size` raw items together

---

## 4. Run KATM — Basic Workflow

### 4.1 Import and fit

```python
from katm import KATM

# Create model with default settings
model = KATM(
    n_topics=5,            # Number of topics to discover
    n_keyphrases=10,       # Extract 10 keyphrases per document
    kp_algorithm="keybert", # Use KeyBERT for keyphrase extraction
    top_n_words=20,        # Store top 20 words per topic
)

# Fit on your corpus
model.fit(documents)
```

### 4.2 Print the topics

```python
# Pretty-print all topics with probabilities
model.print_topics(n_words=10)
```

Typical output:
```
============================================================
KATM Topics (n_topics=5)
============================================================

Topic 0:
  telescope (0.412), galaxies (0.398), infrared (0.385), webb (0.371), images (0.340)

Topic 1:
  mars (0.421), rover (0.398), crater (0.376), nasa (0.341), organic (0.312)

Topic 2:
  quantum (0.445), computing (0.423), chip (0.387), ibm (0.355), qubit (0.321)
...
============================================================
```

### 4.3 Access topics programmatically

```python
# Get top words for a specific topic
topic_0_words = model.get_topic_words(topic_id=0, n=10)
# -> [("telescope", 0.412), ("galaxies", 0.398), ...]

# Get document-topic distribution for doc index 0
doc_topics = model.get_document_topics(doc_idx=0)
# -> {0: 0.72, 2: 0.18, 1: 0.06, ...}

# All topic distributions as a numpy array
doc_topic_matrix = model.fit_transform(documents)
# -> shape: (n_documents, n_topics)
```

---

## 5. Run KATM — Advanced Options

### 5.1 Full parameter reference

```python
model = KATM(
    fast=False,                      # True = use KATMFast; False = standard KATM
    kp_algorithm="keybert",          # "keybert" | "rake" | "yake" | "tfidf" | "gsc"
    n_keyphrases=10,                 # Number of keyphrases extracted per document
    embedding_model="all-MiniLM-L6-v2",  # Sentence-transformer model name
    n_topics=10,                     # Number of GMM clusters (topics)
    soft_threshold=0.15,             # Min cosine similarity for word-topic assignment
    top_n_words=20,                  # Number of top words stored per topic
    min_df=3,                        # Min document frequency for content words
    yake_use_position=False,         # Include word-position feature in YAKE
    tfidf_ngram_range=(1, 2),        # n-gram range for TF-IDF keyphrase mode
    keybert_ngram_range=(1, 2),      # n-gram range for KeyBERT candidates
    content_words_method="tfidf",    # "tfidf" (~fast) or "spacy" (~precise)
    normalize_embeddings=False,      # L2-normalise embeddings before GMM
    anchor_dedup_threshold=0.85,     # Cosine threshold for semantic anchor dedup
    min_anchor_df=2,                 # Min docs a keyphrase must appear in
    max_anchor_df_ratio=0.4,         # Max fraction of docs a phrase may appear in
    mmr_diversity=0.2,               # MMR lambda: relevance/diversity trade-off
    mmr_max_sim=0.85,                # Hard dedup cap for word-list MMR
)
```

### 5.2 When to change parameters

| Parameter | Change when... |
|---|---|
| `kp_algorithm` | KeyBERT is slow on large corpora → try `"rake"` or `"tfidf"` for speed. GSC gives best semantic coverage but is slowest. |
| `n_keyphrases` | Short documents → reduce to 3-5. Long documents → increase to 20-30. |
| `embedding_model` | Need better quality → `"all-mpnet-base-v2"` (slower). Need speed → `"all-MiniLM-L6-v2"` (default). |
| `n_topics` | More topics = finer granularity. Use domain knowledge or grid-search with coherence metrics. |
| `min_df` | Small corpus (<100 docs) → reduce to 1-2. Large corpus (>10k) → increase to 5-10. |
| `content_words_method` | Need speed → `"tfidf"` (default). Need linguistic precision → `"spacy"` (requires `en_core_web_sm`). |
| `anchor_dedup_threshold` | Set to `None` to disable dedup. Reduce to 0.80 for stricter dedup. Increase to 0.90 for looser dedup. |
| `normalize_embeddings` | Enable when documents and keyphrases have very different lengths (improves GMM robustness). |

---

## 6. Compare Keyphrase Extractors

```python
import time
from katm import KATM

algorithms = ["keybert", "rake", "yake", "tfidf", "gsc"]

for algo in algorithms:
    print(f"\n=== {algo.upper()} ===")
    model = KATM(n_topics=5, kp_algorithm=algo, n_keyphrases=8, top_n_words=15)

    t0 = time.time()
    model.fit(documents[:100])  # subset for quick demo
    elapsed = time.time() - t0

    print(f"Fit time: {elapsed:.1f}s")
    model.print_topics(n_words=5)
```

**Typical trade-offs:**
- **KeyBERT** — Best quality, slowest (requires transformer model load)
- **GSC** — Excellent semantic coverage, slow (embeds candidates + sentences)
- **RAKE** — Fast, works offline, can produce noisy short phrases
- **YAKE** — Fast, good for short documents, sensitive to boilerplate text
- **TF-IDF** — Fastest, corpus-aware (down-weights generic words via IDF), reliable on short docs

---

## 7. Use KATMFast for Speed

`KATMFast` is a drop-in replacement for `KATM` with two speedups:
- **S3** — Vectorized anchor deduplication (replaces O(n²) sklearn calls with a single BLAS matrix operation)
- **S4** — Incremental MMR (replaces O(N·K²) loop with O(N·K) updates)

You can switch to the fast variant with a single parameter — no import changes needed:

```python
from katm import KATM

model = KATM(n_topics=10, kp_algorithm="keybert", fast=True)
model.fit(documents)
model.print_topics(n_words=10)
```

Or import `KATMFast` directly:

```python
from katm import KATMFast

model = KATMFast(n_topics=10, kp_algorithm="keybert")
model.fit(documents)
model.print_topics(n_words=10)
```

The API is **identical** to `KATM`. Topics are numerically identical in most cases; only runtime differs.

---

## 8. Inspect Results

### 8.1 Topic words

```python
for topic_id in sorted(model.topics_.keys()):
    words = model.get_topic_words(topic_id, n=10)
    print(f"Topic {topic_id}: {', '.join(w for w, _ in words)}")
```

### 8.2 Document-topic distributions

```python
import numpy as np

# For training documents
doc_topics = model.doc_topic_probs_  # list of np arrays

# For new documents
doc_topics_new = model.transform(new_documents)  # np array shape (n_new, n_topics)

# Find dominant topic per document
dominant_topics = np.argmax(doc_topics_new, axis=1)
```

### 8.3 Find documents for a topic

```python
topic_id = 0
threshold = 0.3
docs_for_topic = [
    i for i, probs in enumerate(model.doc_topic_probs_)
    if probs[topic_id] > threshold
]
print(f"Topic {topic_id} has {len(docs_for_topic)} docs above threshold {threshold}")
```

---

## 9. Evaluate with OCTIS Metrics

KATM ships with evaluation utilities compatible with [OCTIS](https://github.com/MIND-Lab/OCTIS).

```python
from katm import KATM
from katm.octis_eval import compute_metrics
from octis.dataset.dataset import Dataset

# Fit KATM
model = KATM(n_topics=10, kp_algorithm="keybert")
model.fit(documents)

# Prepare model output format for OCTIS
model_output = {
    "topics": [
        [w for w, _ in model.get_topic_words(tid, n=10)]
        for tid in sorted(model.topics_.keys())
    ],
    "topic-document-matrix": np.array(model.doc_topic_probs_).T,  # shape (n_topics, n_docs)
}

# Load OCTIS dataset (e.g., 20NG preprocessed)
dataset = Dataset()
dataset.load_custom_dataset_from_folder("path/to/20ng_octis/")

# Compute metrics
metrics = compute_metrics(model_output, dataset, topk=10)
print(metrics)
```

**Metrics returned:**
- `CV` — Coherence (c_v), highest correlation with human judgments
- `NPMI` — Normalized Pointwise Mutual Information
- `UCI` / `Umass` — Alternative coherence measures
- `TD` — Topic Diversity (fraction of unique words across topics)
- `IRBO` — Inverted Rank-Biased Overlap
- `NMI` — Normalized Mutual Information (requires ground-truth labels)
- `Purity` — Clustering purity (requires labels)
- `F1` — Micro-F1 of SVM on topic features (requires train/test split)

---

## 10. Use Standalone Components

You can use KATM's building blocks independently for custom pipelines.

### 10.1 KeyphraseExtractor

```python
from katm import KeyphraseExtractor

extractor = KeyphraseExtractor(algorithm="rake", n_keyphrases=5)
keyphrases = extractor.extract(documents)
# Returns: [["phrase1", "phrase2", ...], [...], ...]
```

### 10.2 SentenceEmbedder

```python
from katm import SentenceEmbedder

embedder = SentenceEmbedder(model_name="all-MiniLM-L6-v2")
embeddings = embedder.encode(documents, batch_size=32)
# Returns: np array shape (n_docs, 384)
```

### 10.3 GMMTopicClusterer

```python
from katm import GMMTopicClusterer
import numpy as np

clusterer = GMMTopicClusterer(n_topics=5)
embeddings = np.random.randn(100, 384)  # your embeddings
clusterer.fit(embeddings)

# Get topic probabilities
probs = clusterer.predict_proba(embeddings)  # shape (100, 5)
centroids = clusterer.cluster_centers_         # shape (5, 384)
```

### 10.4 WordTopicProjector

```python
from katm import WordTopicProjector

# Requires a fitted clusterer + embedder
projector = WordTopicProjector(clusterer=clusterer, embedder=embedder)
vocab = ["galaxy", "mars", "quantum", "gene", "vaccine", "robot", "neural",...]

topic_words = projector.project_vocabulary(
    vocab,
    min_prob=0.15,
    mmr_diversity=0.2,
)
# Returns: {0: [("quantum", 0.41), ...], 1: [("mars", 0.42), ...], ...}
```

### 10.5 DocumentBuilder

```python
from katm import DocumentBuilder

# Group short posts into coherent documents
builder = DocumentBuilder(strategy="fixed_window", chunk_size=10)
docs = builder.build(raw_texts)
```

### 10.6 Utilities

```python
from katm import clean_text, build_vocabulary, extract_content_words

# Clean messy text
cleaned = clean_text("Some   messy\t\ntext with   extra spaces!!!")

# Build vocabulary with frequency filter
vocab = build_vocabulary(documents, min_df=3)

# Extract content words for custom projection
content_words = extract_content_words(documents, min_df=3, method="tfidf")
```

---

## 11. Run the Notebooks

The `notebooks/` directory contains 6 runnable demonstrations.

```bash
# Start Jupyter
cd notebooks
jupyter notebook

# Or use JupyterLab
jupyter lab
```

| Notebook | What you'll learn |
|---|---|
| `01_quickstart.ipynb` | Load data, fit KATM, print topics, try different extractors |
| `02_api_walkthrough.ipynb` | Every class and parameter explained with examples |
| `03_speed_comparison.ipynb` | See KATM vs KATMFast timing side-by-side |
| `bbc_news_experiment.ipynb` | Multi-category news topic extraction |
| `newsgroups_experiment.ipynb` | Full 20NG with LDA baseline comparison |
| `ekphrastic_experiment.ipynb` | Literary text topic modeling with inline poetry corpus |

All notebooks use `sys.path.insert(0, '../src')` so they work out of the box without installing the package.

---

## 12. End-to-End Example Script

Save this as `run_katm.py` and run with `python run_katm.py`:

```python
"""Complete KATM pipeline from data loading to evaluation."""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from katm import KATM

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading 20 newsgroups...")
categories = ["sci.space", "sci.med", "comp.graphics", "rec.sport.baseball"]
dataset = fetch_20newsgroups(
    subset="train",
    categories=categories,
    remove=("headers", "footers", "quotes"),
)
docs = dataset.data
labels = dataset.target
print(f"Loaded {len(docs)} documents across {len(categories)} categories")

# ── 2. Fit KATM ───────────────────────────────────────────────────────────
print("\nFitting KATM (KeyBERT)...")
model = KATM(
    n_topics=len(categories),
    kp_algorithm="keybert",
    n_keyphrases=10,
    top_n_words=15,
    min_df=2,
    # fast=True,  # Uncomment to use KATMFast instead
)
model.fit(docs)

# ── 3. Print topics ───────────────────────────────────────────────────────
print("\nDiscovered topics:")
model.print_topics(n_words=10)

# ── 4. Transform new documents ────────────────────────────────────────────
print("\nTopic distribution for first 3 documents:")
doc_probs = model.fit_transform(docs)
for i in range(3):
    topic_dist = {t: round(float(p), 3) for t, p in enumerate(doc_probs[i]) if p > 0.05}
    print(f"  Doc {i}: {topic_dist}")

# ── 5. Find dominant topic per document ───────────────────────────────────
dominant = np.argmax(doc_probs, axis=1)
print("\nDominant topic counts:", {t: int((dominant == t).sum()) for t in range(model.n_topics)})

print("\nDone!")
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'katm'` | Run `pip install -e .` from the repo root, or add `sys.path.insert(0, 'src')` |
| `LookupError: Resource punkt not found` | NLTK data auto-downloads on first import; if it fails, run `python -m nltk.downloader punkt punkt_tab stopwords` |
| KeyBERT is very slow on first run | The sentence-transformer model downloads on first use (~80 MB). Subsequent runs are cached. |
| `spacy` content-words method fails | Run `python -m spacy download en_core_web_sm` |
| Out of memory on large corpora | Reduce `max_anchor_df_ratio` (e.g., 0.2), increase `min_anchor_df` (e.g., 5), or use `"tfidf"` / `"rake"` instead of `"keybert"` |
| Topics look generic/boilerplate | Increase `min_df`, enable `normalize_embeddings=True`, or tighten `anchor_dedup_threshold` |

---

## Next Steps

- **Tune hyperparameters:** Grid-search `n_topics`, `kp_algorithm`, and `n_keyphrases` against coherence metrics.
- **Add your own keyphrase extractor:** Subclass `KeyphraseExtractor` and implement `_extract_myalgo()`.
- **Contribute:** Extend `octis_eval.py` with new metrics, or add notebooks for your domain.
