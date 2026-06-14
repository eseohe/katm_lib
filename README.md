# KATM — Keyphrase Anchored Topic Modeling

KATM is a neural topic modeling library that anchors topic representations on document-level keyphrases. By leveraging keyphrase extraction algorithms (KeyBERT, RAKE, YAKE, TF-IDF, or GSC) to surface semantically meaningful multi-word phrases, KATM produces more coherent and interpretable topics than traditional bag-of-words approaches. KATM also ships a fast variant (`KATMFast`) that accelerates the word-topic projection step via vectorized deduplication.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/katm.git
cd katm

# Install dependencies
pip install -r requirements.txt

# Install KATM in editable mode
pip install -e .
```

---

## Getting Started

For a complete step-by-step guide covering data preparation, all parameters, keyphrase extractors, evaluation, and standalone components, see **[USAGE_GUIDE.md](USAGE_GUIDE.md)**.

---

## Quick Start

Fit KATM on the 20 Newsgroups dataset and print the discovered topics:

```python
from sklearn.datasets import fetch_20newsgroups
from katm import KATM

# Load a subset of 20 newsgroups for a quick demo
categories = ["sci.med", "sci.space", "comp.graphics", "rec.sport.baseball"]
dataset = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
docs = dataset.data

# Fit KATM with the default keyphrase extractor (KeyBERT + sentence-transformers)
model = KATM(n_topics=5, kp_algorithm="keybert")
model.fit(docs)

# Or use the fast variant (KATMFast) via a single parameter:
# model = KATM(n_topics=5, kp_algorithm="keybert", fast=True)

# Print discovered topics
for topic_id, keywords in enumerate(model.topics_):
    print(f"Topic {topic_id}: {keywords}")
```

`model.topics_` is a list of lists of `(word, weight)` tuples (or similar keyword representations depending on the keyphrase algorithm used).

---

## API Overview

| Class | Description |
|---|---|
| `KATM` | Full KATM topic model. Pass `fast=True` to get KATMFast without changing imports. |
| `KATMFast` | Accelerated KATM variant using vectorized word-topic projection (S3+S4). |
| `DocumentBuilder` | Preprocesses raw documents into tokenized, cleaned text suitable for topic modeling. |
| `KeyphraseExtractor` | Extracts keyphrases from documents using a configurable algorithm (KeyBERT, RAKE, YAKE, TF-IDF, GSC). |
| `SentenceEmbedder` | Computes sentence-level embeddings using `sentence-transformers` for keyphrase scoring. |
| `GMMTopicClusterer` | Clusters document/keyphrase embeddings into topic groups using Gaussian Mixture Models. |
| `WordTopicProjector` | Projects vocabulary words into topic space via keyphrase-topic alignment. |
| `WordTopicProjectorFast` | Vectorized, deduplicated version of `WordTopicProjector` for faster inference. |

---

## Keyphrase Algorithms

KATM supports multiple keyphrase extraction algorithms, configurable via the `keyphrase_algo` parameter:

- **KeyBERT** — Uses sentence-transformers embeddings and cosine similarity to find keyphrases closest to the document embedding.
- **RAKE** — Rapid Automatic Keyword Extraction using word co-occurrence and phrase boundary detection.
- **YAKE** — Unsupervised keyword extraction based on statistical features (frequency, position, relatedness).
- **TF-IDF** — Selects top-scoring n-grams by TF-IDF weight within each document.
- **GSC** — Graph-based Salient Keyphrase extraction using TextRank-style graph scoring.

Select the algorithm by passing `keyphrase_algo` to `KATM` or `KeyphraseExtractor`:
```python
model = KATM(n_topics=5, keyphrase_algo="rake")   # RAKE-based topics
model = KATM(n_topics=5, keyphrase_algo="yake")   # YAKE-based topics
```

---

## Notebooks

The `notebooks/` directory contains runnable demonstrations of KATM:

| Notebook | Description |
|---|---|
| `01_quickstart.ipynb` | Basic usage — load 20 newsgroups, fit `KATM`, print topics, compare keyphrase algorithms. |
| `02_api_walkthrough.ipynb` | Detailed walkthrough of every public class and parameter in the KATM API. |
| `03_speed_comparison.ipynb` | Benchmark `KATM` vs `KATMFast` with `%%time` magic on a medium-sized corpus. |
| `bbc_news_experiment.ipynb` | Topic extraction on the BBC news dataset; evaluates topic coherence. |
| `newsgroups_experiment.ipynb` | Full 20 Newsgroups experiment with LDA baseline comparison and OCTIS evaluation. |
| `ekphrastic_experiment.ipynb` | KATM applied to ekphrastic poetry corpus; demonstrates topic interpretability on literary text. |

---

## Citation

Research papers for KATM are forthcoming. When available, citation information will be posted here. Early drafts are available:

- **KATM ACL Draft** — `KATM_ACL_final.docx` (submitted to ACL)
- **KATM DH Draft** — `KATM_DH_final.docx` (submitted to Digital Humanities)

Please check back for published paper DOIs and BibTeX entries.

---

## License

MIT License