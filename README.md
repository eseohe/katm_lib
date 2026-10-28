# KATM — Keyphrase Anchored Topic Modeling

KATM is a topic modeling library that anchors topics on keyphrases. It extracts semantically meaningful multi-word phrases (KeyBERT, RAKE, YAKE, TF-IDF, or FLaKE), embeds them as **anchors**, clusters the anchors into topics, then projects the vocabulary onto the topics to read each one off as a list of words. This gives more coherent and interpretable topics than bag-of-words approaches.

The pipeline has four swappable stages:

1. **Anchors**: keyphrases from each document (`keyphrase_scope="per_document"`), one shared corpus-wide pool (`"global"`), or FLaKE's coverage-based selection (`"flake"`).
2. **Clustering**: the anchor embeddings are grouped into topics. Choose between spherical k-means, Gaussian mixtures and others with `clustering_method`.
3. **Word scoring**: words are ranked for each topic by similarity, margin or posterior (`word_scoring`).
4. **Content words**: the vocabulary the topic words are drawn from (`content_words_method`), optionally POS-filtered.

---

## Installation

```bash
git clone https://github.com/eseohe/katm_lib.git
cd katm_lib

pip install -r requirements.txt
pip install -e .
```

`keyphrase_scope="flake"` and the streaming content-word extractors additionally need the `flake` library (FLaKE keyphrase extraction, a separate package) (a clear `ImportError` is raised if it is missing):

```bash
pip install -e /path/to/flake
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

categories = ["sci.med", "sci.space", "comp.graphics", "rec.sport.baseball"]
dataset = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
docs = dataset.data

# Spherical k-means on the anchors (recommended, see "Clustering the anchors")
model = KATM(n_topics=5, kp_algorithm="keybert", clustering_method="spherical_kmeans")
model.fit(docs)

for topic_id, words in model.topics_.items():
    print(f"Topic {topic_id}: {words}")
```

`model.topics_` is a dict mapping each topic id to its list of words.

---

## Clustering the anchors

The anchors (keyphrase embeddings) are clustered into `n_topics` groups with `clustering_method`. Every method is selectable from `KATM`; extra arguments go through `clustering_kwargs`.

| `clustering_method` | What it does | Notes |
|---|---|---|
| `"gmm"` (library default) | Gaussian mixture (`GMMTopicClusterer`), soft assignments with `soft_threshold` | Original method; seed-sensitive, can produce one catch-all cluster |
| **`"spherical_kmeans"`** | k-means on L2-normalised anchors, i.e. cosine geometry, 10 inits | **Recommended.** Stable across seeds, no covariance to tune |
| `"gmm_ninit"` | GMM with 10 initialisations | Nearly as stable as spherical k-means |
| `"gmm_hub"` | GMM with hub-cluster discarding | Can lose a real topic on short texts |
| `"agglomerative"` | Ward linkage | Deterministic; weaker on some corpora |
| `"overcluster_merge"` | Over-cluster then merge | Tends to chain into one catch-all cluster |

```python
from katm import KATM

# Spherical k-means (recommended)
model = KATM(n_topics=10, clustering_method="spherical_kmeans")

# More restarts, fixed seed
model = KATM(n_topics=10, clustering_method="spherical_kmeans",
             clustering_kwargs={"n_init": 20, "random_state": 7})

# The original Gaussian mixture
model = KATM(n_topics=10, clustering_method="gmm", gmm_covariance_type="full")
```

**Why spherical k-means.** In the KATM-FLaKE experiments (FLaKE anchors, K = 1000, seeds 0/1/2; BBC News, AG News, 20 Newsgroups) it matched or beat the GMM on all three datasets while removing the seed instability: on BBC, NMI standard deviation fell from 0.079 to 0.002 and all five gold classes received a topic in every seed. Clustering takes under a second for every method. These results come from the research testbed (`katm_flake`) and were not re-run when the library was synced; treat them as guidance and compare on your own corpus.

Notes:

- For every method other than `"gmm"`, `soft_threshold`, `gmm_covariance_type` and `discard_hub_clusters` are not used, and `model.gmm_` is `None`.
- The library default stays `"gmm"` so existing code behaves as before. Pass `clustering_method="spherical_kmeans"` to opt in.
- `AltTopicClusterer(n_topics, method="spherical_kmeans", ...)` can be used on its own on any embedding matrix, as `GMMTopicClusterer` can.

### Choosing the number of topics

```python
model = KATM(n_topics="auto", n_topics_range=(2, 20), n_topics_method="bic")
```

`select_n_topics` can also be called directly on the anchor embeddings. The scan curve is stored in `model.n_topics_selection_scores_`. See `USAGE_GUIDE.md` §5.5.

### Clustering initialisation and anchor filtering (exploratory)

- `clustering_init="coverage"` seeds spherical k-means or GMM from FLaKE's coverage graph (only with `keyphrase_scope="flake"`).
- `anchor_isolation_drop_frac` drops the most isolated anchors before clustering.

---

## Choosing topic words

Once anchors are clustered, `word_scoring` decides how candidate words are ranked for each topic:

| `word_scoring` | Score | Effect |
|---|---|---|
| `"similarity"` (default) | cosine to the topic centroid | Original behaviour; broad topics can pick up generic words |
| `"posterior"` | cosine × softmax(cosines / `word_scoring_temp`) | Recommended with spherical k-means: replaces generic intruders (*movie, business, something*) with topical words |
| `"margin"` | cosine to this topic − cosine to the best other topic | Purest lists but favours rare, very specific words |

```python
model = KATM(n_topics=10, clustering_method="spherical_kmeans", word_scoring="posterior")
```

---

## API Overview

| Class / function | Description |
|---|---|
| `KATM` | Full KATM topic model. |
| `DocumentBuilder` | Preprocesses raw documents into cleaned text suitable for topic modeling. |
| `KeyphraseExtractor` | Extracts keyphrases from documents using a configurable algorithm (KeyBERT, RAKE, YAKE, TF-IDF). |
| `SentenceEmbedder` | Computes sentence-level embeddings using `sentence-transformers`. |
| `AltTopicClusterer` | Clusters anchor embeddings: spherical k-means (default method of this class), GMM variants, agglomerative, over-cluster-merge. |
| `GMMTopicClusterer` | The original Gaussian-mixture clusterer. |
| `select_n_topics` | Chooses the number of topics from the anchor embeddings. |
| `WordTopicProjector` | Projects vocabulary words into topic space via keyphrase-topic alignment. |
| `clean_text`, `build_vocabulary`, `extract_content_words`, `strip_email_addresses` | Text utilities. |

---

## Keyphrase Algorithms

KATM supports multiple keyphrase extraction algorithms, configurable via the `kp_algorithm` parameter:

- **KeyBERT**: sentence-transformer embeddings and cosine similarity to find keyphrases closest to the document embedding.
- **RAKE**: Rapid Automatic Keyword Extraction using word co-occurrence and phrase boundary detection.
- **YAKE**: unsupervised keyword extraction from statistical features (frequency, position, relatedness).
- **TF-IDF**: top-scoring n-grams by TF-IDF weight within each document.

```python
model = KATM(n_topics=5, kp_algorithm="rake")   # RAKE-based topics
model = KATM(n_topics=5, kp_algorithm="yake")   # YAKE-based topics
```

### Keyphrase scope: per-document, global, or FLaKE

With `kp_algorithm="keybert"`, `keyphrase_scope` controls how the anchor pool is built:

- `"per_document"` (default): KeyBERT runs once per document; candidates are pooled and filtered by document frequency.
- `"global"`: one pass builds and ranks a single shared candidate vocabulary for the whole corpus. Typically 2.5–3.5× faster on medium and large corpora, since each unique candidate is embedded once.
- `"flake"`: the anchor pool is FLaKE's corpus-level selection (facility location over a document × phrase relevance graph). Phrases rank high in the documents they represent and together cover the corpus, with no separate frequency filter or semantic dedup, and FLaKE's phrase embeddings are reused as anchor embeddings. In the testbed experiments it gave about a 3× speed-up over per-document KeyBERT with equal NMI/Purity/F1. Needs the `flake` library.

```python
model = KATM(n_topics=5, kp_algorithm="keybert", keyphrase_scope="global", n_keyphrases_total=500)

# FLaKE anchors + spherical k-means + posterior word scoring
model = KATM(n_topics=10, keyphrase_scope="flake", n_keyphrases_total=1000,
             clustering_method="spherical_kmeans", word_scoring="posterior",
             flake_kwargs={"min_df": 5, "max_df": 0.3})
```

There's no universally better setting; compare on your own corpus. See `USAGE_GUIDE.md` §5.3 for the full parameter set.

---

## Content words and large corpora

`content_words_method` chooses the vocabulary topic words are drawn from: `"tfidf"` (default), `"spacy"` (full POS tagging, slow), or the fast POS-filtered `"sampled_pos"`, `"type_pos"` and `"type_pos_templates"`. `streaming=True` fits corpora that don't fit in memory (works with `keyphrase_scope="flake"`). See `USAGE_GUIDE.md` §5.8.

---

## Notebooks

The `notebooks/` directory contains runnable demonstrations of KATM:

| Notebook | Description |
|---|---|
| `01_quickstart.ipynb` | Basic usage: load 20 newsgroups, fit `KATM`, print topics, compare keyphrase algorithms. |
| `02_api_walkthrough.ipynb` | Detailed walkthrough of every public class and parameter in the KATM API. |
| `bbc_news_experiment.ipynb` | Topic extraction on the BBC news dataset; evaluates topic coherence. |
| `bbc_classification_experiment.ipynb` | BBC news with document classification evaluation. |
| `newsgroups_experiment.ipynb` | Full 20 Newsgroups experiment with LDA baseline comparison and OCTIS evaluation. |
| `ekphrastic_experiment.ipynb` | KATM applied to an ekphrastic poetry corpus; topic interpretability on literary text. |

---

## Citation

Research papers for KATM are forthcoming. When available, citation information will be posted here. Early drafts are available:

- **KATM ACL Draft**: `KATM_ACL_final.docx` (submitted to ACL)
- **KATM DH Draft**: `KATM_DH_final.docx` (submitted to Digital Humanities)

Please check back for published paper DOIs and BibTeX entries.

---

## License

MIT License
