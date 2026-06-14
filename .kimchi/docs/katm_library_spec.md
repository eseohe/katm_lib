# KATM Library Packaging Spec

## Goal
Package the existing Keyphrase Anchored Topic Modeling (KATM) code into a properly structured, installable Python library with documentation, requirements, and test notebooks.

## Source
All code lives under:
`/Users/eustaceebhotemhen/Documents/Eustace/EUSTACE/key_key_words_extraction/gemini/keyphrase_anchored_tm/`

Exclude the `original_impl/` folder. All other `.py` files in the root, `__init__.py`, and `octis_eval/` are included.

## Target Directory
`/Users/eustaceebhotemhen/Documents/Eustace/EUSTACE/repos/katm/`

## Library Structure

```
katm/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── src/
│   └── katm/
│       ├── __init__.py
│       ├── topic_model.py           (copied from topic_model.py)
│       ├── topic_model_fast.py      (copied from topic_model_fast_exp.py)
│       ├── topic_assignment.py      (copied from topic_assignment.py)
│       ├── topic_assignment_fast.py (copied from topic_assignment_fast_exp.py)
│       ├── clustering.py            (copied from clustering.py)
│       ├── embedding.py             (copied from embedding.py)
│       ├── keyphrase_extractor.py   (copied from keyphrase_extractor.py)
│       ├── document_builder.py      (copied from document_builder.py)
│       ├── utils.py                 (copied from utils.py)
│       └── octis_eval.py            (copied from octis_eval/evaluate.py)
├── notebooks/
│   ├── 01_quickstart.ipynb          (basic usage, built-in sklearn dataset)
│   ├── 02_api_walkthrough.ipynb     (detailed API demo of all options)
│   ├── 03_speed_comparison.ipynb    (compare KATM vs KATMFast)
│   ├── bbc_news_experiment.ipynb    (BBC news topic extraction)
│   ├── newsgroups_experiment.ipynb  (20 newsgroups full experiment)
│   └── ekphrastic_experiment.ipynb  (ekphrastic poetry topics)
└── tests/
    ├── test_import.py
    └── test_pipeline.py
```

## Chunk 1: Source Package (`src/katm/`)

### Files to create
All 11 `.py` files under `src/katm/`.

### Rules
- **Copy source code verbatim** — do not modify algorithms, functions, or comments.
- Keep relative imports as-is (`.module`) — they are correct inside a package.
- For `topic_model_fast.py`, class `KATMFast` and `_vectorized_dedup` stay the same.
- For `topic_assignment_fast.py`, class `WordTopicProjectorFast` stays the same.
- `octis_eval.py`: the code from `octis_eval/evaluate.py` gets flattened into a single module.

### `__init__.py` content
```python
"""Keyphrase Anchored Topic Modeling (KATM) package."""

from .topic_model import KATM
from .topic_model_fast import KATMFast
from .document_builder import DocumentBuilder
from .keyphrase_extractor import KeyphraseExtractor
from .embedding import SentenceEmbedder
from .clustering import GMMTopicClusterer
from .topic_assignment import WordTopicProjector
from .topic_assignment_fast import WordTopicProjectorFast
from .utils import clean_text, build_vocabulary, extract_content_words

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
```

### Acceptance criteria
1. All 11 files exist and contain correct code.
2. `PYTHONPATH=src python -c "import katm; print(katm.__version__)"` succeeds.
3. `PYTHONPATH=src python -c "from katm import KATM, KATMFast, DocumentBuilder, KeyphraseExtractor, SentenceEmbedder, GMMTopicClusterer, WordTopicProjector, WordTopicProjectorFast"` succeeds.

## Chunk 2: Requirements & Packaging Metadata

### `requirements.txt`
```
numpy
scikit-learn
sentence-transformers
torch
nltk
keybert
yake
rake-nltk
spacy
gensim
pandas
matplotlib
jupyter
notebook
octis
```

### `setup.py`
Standard setuptools `setup.py` with package discovery under `src/`, version `0.1.0`.

### `pyproject.toml`
Standard PEP 517/518 `pyproject.toml` with `build-system`, `[project]` metadata.

### Acceptance criteria
1. `pip install -e .` parses without error.
2. `python -c "import katm"` succeeds after install.

## Chunk 3: README.md

### Content sections
1. Title & one-line description.
2. Installation instructions.
3. Quick start code block (fit KATM on 20 newsgroups or a toy corpus).
4. API overview with short descriptions of each public class.
5. Notebook index.
6. Citation info.
7. License placeholder (MIT).

## Chunk 4: Notebooks

### Notebook content rules
- Self-contained with inline dataset loading (sklearn datasets).
- No reliance on external data paths outside the repo.
- Clear markdown cells explaining each step.
- Output cells should be cleared.

### Notebook list
1. `01_quickstart.ipynb` — Load 20 newsgroups subset, fit KATM, print topics, compare algorithms.
2. `02_api_walkthrough.ipynb` — Walk through every major class and parameter.
3. `03_speed_comparison.ipynb` — Compare KATM vs KATMFast with `%%time`.
4. `bbc_news_experiment.ipynb` — BBC news topic extraction demo.
5. `newsgroups_experiment.ipynb` — Full 20 newsgroups experiment with LDA baseline.
6. `ekphrastic_experiment.ipynb` — Poetry/ekphrastic text demo with inline corpus.

### Acceptance criteria
1. All 6 `.ipynb` files exist under `notebooks/`.
2. Notebook JSON is valid (`jupyter nbconvert --to notebook` works).

## Chunk 5: Basic Tests

### `tests/test_import.py`
Import katm and all public symbols. Assert `__version__ == "0.1.0"`.

### `tests/test_pipeline.py`
Fit KATM on a tiny synthetic corpus (5 documents). Assert topics_ is populated. Assert transform() returns proper shape. Fit KATMFast on same corpus.

### Acceptance criteria
1. `python -m pytest tests/` passes.
2. Tests run in under 30 seconds.
