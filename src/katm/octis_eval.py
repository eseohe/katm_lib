"""Metric computation for OCTIS evaluation.

Metrics computed
----------------
Coherence (all via Gensim CoherenceModel, evaluated on training corpus):
  CV    c_v     [0, 1]      Indirect confirmation using NPMI vectors.
                            Highest correlation with human judgements.
                            Primary coherence metric in most papers.
  NPMI  c_npmi  [-1, +1]   Normalised PMI over a sliding window.
                            Standard in recent papers; also reported.
  UCI   c_uci   (-∞, +∞)   PMI over a sliding window (unnormalised).
  Umass u_mass  (-∞, 0)    Log-ratio of document co-occurrence counts.
                            Oldest measure; negative = better.

Diversity:
  TD    TopicDiversity      Fraction of unique words across all topics' top-k.
                            Range [0, 1]; 1 = fully distinct vocabularies.
  IRBO  InvertedRBO         Rank-Based Overlap inverted; penalises overlap in
                            ranked topic word lists. Range [0, 1].

Clustering (hard assignment via argmax of doc-topic matrix):
  NMI   Normalised Mutual Information between argmax topic assignments and
        ground-truth labels.  Range [0, 1]; 1 = perfect alignment.
  Purity  Fraction of documents in each cluster's majority class, averaged
          over clusters weighted by size.  Range [0, 1].

Classification:
  F1    Micro-F1 of a linear SVM trained on doc-topic-matrix features
        (train partition) and evaluated on the test partition.

Usage
-----
    from evaluate import compute_metrics
    m = compute_metrics(model_output, dataset, topk=10)
    # m keys: 'CV', 'NPMI', 'UCI', 'Umass', 'TD', 'IRBO', 'NMI', 'Purity', 'F1'
"""

import numpy as np
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.classification_metrics import F1Score


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_output(model_output, topk):
    """Strip empty/padding strings; enforce uniform topic length = topk."""
    return {
        "topics": [
            [w for w in topic if w][:topk]
            for topic in model_output.get("topics", [])
            if any(w for w in topic)
        ],
        "topic-document-matrix":      model_output.get("topic-document-matrix"),
        "test-topic-document-matrix": model_output.get("test-topic-document-matrix"),
    }


def _safe(fn, label):
    """Run fn(); return NaN and print on any exception."""
    try:
        return round(float(fn()), 4)
    except Exception as exc:
        print(f"    {label} error: {exc}")
        return float("nan")


def _nmi_purity(tdm, labels_all, last_train_idx):
    """Compute NMI and Purity from hard topic assignments.

    Args:
        tdm: (K, D_train) topic-document matrix (train partition).
        labels_all: full label list from dataset.get_labels().
        last_train_idx: index of last training document (from metadata).

    Returns:
        (nmi, purity) floats, or (nan, nan) on failure.
    """
    from sklearn.metrics import normalized_mutual_info_score
    from collections import Counter

    train_labels = labels_all[:last_train_idx]
    if tdm is None or len(train_labels) == 0:
        return float("nan"), float("nan")

    # Hard assignment: topic with highest probability per document
    hard_assignments = np.argmax(tdm, axis=0)   # shape (D_train,)

    if len(hard_assignments) != len(train_labels):
        return float("nan"), float("nan")

    # NMI
    nmi = round(float(normalized_mutual_info_score(
        train_labels, hard_assignments, average_method="arithmetic"
    )), 4)

    # Purity: for each topic cluster, find majority class count
    K = tdm.shape[0]
    total = len(train_labels)
    majority_sum = 0
    for k in range(K):
        cluster_labels = [train_labels[i] for i, t in enumerate(hard_assignments) if t == k]
        if cluster_labels:
            majority_sum += Counter(cluster_labels).most_common(1)[0][1]
    purity = round(majority_sum / total, 4) if total > 0 else float("nan")

    return nmi, purity


# ── main metric function ──────────────────────────────────────────────────────

def compute_metrics(model_output: dict, dataset, topk: int = 10) -> dict:
    """Compute all standard topic model evaluation metrics.

    Args:
        model_output: dict with keys 'topics', 'topic-document-matrix',
                      and optionally 'test-topic-document-matrix'.
        dataset: octis.dataset.dataset.Dataset.
        topk: number of top words per topic for coherence/diversity metrics.

    Returns:
        dict with keys: CV, NPMI, UCI, Umass, TD, IRBO, NMI, Purity, F1.
        Any metric that fails returns float('nan').
    """
    train_corpus, val_corpus, test_corpus = dataset.get_partitioned_corpus()
    out = _clean_output(model_output, topk)

    nan_result = {k: float("nan") for k in
                  ("CV", "NPMI", "UCI", "Umass", "TD", "IRBO", "NMI", "Purity", "F1")}

    if not out["topics"]:
        return nan_result

    effective_topk = min(topk, min(len(t) for t in out["topics"]))
    if effective_topk < 2:
        return nan_result

    m = {}

    # ── Coherence metrics ──────────────────────────────────────────────────────
    for key, measure in [("CV", "c_v"), ("NPMI", "c_npmi"), ("UCI", "c_uci"), ("Umass", "u_mass")]:
        m[key] = _safe(
            lambda measure=measure: Coherence(
                texts=train_corpus, topk=effective_topk,
                measure=measure, processes=1,
            ).score(out),
            key,
        )

    # ── Diversity metrics ──────────────────────────────────────────────────────
    m["TD"]   = _safe(lambda: TopicDiversity(topk=effective_topk).score(out), "TD")
    m["IRBO"] = _safe(lambda: InvertedRBO(topk=effective_topk).score(out), "IRBO")

    # ── Clustering: NMI and Purity ─────────────────────────────────────────────
    labels = dataset.get_labels()
    tdm    = out.get("topic-document-matrix")
    if labels is not None and tdm is not None:
        last_train = dataset.get_metadata().get("last-training-doc", len(train_corpus))
        nmi, purity = _nmi_purity(tdm, labels, last_train)
    else:
        nmi, purity = float("nan"), float("nan")
    m["NMI"]    = nmi
    m["Purity"] = purity

    # ── Classification: micro-F1 ───────────────────────────────────────────────
    if labels is not None and out.get("test-topic-document-matrix") is not None:
        m["F1"] = _safe(lambda: F1Score(dataset, average="micro").score(out), "F1")
    else:
        m["F1"] = float("nan")

    return m


# ── table formatter ───────────────────────────────────────────────────────────

_COLS = ["CV", "NPMI", "UCI", "Umass", "TD", "IRBO", "NMI", "Purity", "F1"]


def fmt(v):
    try:
        import math
        return f"{v:.4f}" if not math.isnan(v) else "   NaN"
    except Exception:
        return str(v)


def results_to_table(all_results: dict, topk: int = 10) -> str:
    """Format a nested {dataset: {model: (output, dataset)}} dict as a table."""
    rows = []
    for ds_name, model_results in all_results.items():
        for model_name, (model_out, ds) in model_results.items():
            if model_out is None:
                metrics = {k: "FAIL" for k in _COLS}
            else:
                metrics = compute_metrics(model_out, ds, topk=topk)
            rows.append((ds_name, model_name, metrics))

    if not rows:
        return "No results."

    col_header = "  ".join(f"{c:>7}" for c in _COLS)
    header = f"{'Dataset':<16} {'Model':<12}  {col_header}"
    sep = "-" * len(header)
    lines = [header, sep]
    for ds, model, metrics in rows:
        vals = "  ".join(f"{fmt(metrics.get(c, float('nan'))):>7}" for c in _COLS)
        lines.append(f"{ds:<16} {model:<12}  {vals}")
    return "\n".join(lines)