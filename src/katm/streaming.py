"""Streaming building blocks for KATM(streaming=True).

A *corpus* is a list of texts, or a zero-argument callable returning a fresh iterator of texts or
(id, text) pairs (it may be iterated several times, never held in memory as a list).

  pairs_factory                      normalise any corpus into a callable yielding (id, text)
  extract_perdoc_keyphrases_streaming  per-document keyphrases (KeyBERT / RAKE / YAKE / ...) from a stream:
                                     batches of documents go through the SAME KeyphraseExtractor the in-memory
                                     path uses, and only a phrase -> document-frequency Counter is kept,
                                     so memory is bounded by the number of distinct keyphrases.
  extract_content_words_stream       content-word vocabulary from a stream (tfidf / spacy / sampled_pos / type_pos)
  iter_batches                       fixed-size lists of texts from a corpus
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Iterable, Iterator, List, Optional, Tuple


def pairs_factory(corpus) -> Callable[[], Iterator[Tuple[object, str]]]:
    """Corpus -> callable returning a fresh iterator of (id, text); empty texts are skipped."""
    if isinstance(corpus, (str, bytes)):
        raise TypeError("pass a list of texts or a zero-argument callable returning an iterator")
    src = corpus if callable(corpus) else (lambda: iter(corpus))

    def factory():
        for i, item in enumerate(src()):
            if isinstance(item, tuple):
                did, text = item[0], item[1]
            else:
                did, text = i, item
            if text and text.strip():
                yield did, text
    return factory


def iter_batches(factory, batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for _, text in factory():
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_noun_final_vocab_streaming(corpus, language: str, ngram_range: tuple, batch_size: int = 256,
                                     progress_every: int = 0) -> set:
    """Corpus-wide noun-final n-gram vocabulary (utils.build_noun_final_vocab) from a stream.

    KeyBERT's POS-restricted candidates are NOT document-local: an n-gram is a candidate for a document if it
    is a right-headed noun phrase ANYWHERE in the call's documents. Extracting in batches with a per-batch
    vocabulary would therefore change which candidates a document sees. This pass takes the union over the
    whole stream (a set of distinct n-grams, not the corpus), after which every batch is extracted against
    the same vocabulary - identical to the in-memory result. Cost: one spaCy tagging pass over the corpus.
    """
    from .utils import build_noun_final_vocab
    vocab: set = set()
    n, last, t0 = 0, 0, time.perf_counter()
    for batch in iter_batches(pairs_factory(corpus), batch_size):
        vocab |= build_noun_final_vocab(batch, language, ngram_range)
        n += len(batch)
        if progress_every and n - last >= progress_every:
            last = n
            el = time.perf_counter() - t0
            print(f"  [pos vocab] {n} docs, {len(vocab)} noun-final n-grams, {el:.0f}s ({n / el:.1f} docs/s)", flush=True)
    return vocab


def extract_perdoc_keyphrases_streaming(
    corpus, extractor, min_df: int = 2, max_df_ratio: float = 0.4, batch_size: int = 64,
    max_counter_items: int = 20_000_000, progress_every: int = 0, return_info: bool = False,
):
    """Streaming counterpart of KATM's per-document anchor pool.

    In memory:   doc_keyphrases = extractor.extract(all documents); count each phrase once per document;
                 keep min_df <= df <= max_df_ratio * n_docs, ordered by (-df, phrase).
    Streaming:   the same, with `extractor.extract` called on batches of `batch_size` documents. KeyBERT
                 (and the other algorithms) score every document independently, so batching does not change
                 any document's keyphrases; only the Counter (distinct phrases -> df) is accumulated.

    `extractor` is a configured katm.keyphrase_extractor.KeyphraseExtractor. If the Counter grows past
    `max_counter_items`, phrases seen once so far are dropped (a lossy safety valve; a dropped phrase seen
    again later is counted from 1 - irrelevant at the default size, reported in info["pruned"]).
    """
    factory = pairs_factory(corpus)
    doc_freq: Counter = Counter()
    n_docs, pruned, last = 0, 0, 0
    t0 = time.perf_counter()
    for batch in iter_batches(factory, batch_size):
        for kps in extractor.extract(batch):
            doc_freq.update(set(kps))
        n_docs += len(batch)
        if len(doc_freq) > max_counter_items:
            doc_freq = Counter({k: v for k, v in doc_freq.items() if v > 1})
            pruned += 1
        if progress_every and n_docs - last >= progress_every:
            last = n_docs
            el = time.perf_counter() - t0
            print(f"  [keyphrases] {n_docs} docs, {len(doc_freq)} distinct phrases, {el:.0f}s ({n_docs / el:.1f} docs/s)",
                  flush=True)
    max_df = int(max_df_ratio * n_docs) if max_df_ratio else n_docs
    phrases = [kp for kp, f in sorted(doc_freq.items(), key=lambda x: (-x[1], x[0])) if min_df <= f <= max_df]
    info = {"n_docs": n_docs, "n_distinct_phrases": len(doc_freq), "pruned": pruned,
            "doc_freq": {p: doc_freq[p] for p in phrases}, "seconds": time.perf_counter() - t0}
    return (phrases, info) if return_info else phrases


def extract_content_words_stream(corpus, min_df: int = 3, method: str = "tfidf", language: str = "english",
                                 pos_filter: str = None, max_df: float = 1.0) -> List[str]:
    """Streaming dispatcher mirroring utils.extract_content_words(method=...)."""
    from . import fast_content_words as fcw
    from . import utils
    factory = pairs_factory(corpus)
    if language != "english" and method in ("tfidf", "spacy"):
        raise NotImplementedError(f'streaming content_words_method="{method}" supports English only')
    if method == "spacy":
        return utils.extract_content_words_spacy_streaming(factory, min_df=min_df, max_df=max_df, language=language,
                                                           pos_filter=pos_filter)
    if method == "sampled_pos":
        return fcw.extract_content_words_sampled_streaming(factory, min_df=min_df, max_df=max_df, language=language,
                                                           pos_filter=pos_filter)
    if method in ("type_pos", "type_pos_templates"):
        return fcw.extract_content_words_typed_streaming(
            factory, min_df=min_df, max_df=max_df, language=language, pos_filter=pos_filter,
            context="templates" if method == "type_pos_templates" else "none")
    vocab, _ = utils.extract_content_words_streaming(factory, min_df=min_df, language=language)
    return vocab
