"""Fast POS-filtered content-word vocabulary ("sampled_pos") for KATM.

Why: content_words_method="spacy" tags EVERY token of the corpus (12.7M tokens for ELTeC-100,
~11 min on 2 cores) although the output is only a word LIST of ~10-20k types. The decision
"is this word a common noun / adjective / finite verb?" is a property of the word type that a
handful of tagged occurrences settles. So:

  1. count candidate types once (CountVectorizer, lowercase; >=2 letters; df >= min_df;
     stop words removed) - same cost as method="tfidf";
  2. collect up to `samples_per_word` occurrences of every candidate as short token windows
     (+-`window` tokens around the word) - one regex pass over the corpus, stops early once
     every word has its quota;
  3. POS-tag ONLY those windows with spaCy (tagger + attribute_ruler + lemmatizer; parser/NER
     off) - typically 10-15% of the corpus tokens;
  4. drop words capitalised in >= `max_cap_ratio` of their non-sentence-initial occurrences
     (names; English/French only - disabled for German);
  5. group words by their (sampled) spaCy lemma and keep a lemma if the ESTIMATED number of
     documents where it occurs with a passing tag (sum over its forms of df x sampled pass
     share) reaches min_df - the same criterion method="spacy" applies with exact counts.
     POS filter: paper_restricted = NN/NNS/JJ/JJR/JJS + finite lexical verbs VBZ/VBP/VBD
     (English; German/French tags are mapped by utils._passes_paper_restricted_filter);
     None = broad NOUN/VERB/ADJ/ADV/PROPN. Returns the most frequent passing surface form per
     lemma (method="spacy" output convention).

ELTeC-100 (25,411 passages, min_df=20, 2 CPU cores): 47 s vs 653 s for method="spacy";
97% of its words are in the spaCy vocabulary, it recovers 89% of the spaCy vocabulary, and
KATM topics share ~79% of their top-10 words with the spaCy-vocabulary topics (removing a
random 10% of the spaCy vocabulary itself gives 80-89%). See experiments_fast_content_words.py.

The capitalisation test in (4) is free (computed during the regex pass) and catches proper
nouns the tagger mislabels as NN in short windows.
"""
from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

_TOK = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*|[.!?;:,\"]", re.UNICODE)
_SENT_END = {".", "!", "?", '"'}
_PR_NOUN_ADJ = {"NN", "NNS", "JJ", "JJR", "JJS"}
_PR_FIN = {"VBZ", "VBP", "VBD"}
_AUX = {"be", "have", "do"}
_BROAD = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def _passes(tok, pos_filter, language="english"):
    if pos_filter == "paper_restricted" and language != "english":
        from .utils import _passes_paper_restricted_filter
        return _passes_paper_restricted_filter(tok, language)
    if pos_filter == "paper_restricted":
        if tok.tag_ in _PR_NOUN_ADJ:
            return True
        return tok.tag_ in _PR_FIN and tok.pos_ != "AUX" and tok.lemma_.lower() not in _AUX
    return tok.pos_ in _BROAD


def extract_content_words_sampled(texts: List[str], min_df: int = 3, language: str = "english",
                                  pos_filter: str = "paper_restricted", max_df: float = 1.0,
                                  samples_per_word: int = 5, window: int = 8, min_pass_share: float = 0.5,
                                  max_cap_ratio: float = None, min_len: int = 2, seed: int = 0, decision: str = "lemma_df",
                                  return_info: bool = False):
    import spacy
    from sklearn.feature_extraction.text import CountVectorizer
    from .utils import _stopwords_for, _SPACY_MODEL

    if max_cap_ratio is None:   # German capitalises every noun -> capitalisation says nothing about names
        max_cap_ratio = 1.01 if language == "german" else 0.7
    t0 = time.perf_counter()
    stop = _stopwords_for(language)
    n = len(texts)
    cv = CountVectorizer(lowercase=True, min_df=min(min_df, n), max_df=max_df if max_df < 1.0 else 1.0,
                         token_pattern=r"(?u)\b[^\W\d_]{%d,}\b" % min_len, binary=True, dtype="int32")
    X = cv.fit_transform(texts)
    vocab = cv.get_feature_names_out()
    df = dict(zip(vocab, X.sum(axis=0).A1))
    cand = {w for w in vocab if w not in stop and not re.search(r"(.)\1{2,}", w) and len(w) <= 25}
    t_count = time.perf_counter() - t0

    # ---- one pass: sample windows + capitalisation statistics (non-sentence-initial)
    t0 = time.perf_counter()
    need = {w: samples_per_word for w in cand}
    caps, mids = Counter(), Counter()
    windows: List[Tuple[List[str], int, str]] = []
    order = list(range(n))
    random.Random(seed).shuffle(order)
    for di in order:
        toks = _TOK.findall(texts[di])
        prev = "."
        for i, tk in enumerate(toks):
            lw = tk.lower()
            if lw in cand:
                if prev not in _SENT_END:
                    mids[lw] += 1
                    if tk[0].isupper():
                        caps[lw] += 1
                if need.get(lw, 0) > 0:
                    need[lw] -= 1
                    if need[lw] == 0:
                        del need[lw]
                    lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                    windows.append((toks[lo:hi], i - lo, lw))
            prev = tk
        # capitalisation needs enough evidence, sampling only needs the quota: stop when both
        # are satisfied for (almost) every word
        if not need and sum(mids.values()) > 20 * len(cand):
            break
    t_sample = time.perf_counter() - t0

    # ---- tag only the windows
    t0 = time.perf_counter()
    from spacy.tokens import Doc
    nlp = spacy.load(_SPACY_MODEL.get(language, language), disable=["parser", "ner"])
    docs = (Doc(nlp.vocab, words=w) for w, _, _ in windows)
    passed, seen = Counter(), Counter()
    lemma_of: Dict[str, Counter] = defaultdict(Counter)
    for (words, idx, lw), doc in zip(windows, nlp.pipe(docs, batch_size=512)):
        tok = doc[idx]
        seen[lw] += 1
        passed[lw] += int(_passes(tok, pos_filter, language))
        lemma_of[lw][tok.lemma_.lower()] += 1
    t_tag = time.perf_counter() - t0

    # ---- decide + lemma grouping (most frequent surface form per lemma)
    # decision="lemma_df" mirrors method="spacy": a lemma is kept if the ESTIMATED number of
    # documents in which it occurs with a passing tag (sum over its surface forms of
    # df * sampled pass share) reaches min_df. decision="majority": each surface form on its own,
    # kept if >= min_pass_share of its sampled occurrences pass.
    def capitalised(w):
        return mids[w] >= 3 and caps[w] / mids[w] >= max_cap_ratio
    groups: Dict[str, List[str]] = defaultdict(list)
    for w in cand:
        if seen[w] and not capitalised(w):
            groups[lemma_of[w].most_common(1)[0][0]].append(w)
    out = []
    for lem, ws in groups.items():
        if lem in stop:
            continue
        if decision == "lemma_df":
            est = sum(df[w] * passed[w] / seen[w] for w in ws)
            ok = [w for w in ws if passed[w]]
            if est >= min_df and ok:
                out.append(max(ok, key=lambda x: df[x]))
        else:
            ok = [w for w in ws if passed[w] / seen[w] >= min_pass_share]
            if ok:
                out.append(max(ok, key=lambda x: df[x]))
    out = sorted(out)
    info = {"n_candidates": len(cand), "n_windows": len(windows),
            "window_tokens": sum(len(w) for w, _, _ in windows),
            "timings": {"count": t_count, "sample": t_sample, "tag": t_tag}}
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------------------------
# Streaming counterpart of extract_content_words_sampled, for corpora too large to hold in
# memory (SPGC, PG-19) - see flake/stream.py, whose
# contextual_embed_streaming this mirrors.
# ---------------------------------------------------------------------------------------------
def extract_content_words_sampled_streaming(
    corpus, min_df: int = 3, max_df: float = 1.0, language: str = "english",
    pos_filter: str = "paper_restricted", samples_per_word: int = 5, window: int = 8,
    min_pass_share: float = 0.5, max_cap_ratio: float = None, min_len: int = 2, seed: int = 0,
    decision: str = "lemma_df", batch_size: int = 512, return_info: bool = False,
):
    """Streaming counterpart of extract_content_words_sampled (above) - same method
    ("sampled_pos": POS-tag a few sampled occurrence windows per word type instead of
    every token of the corpus), same output convention, but built from the corpus as a
    stream rather than an in-memory list of texts.

    What has to change, and what doesn't:

    - Counting candidates (step 1) is the same streaming document-frequency counter
      already used by katm.utils.extract_content_words_streaming - no new
      mechanism, just reused.
    - Collecting sample windows (step 2) CANNOT reuse the in-memory version's own
      mechanism unchanged. That version gets an unbiased sample by shuffling the
      entire document list in memory first, then scanning in that random order with
      early-stop once every word has its quota - the shuffle is exactly the thing
      that requires holding the whole corpus in memory, which is the problem this
      function exists to avoid. The actual goal - "a handful of a word's occurrences,
      sampled fairly across the WHOLE corpus, not just whatever is read first" - is
      the identical problem contextual_embed_streaming already solved for anchor
      embedding, with reservoir sampling (Algorithm R): keep the first
      `samples_per_word` occurrences seen, then for the k-th occurrence after that,
      replace a uniformly random already-kept slot with probability
      samples_per_word / k. This gives a provably uniform sample over an arbitrarily
      large, single-pass stream without ever shuffling or holding the corpus at all -
      arguably a more principled mechanism than the in-memory version's shuffle, not
      merely a workaround for it. Capitalisation/mid-sentence statistics (for the
      proper-noun filter) are accumulated in the SAME single pass, at no extra cost.
    - No early-stopping once quotas are filled (the in-memory version has this as a
      speed optimization): stopping early would silently reintroduce the exact
      early-shard bias reservoir sampling exists to avoid, so this always reads the
      full corpus once for this pass - consistent with every other streaming pass in
      this project (e.g. the ~32-40s unigram count over 2,488 PG-19 books), not a
      regression specific to this function.
    - Tagging the collected windows (step 3) and the capitalisation-filter + lemma-
      grouping + decision logic (steps 4-5) are pure in-memory work over an already-
      bounded collection (len(candidates) x samples_per_word windows, independent of
      corpus size) - unchanged from the in-memory version.

    Args: same meaning as extract_content_words_sampled, except `corpus` replaces
        `texts`: the same Corpus type every function in
        streaming.pairs_factory accepts - a callable returning a fresh
        (id, text) iterator (or iter_text_files/iter_parquet_texts), called twice
        (once to count, once to sample - the same re-iterable contract pass 1a/1b
        already rely on).

    Returns: (vocab, info) if return_info, else vocab - same convention as
        extract_content_words_sampled. info adds "n_docs" (documents streamed).
    """
    import random
    from .utils import _stopwords_for, _SPACY_MODEL
    from .streaming import pairs_factory as _as_factory

    if max_cap_ratio is None:
        max_cap_ratio = 1.01 if language == "german" else 0.7

    t0 = time.perf_counter()
    stop = _stopwords_for(language)
    token_re = re.compile(r"\b[^\W\d_]{%d,}\b" % min_len, re.UNICODE)
    bad_token = re.compile(r"(.)\1{2,}")

    doc_freq: Counter = Counter()
    factory = _as_factory(corpus)
    n_docs = 0
    for _, text in factory():
        n_docs += 1
        doc_freq.update({w.lower() for w in token_re.findall(text)})
    t_count = time.perf_counter() - t0

    min_df_eff = min(min_df, n_docs)
    max_df_count = max_df * n_docs if max_df < 1.0 else n_docs
    cand = {w for w, c in doc_freq.items()
           if min_df_eff <= c <= max_df_count and w not in stop
           and not bad_token.search(w) and len(w) <= 25}
    df = {w: doc_freq[w] for w in cand}

    t0 = time.perf_counter()
    rng = random.Random(seed)
    caps, mids, seen_count = Counter(), Counter(), Counter()
    reservoirs: Dict[str, list] = {}
    for _, text in factory():
        toks = _TOK.findall(text)
        prev = "."
        for i, tk in enumerate(toks):
            lw = tk.lower()
            if lw in cand:
                if prev not in _SENT_END:
                    mids[lw] += 1
                    if tk[0].isupper():
                        caps[lw] += 1
                seen_count[lw] += 1
                lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                item = (toks[lo:hi], i - lo)
                bucket = reservoirs.setdefault(lw, [])
                if len(bucket) < samples_per_word:
                    bucket.append(item)
                else:
                    j = rng.randint(0, seen_count[lw] - 1)
                    if j < samples_per_word:
                        bucket[j] = item
            prev = tk
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load(_SPACY_MODEL.get(language, language), disable=["parser", "ner"])
    windows = [(w, idx, toks) for w, bucket in reservoirs.items() for toks, idx in bucket]
    docs = (Doc(nlp.vocab, words=ws) for _, _, ws in windows)
    passed, seen = Counter(), Counter()
    lemma_of: Dict[str, Counter] = defaultdict(Counter)
    for (w, idx, _), doc in zip(windows, nlp.pipe(docs, batch_size=batch_size)):
        tok = doc[idx]
        seen[w] += 1
        passed[w] += int(_passes(tok, pos_filter, language))
        lemma_of[w][tok.lemma_.lower()] += 1
    t_tag = time.perf_counter() - t0

    def capitalised(w):
        return mids[w] >= 3 and caps[w] / mids[w] >= max_cap_ratio

    groups: Dict[str, List[str]] = defaultdict(list)
    for w in cand:
        if seen[w] and not capitalised(w):
            groups[lemma_of[w].most_common(1)[0][0]].append(w)
    out = []
    for lem, ws in groups.items():
        if lem in stop:
            continue
        if decision == "lemma_df":
            est = sum(df[w] * passed[w] / seen[w] for w in ws)
            ok = [w for w in ws if passed[w]]
            if est >= min_df and ok:
                out.append(max(ok, key=lambda x: df[x]))
        else:
            ok = [w for w in ws if passed[w] / seen[w] >= min_pass_share]
            if ok:
                out.append(max(ok, key=lambda x: df[x]))
    out = sorted(out)
    info = {"n_candidates": len(cand), "n_docs": n_docs, "n_windows": len(windows),
           "timings": {"count": t_count, "sample": t_sample, "tag": t_tag}}
    return (out, info) if return_info else out


# ---------------------------------------------------------------------------------------------
# content_words_method="type_pos": tag every unique candidate word ONCE, without its contexts.
# ---------------------------------------------------------------------------------------------
_TEMPLATES = ("{w}", "It was {w} .", "They {w} it .", "the {w} of")


def extract_content_words_typed_streaming(
    corpus, min_df: int = 3, max_df: float = 1.0, language: str = "english",
    pos_filter: str = "paper_restricted", context: str = "none", max_cap_ratio: float = None,
    min_len: int = 2, return_info: bool = False,
):
    """Streaming counterpart of extract_content_words_typed (below) -
    "type_pos": tag each unique candidate word ONCE, in isolation (context="none",
    just the word alone as a one-token document) or in 4 neutral template frames
    (context="templates"), never touching the word's real corpus occurrences at all.

    This one needs even less new engineering than extract_content_words_sampled_
    streaming: the tagging step was ALREADY corpus-size-independent (it costs
    O(vocabulary size), never O(corpus size), in both the in-memory and streaming
    versions) - the only two corpus-dependent parts are candidate counting (reused,
    same streaming counter as extract_content_words_streaming) and the optional
    capitalisation/mid-sentence pass used by the proper-noun filter (one streaming
    scan, reusing the same per-word bookkeeping as extract_content_words_sampled_
    streaming, just without collecting any context windows - pure counting, cheaper
    than the sampled_pos pass since there is nothing to reservoir-sample).

    Args: same meaning as extract_content_words_typed, except `corpus` replaces
        `texts` (same Corpus type as every function in
        streaming.pairs_factory - a callable returning a fresh
        (id, text) iterator, called twice: once to count, once for capitalisation
        stats).

    Returns: (vocab, info) if return_info, else vocab - same convention as
        extract_content_words_typed. info adds "n_docs".
    """
    from .utils import _stopwords_for, _SPACY_MODEL
    from .streaming import pairs_factory as _as_factory

    if max_cap_ratio is None:
        max_cap_ratio = 1.01 if language == "german" else 0.7
    if context == "templates" and language != "english":
        raise ValueError('context="templates" is English-only')

    t0 = time.perf_counter()
    stop = _stopwords_for(language)
    token_re = re.compile(r"\b[^\W\d_]{%d,}\b" % min_len, re.UNICODE)
    bad_token = re.compile(r"(.)\1{2,}")

    doc_freq: Counter = Counter()
    factory = _as_factory(corpus)
    n_docs = 0
    for _, text in factory():
        n_docs += 1
        doc_freq.update({w.lower() for w in token_re.findall(text)})
    t_count = time.perf_counter() - t0

    min_df_eff = min(min_df, n_docs)
    max_df_count = max_df * n_docs if max_df < 1.0 else n_docs
    cand = sorted(w for w, c in doc_freq.items()
                 if min_df_eff <= c <= max_df_count and w not in stop
                 and not bad_token.search(w) and len(w) <= 25)
    df = {w: doc_freq[w] for w in cand}

    t0 = time.perf_counter()
    candset = set(cand)
    caps, mids = Counter(), Counter()
    if max_cap_ratio <= 1.0:
        for _, text in factory():
            prev = "."
            for tk in _TOK.findall(text):
                lw = tk.lower()
                if lw in candset and prev not in _SENT_END:
                    mids[lw] += 1
                    caps[lw] += tk[0].isupper()
                prev = tk
    t_caps = time.perf_counter() - t0

    t0 = time.perf_counter()
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load(_SPACY_MODEL.get(language, language), disable=["parser", "ner"])
    frames = _TEMPLATES if context == "templates" else ("{w}",)
    jobs = []
    for w in cand:
        for f in frames:
            words = f.format(w=w).split()
            jobs.append((w, words.index(w), words))
    passed, lemma_of = Counter(), defaultdict(Counter)
    docs = (Doc(nlp.vocab, words=ws) for _, _, ws in jobs)
    for (w, idx, _), doc in zip(jobs, nlp.pipe(docs, batch_size=2048)):
        tok = doc[idx]
        passed[w] += int(_passes(tok, pos_filter, language))
        lemma_of[w][tok.lemma_.lower()] += 1
    t_tag = time.perf_counter() - t0

    groups: Dict[str, List[str]] = defaultdict(list)
    for w in cand:
        if passed[w] and not (mids[w] >= 3 and caps[w] / mids[w] >= max_cap_ratio):
            groups[lemma_of[w].most_common(1)[0][0]].append(w)
    out = sorted(max(ws, key=lambda x: df[x]) for lem, ws in groups.items() if lem not in stop)
    info = {"n_candidates": len(cand), "n_docs": n_docs, "n_tagged_items": len(jobs),
           "timings": {"count": t_count, "caps": t_caps, "tag": t_tag}}
    return (out, info) if return_info else out


def extract_content_words_typed(texts: List[str], min_df: int = 3, language: str = "english",
                                pos_filter: str = "paper_restricted", max_df: float = 1.0,
                                context: str = "none", max_cap_ratio: float = None, min_len: int = 2,
                                return_info: bool = False):
    """Type-level POS: each unique candidate word (df >= min_df) is tagged once, out of context.

    context="none"       the word alone as a one-token document (pure type tagging).
    context="templates"  (English only) the word in 4 neutral frames - alone, "It was _ .",
                         "They _ it .", "the _ of" - kept if it passes the POS filter in ANY frame,
                         i.e. if it CAN be a noun/adjective/finite verb. Lemma = majority over frames.
    Capitalisation (name) filter and lemma grouping as in extract_content_words_sampled; the
    capitalisation statistics need one regex pass over the corpus (no tagging).
    No document frequency per POS is available here, so a lemma is kept if any of its forms
    passes (method="spacy" requires min_df passing documents).
    """
    import spacy
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.tokens import Doc
    from .utils import _stopwords_for, _SPACY_MODEL

    if max_cap_ratio is None:
        max_cap_ratio = 1.01 if language == "german" else 0.7
    if context == "templates" and language != "english":
        raise ValueError('context="templates" is English-only')
    t0 = time.perf_counter()
    stop = _stopwords_for(language)
    n = len(texts)
    cv = CountVectorizer(lowercase=True, min_df=min(min_df, n), max_df=max_df if max_df < 1.0 else 1.0,
                         token_pattern=r"(?u)\b[^\W\d_]{%d,}\b" % min_len, binary=True, dtype="int32")
    X = cv.fit_transform(texts)
    vocab = cv.get_feature_names_out()
    df = dict(zip(vocab, X.sum(axis=0).A1))
    cand = sorted(w for w in vocab if w not in stop and not re.search(r"(.)\1{2,}", w) and len(w) <= 25)
    t_count = time.perf_counter() - t0

    t0 = time.perf_counter()
    candset = set(cand)
    caps, mids = Counter(), Counter()
    if max_cap_ratio <= 1.0:
        for s in texts:
            prev = "."
            for tk in _TOK.findall(s):
                lw = tk.lower()
                if lw in candset and prev not in _SENT_END:
                    mids[lw] += 1
                    caps[lw] += tk[0].isupper()
                prev = tk
    t_caps = time.perf_counter() - t0

    t0 = time.perf_counter()
    nlp = spacy.load(_SPACY_MODEL.get(language, language), disable=["parser", "ner"])
    frames = _TEMPLATES if context == "templates" else ("{w}",)
    jobs = []
    for w in cand:
        for f in frames:
            words = f.format(w=w).split()
            jobs.append((w, words.index(w), words))
    passed, lemma_of = Counter(), defaultdict(Counter)
    docs = (Doc(nlp.vocab, words=ws) for _, _, ws in jobs)
    for (w, idx, _), doc in zip(jobs, nlp.pipe(docs, batch_size=2048)):
        tok = doc[idx]
        passed[w] += int(_passes(tok, pos_filter, language))
        lemma_of[w][tok.lemma_.lower()] += 1
    t_tag = time.perf_counter() - t0

    groups: Dict[str, List[str]] = defaultdict(list)
    for w in cand:
        if passed[w] and not (mids[w] >= 3 and caps[w] / mids[w] >= max_cap_ratio):
            groups[lemma_of[w].most_common(1)[0][0]].append(w)
    out = sorted(max(ws, key=lambda x: df[x]) for lem, ws in groups.items() if lem not in stop)
    info = {"n_candidates": len(cand), "n_tagged_items": len(jobs),
            "timings": {"count": t_count, "caps": t_caps, "tag": t_tag}}
    return (out, info) if return_info else out
