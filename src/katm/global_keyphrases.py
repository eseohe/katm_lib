"""Shared global-scope keyphrase extraction for KATM's optional
keyphrase_scope="global" parameter. Used by both topic_model.py (KATM) and
topic_model_global_exp.py (KATMGlobal) so the two never carry independent,
driftable copies of the same logic.

Two correctness fixes over the first working version of this idea (and over
the original standalone prototype, gemini/sdtm/katm_global.py) - both found
and validated on real benchmark datasets, documented in full with before/
after evidence in sdtm/KATM_GLOBAL_REPORT.md:

1. Cross-document frequency validation. Candidates are built with
   sklearn's CountVectorizer using min_anchor_df/max_anchor_df_ratio as
   min_df/max_df directly, so a candidate must clear the same cross-
   document-frequency bar per-document mode enforces before it is even
   embedded - not filtered after the fact, and not skipped entirely (the
   original prototype had no equivalent of this filter at all, letting
   one-off named entities/hashtags into the anchor pool on short,
   heterogeneous text).

2. Local-relevance ranking, not similarity to one global reference vector.
   Candidates are ranked by top-k average similarity to their k best-
   matching documents, not by cosine similarity to a single reference
   vector - whether a truncated concatenated-corpus string (silently only
   reflecting the embedding model's max-sequence-length worth of the
   corpus, i.e. a handful of documents) or the mean of all document
   embeddings (tried first here; confirmed on 20 Newsgroups to
   systematically favor generic discourse-register words mildly related to
   every document - believe, doubt, dispute, cheap - over subject-
   distinctive words strongly related to only a subset - ford, encryption,
   hockey). Top-k ranking rewards a candidate only if it is strongly
   related to a meaningful cluster of documents, which is what per-document
   KeyBERT already gets for free.

A third fix, found later, on real per-class mental-health data (single-
subreddit corpora - every document already about the same broad condition):
fix 2 changed WHO a candidate is compared against (top-k best-matching
documents, not one blended reference vector), but each of those per-
document comparisons still used a single whole-document embedding - which
SentenceTransformer.encode() silently truncates at max_seq_length, the same
bug keyphrase_scope="per_document"'s long_document_strategy="chunk" fixes
on the OTHER code path (keyphrase_extractor.py). A candidate like "xanax
anxiety" was correctly generated from the full, untruncated document text
(CountVectorizer doesn't truncate), but if that document's Xanax mention
lived past token 256 of its ~1700-token merged-posts document, the
embedding it was scored against never reflected it - so genuinely relevant
documents scored as irrelevant, and the whole medication substrand was
starved down to a handful of anchors, never enough for GMM to dedicate a
component to (confirmed directly: only 5-13 of ~950 raw medication-related
candidates survived to the final anchor pool on the anxiety/bipolar
per-class corpora, none concentrated enough to form their own topic).
Fixed the same way as the per-document path: each document is split into
sentence-packed chunks that fit within max_seq_length (a no-op single
chunk when it already fits), and a candidate's relevance to a document is
the MAXIMUM similarity across that document's own chunks, not a single
possibly-truncated whole-document vector.

A fourth issue, deeper than fixes 2/3 above and found by directly asking
why keyphrase_scope="global" still underperforms keyphrase_scope=
"per_document" on the bipolar per-class corpus even after all three fixes:
fix 2's top-k-average ranking is not actually the corpus-wide equivalent of
what per-document mode does. Per-document mode's "local relevance" is a
within-one-document operation - a candidate just needs to be one of that
document's own best-N candidates, with zero cross-document comparison.
Top-k-average instead requires roughly k genuinely well-matching documents
to avoid the average being dragged down by forced inclusion of weak ones
(see top_k_docs's docstring above) - structurally different from "was this
locally salient somewhere", and biased toward broad-but-moderate relevance
over narrow-but-strong relevance.

ranking_mode="membership" ports per-document mode's actual mechanism onto
the shared, frequency-filtered candidate vocabulary instead: for each
document, find its own top-N candidates (by the same per-document,
max-over-chunks relevance already computed for fix 3), then rank corpus-wide
candidates by how many documents' own top-N they appear in - a vote/
membership count, not an average, so a candidate that's the best thing in
just 2-3 documents can still win, exactly as it would under per-document
mode's own doc_freq pooling. Measured directly on the bipolar per-class
corpus: medication-candidate survival into the top-500 nearly triples
(top_k_average: 19/1024, 1.9% -> membership: 57/1024, 5.6%), and the best
achievable GMM-cluster purity roughly doubles (15% -> 35%). Default remains
"top_k_average" (unchanged behavior) since ranking_mode="membership" was
only validated on the per-class mental-health corpora, not the three
datasets fix 1/2/3 were validated across - opt-in until checked more
broadly.
"""
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import SentenceEmbedder


def extract_global_keyphrases(
    documents: List[str],
    embedder: SentenceEmbedder,
    keybert_ngram_range: tuple,
    language: str,
    min_anchor_df: int,
    max_anchor_df_ratio: float,
    n_keyphrases_total: int,
    top_k_docs: "int | None" = None,
    ranking_mode: str = "top_k_average",
    membership_n_per_doc: int = 10,
    candidate_pos_filter: bool = False,
    exclude_abstract_suffixes: bool = False,
) -> List[str]:
    """
    exclude_abstract_suffixes: False (default). Only has any effect when
        candidate_pos_filter=True and language="german" - passed straight
        through to utils.build_noun_final_vocab (see its own docstring):
        additionally drops German candidates headed by a word ending in a
        productive abstract-nominalization suffix (-ung/-heit/-keit/-nis/
        -schaft/-tum). Experimental - testing whether German's specific
        weakness (candidates skewing toward abstract emotion/cognition
        nouns more than English's, on matched novels - see project notes)
        responds to a language-specific tightening on top of the plain
        noun-final filter.
    candidate_pos_filter: False (default, unchanged behavior - plain
        CountVectorizer candidates, stop-word-filtered only). If True,
        restricts the candidate pool itself to right-headed noun phrases
        (last token NOUN/PROPN) before ranking - the global-scope test of
        keyphrase_scope="per_document"'s own keybert_candidate_pos_filter
        fix (see utils.build_noun_final_vocab and keyphrase_extractor
        .KeyphraseExtractor._noun_final_candidate_vectorizer's docstring
        for the original diagnosis: grammatical/register-class topics -
        e.g. all speech verbs - forming because the anchor pool itself was
        full of non-noun-final candidates, not just the displayed word
        list). Opt-in, exploratory - not yet validated the way the other
        global-mode fixes in this module are; see gemini/subtopic_discovery
        prose-experiment notes for the literary-prose test this was built
        for. min_anchor_df/max_anchor_df_ratio are still applied, just by
        hand on the resulting matrix rather than via CountVectorizer's own
        min_df/max_df (which are silently ignored whenever a fixed
        `vocabulary=` is supplied).
    ranking_mode: "top_k_average" (default, unchanged behavior) or
        "membership" - see module docstring's fourth-fix section for the
        full mechanism and the bipolar evidence. "membership" ignores
        top_k_docs entirely (there is no averaging window in that mode).
    membership_n_per_doc: Only used when ranking_mode="membership" - the
        size of each document's own top-N window when counting membership
        votes, mirroring per-document mode's own n_keyphrases (default 10,
        matching per-document mode's own default).
    top_k_docs: Number of documents averaged per candidate in the local-
        relevance ranking (see module docstring, fix #2). None (default)
        keeps the original auto formula, max(5, min(50, n_docs // 20)) -
        byte-identical behavior to before this parameter existed.

        This average is a fixed-window operation: a candidate with fewer
        than top_k_docs documents genuinely similar to it (diagnosed
        directly on the bipolar per-class corpus - e.g. "depakote" has ZERO
        documents above 0.5 cosine similarity, yet the top-25 average still
        has to be filled from somewhere) gets diluted by forced inclusion of
        weak/irrelevant documents, dragging its score down toward however
        many weak matches exist - NOT toward how strongly it matches where
        it IS actually discussed. A near-universal condition-identity bigram
        (e.g. "bipolar disorder", relevant to the majority of documents in a
        single-condition subreddit corpus) never hits this floor, so it
        keeps a high, undiluted score - systematically outcompeting rarer
        but genuinely on-topic candidates (specific medication names) for
        the fixed n_keyphrases_total slots. Lowering top_k_docs shrinks the
        window a candidate needs "real" support across, at the cost of
        noisier estimates from very few documents; this parameter exists so
        that trade-off can be swept and evaluated per-corpus rather than
        living inside one hardcoded formula.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from .utils import _stopwords_for as _kp_stopwords_for
    kp_stop_words = "english" if language == "english" else list(_kp_stopwords_for(language))

    n_docs = len(documents)
    min_df = max(1, min_anchor_df) if min_anchor_df else 1
    max_df = max_anchor_df_ratio if max_anchor_df_ratio else 1.0
    # CountVectorizer requires min_df as an int <= n_docs; guard against a
    # min_anchor_df larger than the corpus itself (tiny test corpora).
    min_df = min(min_df, n_docs)

    if candidate_pos_filter:
        # Opt-in test of the per-document path's own fix (see
        # keyphrase_extractor.KeyphraseExtractor._noun_final_candidate_
        # vectorizer / utils.build_noun_final_vocab): restrict the candidate
        # pool itself to right-headed noun phrases before anything is
        # ranked, instead of the plain CountVectorizer's stop-word-only
        # filter below. A fixed `vocabulary=` bypasses sklearn's own min_df/
        # max_df pruning entirely, so that filter is re-applied by hand here
        # on the resulting document-term matrix - same threshold semantics
        # (min_df as an absolute doc count, max_df as a ratio) as the
        # unrestricted branch just above.
        from .utils import build_noun_final_vocab
        vocab = build_noun_final_vocab(documents, language, keybert_ngram_range,
                                        exclude_abstract_suffixes=exclude_abstract_suffixes)
        if not vocab:
            return []  # no noun-final candidates found; caller sees an empty pool
        cv = CountVectorizer(vocabulary=sorted(vocab), ngram_range=keybert_ngram_range,
                              token_pattern=r"(?u)\b\w+\b")
        counts = cv.fit_transform(documents)
        doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
        max_df_count = max_df * n_docs
        keep = (doc_freq >= min_df) & (doc_freq <= max_df_count)
        candidates = [w for w, k in zip(cv.get_feature_names_out(), keep) if k]
        if not candidates:
            return []
    else:
        cv = CountVectorizer(
            ngram_range=keybert_ngram_range, stop_words=kp_stop_words,
            min_df=min_df, max_df=max_df, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        try:
            cv.fit(documents)
        except ValueError:
            return []  # empty vocabulary after frequency filtering
        candidates = list(cv.get_feature_names_out())
        if not candidates:
            return []

    # Chunk each document (sentence-packed, token-budget-limited, exactly the
    # same helper long_document_strategy="chunk" uses on the per-document
    # path) so no document is silently truncated at the embedding model's
    # max_seq_length before it is compared against candidates. Short
    # documents that already fit come back as a single chunk (a strict
    # no-op), so this never changes behavior on short-document corpora.
    from .keyphrase_extractor import _chunk_text_for_embedding
    st_model = embedder._model
    tokenizer = st_model.tokenizer
    max_tokens = getattr(st_model, "max_seq_length", 256) or 256

    doc_chunks = [_chunk_text_for_embedding(doc, tokenizer, max_tokens) for doc in documents]
    chunk_counts = [len(chunks) for chunks in doc_chunks]
    all_chunks = [c for chunks in doc_chunks for c in chunks]

    chunk_embeddings = embedder.encode(all_chunks, batch_size=32)
    candidate_embeddings = embedder.encode(candidates, batch_size=64)

    chunk_sim = cosine_similarity(candidate_embeddings, chunk_embeddings)  # (n_candidates, n_chunks)
    # Max-over-chunks per document: a candidate is relevant to a document if
    # it is relevant to ANY of that document's chunks, not the average -
    # a merged multi-post document's chunks can be about genuinely different
    # sub-topics (e.g. one post about medication, another about therapy).
    # chunk_counts are in document order, so reduceat's segment boundaries
    # line up with all_chunks' document boundaries without any extra bookkeeping.
    starts = np.cumsum([0] + chunk_counts[:-1])
    sim_matrix = np.maximum.reduceat(chunk_sim, starts, axis=1)  # (n_candidates, n_docs)

    if ranking_mode == "membership":
        # Per-document top-N membership count (see module docstring's fourth
        # fix): for each document, find its own top-N candidates from
        # sim_matrix (already max-over-chunks per document, from fix 3),
        # then rank corpus-wide by how many documents' own top-N a candidate
        # appears in - a vote count, not an average, so no candidate needs
        # k genuinely-strong-matching documents the way top_k_average does.
        # NOT vectorized across documents on purpose - tried two array-level
        # rewrites (whole-matrix argpartition(axis=0) in one call; transpose
        # to a contiguous (n_docs, n_candidates) layout first, then
        # argpartition(axis=1)) and benchmarked both against this loop at
        # 20-Newsgroups scale (40k candidates x 4k docs): axis=0 on the
        # non-contiguous column layout was 2.3x SLOWER (cache-unfriendly
        # strided access), and the transpose was a wash (its own copy cost
        # what it saved). argpartition is already an O(n) numpy/C routine
        # per call; the real cost is the O(n_candidates x n_docs) work
        # itself, which every rewrite still has to do - the Python loop's
        # overhead across n_docs calls is negligible next to that, so there
        # was no slack to vectorize away. Kept as the simplest form that
        # ties the fastest option found.
        membership_count = np.zeros(len(candidates), dtype=int)
        n_eff = min(membership_n_per_doc, len(candidates))
        for d in range(n_docs):
            col = sim_matrix[:, d]
            top_n_idx = np.argpartition(col, -n_eff)[-n_eff:]
            membership_count[top_n_idx] += 1
        # Tie-break by mean relevance across all documents - membership
        # count alone has many ties among candidates that never made any
        # document's top-N (all at 0), so this keeps the ranking meaningful
        # past the candidates that actually won a vote.
        mean_score = sim_matrix.mean(axis=1)
        order = np.lexsort((-mean_score, -membership_count))
        top_idx = order[:n_keyphrases_total]
    elif ranking_mode == "top_k_average":
        k = top_k_docs if top_k_docs is not None else max(5, min(50, n_docs // 20))
        k = max(1, min(k, n_docs))  # guard against k <= 0 or k > n_docs
        top_k_sims = np.sort(sim_matrix, axis=1)[:, -k:]
        local_relevance = top_k_sims.mean(axis=1)
        top_idx = np.argsort(local_relevance)[::-1][:n_keyphrases_total]
    else:
        raise ValueError(
            f"Unknown ranking_mode {ranking_mode!r} - must be 'top_k_average' or 'membership'"
        )

    return [candidates[i] for i in top_idx]
