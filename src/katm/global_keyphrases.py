"""Corpus-level keyphrase extraction for KATM's keyphrase_scope="global"
option — an alternative to running KeyBERT once per document (the default,
"per_document" path) that instead builds one shared candidate vocabulary
for the whole corpus and ranks it once. Typically 2.5-3.5x faster than
per_document mode on medium/large corpora, since it embeds each unique
candidate once instead of once per document it appears in.

Two things per-document KeyBERT gets "for free" that a single corpus-wide
pass has to earn back explicitly, both handled here:

1. Cross-document frequency filtering. Candidates are built with sklearn's
   CountVectorizer using min_anchor_df/max_anchor_df_ratio as min_df/max_df
   directly, so a candidate must already clear the same document-frequency
   bar KATM's per-document path applies *after* extraction — here it's
   applied *before* any candidate is even embedded, which is also what
   keeps global mode fast.

2. Local-relevance ranking, not similarity to one global reference vector.
   A candidate is ranked by its top-k average similarity to its k best-
   matching documents (see top_k_docs below), not by similarity to a single
   corpus-wide vector (a truncated concatenated-corpus string, or the mean
   of all document embeddings) — either of those was found to systematically
   favor generic discourse-register words mildly related to every document
   over subject-distinctive words strongly related to only a subset.
   ranking_mode="membership" is a second, opt-in ranking strategy for corpora
   where even top-k averaging under-serves a real but narrow subtopic — see
   its own docstring below.

A document's own relevance to a candidate is computed as the MAXIMUM
similarity across that document's own chunks (via
keyphrase_extractor._chunk_text_for_embedding — the same helper
long_document_strategy="chunk" uses on the per-document path), not a single
whole-document embedding, so a long document isn't silently truncated at the
embedding model's max_seq_length before it's ever compared against
candidates. Short documents that already fit come back as a single chunk (a
strict no-op).
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
) -> List[str]:
    """
    Args:
        documents: Corpus documents.
        embedder: A fitted/loaded SentenceEmbedder (reused so this doesn't
            trigger a second model load).
        keybert_ngram_range: n-gram range for candidate generation — same
            parameter KATM's per-document KeyBERT path uses.
        language: "english" (default elsewhere in KATM) or any other
            language code — selects the stopword list via
            utils._stopwords_for.
        min_anchor_df, max_anchor_df_ratio: Same document-frequency bounds
            KATM's per-document anchor pool applies, used here as
            CountVectorizer's min_df/max_df directly.
        n_keyphrases_total: Total keyphrase budget extracted once from the
            whole corpus.
        top_k_docs: Number of documents averaged per candidate in the local-
            relevance ranking, only used when ranking_mode="top_k_average".
            None (default) uses the auto formula max(5, min(50, n_docs //
            20)). A candidate with fewer than top_k_docs documents genuinely
            similar to it gets diluted by forced inclusion of weak matches
            just to fill the averaging window — lowering top_k_docs shrinks
            how much "real" support a candidate needs, at the cost of
            noisier estimates from fewer documents.
        ranking_mode: "top_k_average" (default) or "membership" — see
            membership_n_per_doc below for what the latter changes.
        membership_n_per_doc: Only used when ranking_mode="membership" - the
            size of each document's own top-N window when counting
            membership votes (see below), mirroring per-document mode's own
            n_keyphrases default of 10.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from .utils import _stopwords_for as _kp_stopwords_for
    kp_stop_words = "english" if language == "english" else list(_kp_stopwords_for(language))

    n_docs = len(documents)
    min_df = max(1, min_anchor_df) if min_anchor_df else 1
    max_df = max_anchor_df_ratio if max_anchor_df_ratio else 1.0
    # CountVectorizer requires min_df as an int <= n_docs; guard against a
    # min_anchor_df larger than the corpus itself (tiny corpora).
    min_df = min(min_df, n_docs)

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

    # Chunk each document (sentence-packed, token-budget-limited — see module
    # docstring) so no document is silently truncated at the embedding
    # model's max_seq_length before it's compared against candidates.
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
    # it is relevant to ANY of that document's chunks, not the average - a
    # merged multi-part document's chunks can be about genuinely different
    # sub-topics. chunk_counts are in document order, so reduceat's segment
    # boundaries line up with all_chunks' document boundaries directly.
    starts = np.cumsum([0] + chunk_counts[:-1])
    sim_matrix = np.maximum.reduceat(chunk_sim, starts, axis=1)  # (n_candidates, n_docs)

    if ranking_mode == "membership":
        # Ports per-document mode's actual mechanism onto the shared,
        # frequency-filtered candidate vocabulary: for each document, find
        # its own top-N candidates (from sim_matrix, already max-over-chunks
        # per document above), then rank corpus-wide candidates by how many
        # documents' own top-N they appear in - a vote/membership count, not
        # an average, so a candidate that's the best thing in just a couple
        # of documents can still win, unlike top_k_average which needs
        # roughly top_k_docs genuinely well-matching documents to avoid its
        # average being dragged down by forced inclusion of weak ones. Not
        # vectorized across documents on purpose - the O(n_candidates x
        # n_docs) argpartition work dominates regardless of loop structure,
        # and two array-level rewrites benchmarked slower at scale (strided
        # access / transpose-copy cost) than this simple per-document loop.
        membership_count = np.zeros(len(candidates), dtype=int)
        n_eff = min(membership_n_per_doc, len(candidates))
        for d in range(n_docs):
            col = sim_matrix[:, d]
            top_n_idx = np.argpartition(col, -n_eff)[-n_eff:]
            membership_count[top_n_idx] += 1
        # Tie-break by mean relevance across all documents - membership count
        # alone has many ties among candidates that never made any document's
        # top-N (all at 0), so this keeps the ranking meaningful past the
        # candidates that actually won a vote.
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
