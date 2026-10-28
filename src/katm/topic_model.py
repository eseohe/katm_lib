"""Main topic model module for KATM."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from .clustering import GMMTopicClusterer
from .embedding import SentenceEmbedder
from .keyphrase_extractor import KeyphraseExtractor
from .topic_assignment import WordTopicProjector
from .utils import extract_content_words


_EMBED_CACHE: dict = {}


class _CachedSentenceEmbedder:
    def __init__(self, base):
        self._base, self._model, self.model_name = base, base._model, getattr(base, "model_name", "")

    def encode(self, texts, batch_size=32):
        key = self.model_name
        cache = _EMBED_CACHE.setdefault(key, {})
        miss = [t for t in dict.fromkeys(texts) if t not in cache]
        if miss:
            for t, v in zip(miss, self._base.encode(miss, batch_size=batch_size)):
                cache[t] = v
        return np.stack([cache[t] for t in texts]) if texts else np.array([])


class KATM:
    """Keyphrase Anchored Topic Modeling (KATM) main class."""

    def __init__(
        self,
        kp_algorithm: str = "keybert",
        n_keyphrases: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_topics: Union[int, str] = 10,
        soft_threshold: float = 0.15,
        top_n_words: int = 20,
        min_df: int = 3,
        yake_use_position: bool = False,
        tfidf_ngram_range: tuple = (1, 2),
        keybert_ngram_range: tuple = (1, 2),
        keybert_candidate_pos_filter: bool = False,
        exclude_abstract_suffixes: bool = False,
        content_words_method: str = "tfidf",
        content_pos_filter: str = None,
        normalize_embeddings: bool = False,
        anchor_dedup_threshold: float = 0.85,
        min_anchor_df: int = 2,
        max_anchor_df_ratio: float = 0.4,
        mmr_diversity: float = 0.3,
        mmr_max_sim: float = 0.75,
        lemma_dedup: bool = False,
        centroid_mode: str = "mean",
        centroid_blend_alpha: float = 0.5,
        centroid_top_k: int = 3,
        exclusive_assignment: bool = False,
        keybert_use_mmr: bool = False,
        keybert_diversity: float = 0.5,
        language: str = "english",
        n_topics_range: Tuple[int, int] = (2, 20),
        n_topics_method: str = "bic",
        n_topics_covariance_type: str = "diag",
        gmm_covariance_type: str = "full",
        keyphrase_scope: str = "per_document",
        n_keyphrases_total: int = 500,
        global_top_k_docs: "int | None" = None,
        global_ranking_mode: str = "top_k_average",
        global_membership_n_per_doc: int = 10,
        dim_reduction: Optional[str] = None,
        dim_reduction_n_components: Optional[int] = None,
        long_document_strategy: str = "truncate",
        n_keyphrases_per_chunk: int = 5,
        gsc_candidate_ranking: str = "tfidf",
        gsc_candidate_pool=0.9,
        gsc_chunk_size: int = 200,
        gsc_ngram_range: tuple = (1, 2),
        gsc_dedup: bool = True,
        gsc_method: str = "centroid",
        gsc_batch_size: int = 256,
        contextual_anchor_embedding: bool = False,
        contextual_max_occurrences: int = 20,
        discard_hub_clusters: bool = False,
        hub_overcluster_k: int = 5,
        compute_doc_topic_probs: bool = False,
        flake_kwargs: Optional[dict] = None,
        streaming: bool = False,
        stream_batch_size: int = 64,
        stream_candidate_vocab: str = "global",
        stream_progress_every: int = 0,
        gmm_random_state: int = 42,
        clustering_method: str = "gmm",
        clustering_kwargs: Optional[dict] = None,
        clustering_init: str = "default",
        anchor_isolation_drop_frac: float = 0.0,
        anchor_isolation_k: int = 10,
        anchor_isolation_method: str = "concentration",
        word_scoring: str = "similarity",
        exclude_words: Optional[List[str]] = None,
    ):
        """Initialize KATM model.

        Args:
            kp_algorithm: Keyphrase extraction algorithm ("keybert", "rake", or "yake").
            n_keyphrases: Number of keyphrases to extract per document.
            embedding_model: Sentence-transformer model name.
            n_topics: Number of topics to discover, OR the string "auto" to
                select it automatically from the anchor embeddings during
                fit() (see n_topics_range/n_topics_method). After fit(),
                self.n_topics always holds the actual integer used — when
                "auto" was requested, it is overwritten with the selected
                value and the full per-k diagnostic curve is kept in
                self.n_topics_selection_scores_ (dict with "k"/"bic"/"aic"
                lists) for inspection/plotting.
            n_topics_range: (k_min, k_max) inclusive range to search when
                n_topics="auto". Ignored otherwise. Default (2, 20).
            n_topics_method: "bic" (default), "aic", or "elbow" — see
                clustering.select_n_topics for the reasoning behind each.
                Ignored unless n_topics="auto".
            n_topics_covariance_type: Covariance type for the trial GMMs
                used only during "auto" k-selection (separate from
                gmm_covariance_type below, which governs the actual
                production clustering). Default "diag" — "full" was found
                empirically to make BIC's complexity penalty explode in
                high-dimensional embedding spaces (D~768+), causing it to
                always favor the smallest k in n_topics_range regardless of
                the data. See clustering.select_n_topics for detail.
                Ignored unless n_topics="auto".
            gmm_covariance_type: Covariance type for the production GMM that
                actually clusters the anchor embeddings into topics (see
                GMMTopicClusterer). Default "full". With a large embedding
                dimension (e.g. D=1024) "full" is heavily underdetermined
                against a typical anchor pool; "diag" is far better
                conditioned there and worth trying.
            soft_threshold: Probability threshold for word-topic assignments.
            top_n_words: Number of top words to store per topic.
            min_df: Minimum document frequency for vocabulary.
            yake_use_position: If True, YAKE includes the wpos feature (words
                appearing earlier in a document score higher). Defaults to False
                because position systematically lifts header/boilerplate tokens.
            anchor_dedup_threshold: Cosine similarity threshold for greedy
                semantic deduplication of the anchor pool. Set to None to disable.
            min_anchor_df: Minimum number of documents a keyphrase must appear
                in to be included in the anchor pool. Removes hapax phrases that
                are too specific to one document. Default 2.
            max_anchor_df_ratio: Maximum fraction of documents a keyphrase may
                appear in. Phrases above this threshold are class-agnostic noise
                (e.g. "feel like" in 73% of MH posts). Default 0.4.
            mmr_diversity: MMR lambda for word-list construction. Controls the
                relevance/diversity trade-off: 0.0 = pure relevance (top-k by
                cosine sim), 1.0 = pure diversity. Default 0.3 (70% relevance)
                - raised from an original 0.2 after sweeping 0.1/0.2/0.3/0.5/0.8
                on real per-class mental-health corpora: 0.5+ visibly pulled
                off-topic words into topics (e.g. "manic"/"abortion" showing up
                in a depression-only corpus's word lists, picked purely for
                being dissimilar to already-selected words, not for relevance)
                while 0.3 stayed as coherent as 0.2 with a small diversity gain.
            mmr_max_sim: Hard dedup cap for word-list MMR. Any candidate with
                cosine similarity above this threshold to any already-selected
                word is excluded before scoring. Prevents near-morphological
                duplicates (e.g. anxious/anxiousness, sim=0.93) regardless of
                lambda. Default 0.75 - lowered from an original 0.85 after
                finding morphological near-duplicates (depressed/depressive/
                depressing/depresses/depressant all in one topic's word list)
                sitting at 0.61-0.82 pairwise cosine similarity, i.e. under the
                old cap; 0.75 removes most of that cluster while leaving
                genuinely distinct content words (which sat at 0.33-0.64 in
                the same test) untouched.
            lemma_dedup: If True, additionally excludes any candidate word
                from a topic's word list whose lemma (isolated-token spaCy
                lemmatization, language-appropriate) matches an
                already-selected word's lemma *within that same topic* -
                e.g. stops one topic from listing both "gebückt" and
                "bücken". Independent of mmr_max_sim: in a compressed
                embedding space, morphological variants of one word don't
                necessarily score above mmr_max_sim against each other (nor
                against unrelated words), so that check alone can miss them.
                Within-topic only — the same word appearing across several
                different topics is a different phenomenon this does not
                address. Default False, fully backward-compatible.
            centroid_mode: How the word-list projection scores each topic
                (see WordTopicProjector.project_vocabulary's docstring,
                topic_assignment.py, for the full mechanism and the bipolar
                diagnostic that motivated this). "mean" (default, unchanged
                behavior) - cosine similarity to the plain GMM centroid,
                which was found to systematically underrepresent any
                scattered anchor subgroup within a cluster (e.g. varied
                medication names) relative to a tighter one (e.g. repetitive
                condition-identity phrasing), regardless of its share of the
                cluster. "max", "top_k_anchors", or "blend" instead score
                relevance using the cluster's individual anchor embeddings
                (self.anchor_embeddings_/self.anchor_labels_, already
                computed at Step 4/4d) - opt-in only, since it changes
                output on every corpus, not just ones with this failure mode.
            centroid_blend_alpha: Only used when centroid_mode="blend" - see
                WordTopicProjector.project_vocabulary. Default 0.5.
            centroid_top_k: Only used when centroid_mode="top_k_anchors" -
                see WordTopicProjector.project_vocabulary. Default 3.
            exclusive_assignment: If True, a word can only be listed in the
                one topic it is most similar to (argmax across all topics'
                mean centroids), instead of every topic independently
                filling its own list from any word above soft_threshold -
                see WordTopicProjector.project_vocabulary's docstring for
                the bipolar diagnostic that motivated this (adjacent topics
                sharing 4-7 of 15 displayed words with the default). Default
                False, fully backward-compatible.
            tfidf_ngram_range: n-gram range for TF-IDF keyphrase mode.
                Default (1, 2) extracts unigrams and bigrams.
            keybert_ngram_range: n-gram range for KeyBERT candidate generation.
                Default (1, 2). Use (1, 3) if you need longer anchor phrases but
                expect ~2× longer extraction time.
            keybert_candidate_pos_filter: If True, restricts KeyBERT's own
                candidate n-grams (i.e. the phrases that go on to become
                anchors, not just the final display vocabulary -
                content_pos_filter above only affects the latter) to
                right-headed noun phrases (last token NOUN/PROPN), via a
                custom fixed-vocabulary CountVectorizer built from spaCy POS
                tags. Diagnosed as the correctly-scoped fix for topics that
                form as tight but non-thematic clusters of words sharing a
                grammatical/register class (e.g. "said/asked/murmured", all
                speech verbs) rather than a shared subject - see
                keyphrase_extractor.py's _noun_final_candidate_vectorizer.
            exclude_abstract_suffixes: False (default). Only has any effect
                when keybert_candidate_pos_filter=True and language="german" -
                additionally drops candidates headed by a word ending in a
                productive German abstract-nominalization suffix (-ung/
                -heit/-keit/-nis/-schaft/-tum: Verwunderung, Dankbarkeit,
                Einsamkeit...). Experimental - see utils.
                build_noun_final_vocab's docstring for the full rationale
                and known false positives (Zeitung, Mannschaft).
                Default False preserves prior behavior byte-for-byte; ~2-4x
                slower extraction (one spaCy pass over the corpus).
            content_words_method: ``"tfidf"`` (default), ``"spacy"`` or
                ``"sampled_pos"``. Controls
                how the vocabulary for topic word projection is built. TF-IDF is
                ~300× faster; spaCy is more linguistically precise (POS-filtered,
                lemmatised) but very slow on corpora > ~5,000 docs. "sampled_pos"
                gives spaCy-like POS filtering by tagging only a few
                sampled occurrences of each word (+ a capitalisation name filter for
                English) - see fast_content_words.py.
            content_pos_filter: ``None`` (default) or ``"paper_restricted"``.
                Only affects content_words_method="spacy". None keeps the
                longstanding broad POS set {NOUN, VERB, ADJ, ADV, PROPN}.
                "paper_restricted" keeps only NN/ADJA/ADJD/VVFIN (German,
                exact STTS match) or its hand-mapped English equivalent —
                nouns, adjectives, and *finite lexical* verbs only, excluding
                infinitives/gerunds/participles/adverbs/proper nouns/modals/
                auxiliaries — per Uglanova & Gius 2020 ("The Order of
                Things", CHR 2020), whose LDA runs on German literary prose
                found this restriction gave their cleanest topic models. Add
                this if your topics are forming as tight but non-thematic
                clusters of words sharing a grammatical/register class
                (e.g. all speech-act verbs, all reproach words) rather than
                a shared subject — see utils.py's
                _passes_paper_restricted_filter for the exact tag mapping.
                Default None preserves prior behavior byte-for-byte.
            normalize_embeddings: If True, L2-normalise all embeddings before
                GMM fitting and downstream assignment. This projects keyphrases,
                words, and documents onto the unit sphere so the GMM covariance
                matrices capture directional (angular) variance rather than
                magnitude variance. Makes ``GMM.predict_proba`` valid for
                out-of-distribution inputs (documents vs keyphrases). Default
                False to preserve backward compatibility.
            keybert_use_mmr: If True and kp_algorithm="keybert", diversifies
                KeyBERT's own top-n_keyphrases candidates per document with MMR
                instead of plain top-n-by-relevance. Distinct from mmr_diversity/
                mmr_max_sim above, which operate on the final per-topic word list
                built later from the anchor pool — this instead fixes candidate
                diversity at the very first extraction step, before anchors are
                even built. Default False to preserve backward compatibility.
            keybert_diversity: MMR lambda passed to KeyBERT when
                keybert_use_mmr=True. Default 0.5 (KeyBERT's own default).
            language: ``"english"`` (default) or ``"german"``. Selects the
                stopword list used by KeyBERT extraction and by
                content_words_method="tfidf"/"spacy" word-list construction,
                and (for "spacy") which spaCy pipeline is loaded
                (en_core_web_sm vs de_core_news_sm). embedding_model is
                independent of this — pick a model that actually supports
                the language (e.g. a German or multilingual
                sentence-transformer) separately.
            keyphrase_scope: "flake": the anchor pool is
                FLaKE's corpus-level selection (flake
                .extract_flake_keyphrases, default settings) of
                n_keyphrases_total phrases, used as-is - FLaKE already applies
                the min_anchor_df / max_anchor_df_ratio gate to its candidates
                and its facility-location selection already removes
                redundancy, so KATM's own doc-frequency filtering and greedy
                semantic dedup (anchor_dedup_threshold) are skipped.
                ``flake_kwargs`` are passed through to extract_flake_keyphrases.
            keyphrase_scope (original options): "per_document" (default, unchanged behavior -
                KeyBERT/RAKE/YAKE runs once per document, candidates pooled
                and doc-frequency-filtered) or "global" (one pass over the
                whole corpus instead of one extraction call per document -
                see keyphrase_anchored_tm.global_keyphrases for the two
                correctness fixes this needed before it could match
                per-document quality: cross-document frequency filtering
                applied to candidates before embedding, and ranking by
                top-k average similarity to best-matching documents rather
                than a single corpus-wide reference vector. Full
                before/after evidence across three benchmark datasets in
                sdtm/KATM_GLOBAL_REPORT.md). Only used when
                kp_algorithm="keybert" - other algorithms ignore it and
                always run per-document. Typically 2.5-3.5x faster than
                per_document mode and matched or beat its coherence on 2 of
                3 datasets tested; there is no universally-better setting,
                so compare both on your own corpus before switching.
            n_keyphrases_total: Total keyphrase budget extracted once from
                the whole corpus - only used when keyphrase_scope="global".
                Default 500. No single value was best across all three
                datasets tested (500 for two, 1500 for the third) -
                candidates already surviving the frequency filter are
                nearly free to rank, so sweeping this is cheap.
            global_top_k_docs: Only used when keyphrase_scope="global". Size
                of the per-candidate averaging window in the local-relevance
                ranking (see global_keyphrases.py's extract_global_keyphrases
                docstring for the full mechanism). None (default) keeps the
                original auto formula, max(5, min(50, n_docs // 20)) -
                unchanged behavior. Diagnosed directly on the bipolar
                per-class mental-health corpus: a candidate with fewer than
                this many genuinely similar documents (e.g. "depakote" has
                ZERO documents above 0.5 cosine similarity there) gets its
                score diluted by forced inclusion of weak matches just to
                fill the window, while a near-universal condition-identity
                bigram never hits that floor and keeps a high, undiluted
                score - systematically outcompeting rarer-but-relevant
                candidates (specific medication names) for the fixed
                n_keyphrases_total slots. Lowering this shrinks how much
                "real" support a candidate needs, at the cost of noisier
                estimates from fewer documents - exposed as a parameter so
                that trade-off can be swept per-corpus instead of living
                inside one hardcoded formula.
            global_ranking_mode: Only used when keyphrase_scope="global".
                "top_k_average" (default, unchanged behavior) or
                "membership" - see global_keyphrases.py's module docstring
                (fourth-fix section) for the full mechanism. "membership"
                ports per-document mode's own doc_freq-style selection onto
                the shared global candidate vocabulary (does a candidate win
                a "vote" from enough individual documents' own top-N,
                instead of averaging similarity across a fixed window of
                documents) - nearly triples medication-candidate survival on
                the bipolar per-class corpus (1.9% -> 5.6%) and roughly
                doubles the best achievable GMM-cluster purity (15% -> 35%).
                Opt-in only - validated on the per-class mental-health
                corpora, not yet checked as broadly as the other global-mode
                fixes.
            global_membership_n_per_doc: Only used when
                global_ranking_mode="membership". Per-document top-N window
                size for the membership vote, mirroring per-document mode's
                own n_keyphrases. Default 10.
            dim_reduction: None (default, unchanged behavior) or "pca" -
                applied to anchor embeddings before GMM clustering; every
                downstream embedding (content-word vocabulary, documents)
                is passed through the same fitted reducer via a thin
                wrapper (embedding_reduction.ReducedEmbedder), so
                WordTopicProjector needs no changes to work in the reduced
                space. Consistently *hurt* coherence (typically -0.05 to
                -0.12) on every one of 42 configs tested across three
                datasets - kept as an option for further experimentation,
                not because it was found to help. "umap" was tried and
                removed after producing degenerate topics (coherence
                0.0000) on every dataset - see
                embedding_reduction.py's module docstring.
            dim_reduction_n_components: Target PCA dimensionality. Defaults
                to min(50, n_anchors-1, embedding_dim) when left None.
            n_keyphrases_per_chunk: Only used with long_document_strategy=
                "chunk_extract": keyphrases extracted from each ~256-token
                chunk separately (default 5); a document's phrases are the
                union over its chunks.
            long_document_strategy: "truncate" (default, unchanged behavior),
                "chunk", or "chunk_extract" (per-chunk extraction, see
                n_keyphrases_per_chunk) - only affects kp_algorithm="keybert",
                keyphrase_scope="per_document". KeyBERT ranks candidates by
                similarity to one embedding of the whole document, which
                SentenceTransformer.encode() silently truncates at the
                model's max_seq_length (256 tokens for all-MiniLM-L6-v2) -
                confirmed to affect 100% of a real 300-document sample in
                this project's own mental-health-Reddit notebooks (median
                document length 1,686 tokens, 6.6x the limit) once several
                posts are merged into one document. "chunk" splits any
                document exceeding max_seq_length into sentence-packed
                chunks that fit within it and scores each candidate by its
                maximum similarity across the document's own chunks
                instead - see keyphrase_extractor.KeyphraseExtractor's
                docstring for the full mechanism and why "max" rather than
                the keyphrase_scope="global" fix's top-k-average is the
                right aggregation here. A no-op (identical output to
                "truncate") for any document that already fits within
                max_seq_length, so switching this on is a strict fix, never
                a regression, when your documents happen to be short.
            gsc_candidate_ranking, gsc_candidate_pool, gsc_chunk_size,
                gsc_ngram_range, gsc_dedup, gsc_method, gsc_batch_size: Only affect
                kp_algorithm="gsc" - passed straight through to
                KeyphraseExtractor, whose own docstring documents each in
                full (candidate pre-filtering, chunking, dedup, and the
                "centroid" vs "orthogonal" selection objective).
            contextual_anchor_embedding: False (default, unchanged behavior -
                every anchor keyphrase is embedded as a bare, isolated
                string, exactly as before). If True, each anchor is instead
                embedded by finding its actual occurrences across the
                fitted documents (the sentence each occurrence appears in),
                embedding every occurrence, and averaging - giving the
                anchor a CONTEXTUALIZED embedding instead of one derived
                from the phrase's own default/most-common sense in
                isolation. Added to address a concrete, observed failure
                mode: embedding a bare polysemous word/phrase (e.g. "kitty"
                as a character's nickname, "monkey" as half of the place
                name "Monkey Island") pulls it toward that word's generic
                dictionary sense (a young cat; a primate) regardless of how
                it's actually used in the fitted documents, which can merge
                two semantically unrelated anchors into the same GMM
                topic. Falls back to the isolated-phrase embedding for any
                anchor whose exact surface form isn't found in any
                document (should be rare). This is O(n_anchors x n_docs)
                substring search plus one extra encode() pass over every
                collected occurrence, so it costs more than the default -
                opt in only where anchor quality matters more than fit
                time, and validate on a small sample before a full corpus
                run (see project memory's standing validate-small-first
                practice).
            contextual_max_occurrences: Cap on how many occurrences (across
                all documents) are embedded and averaged per anchor when
                contextual_anchor_embedding=True. Default 20 - keeps the
                cost bounded for very common anchors without needing every
                single occurrence; ignored otherwise.
            discard_hub_clusters: False (default, unchanged behavior). If
                True, the production GMM (Step 5) fits n_topics +
                hub_overcluster_k components instead of n_topics, then keeps
                only the n_topics components furthest (lowest cosine
                similarity) from the global mean of the anchor pool,
                discarding the rest - see GMMTopicClusterer's own docstring
                for the full mechanism and the two-novel empirical finding
                that motivated it (a component whose centroid sits
                unusually close to "the average of everything" is reliably
                a generic-word catch-all, not a real topic). Anchors
                originally assigned to a discarded component are reassigned
                to their nearest surviving centroid, not dropped. Opt-in and
                exploratory - the discovery signal is validated, net effect
                on topic quality is not yet.
            hub_overcluster_k: Only used when discard_hub_clusters=True. How
                many extra components to overcluster with before discarding
                back down to n_topics. Default 5 - an initial guess, not
                independently swept; exposed specifically so it can be
                tuned per-corpus (see GMMTopicClusterer's docstring for the
                size/certainty trade-off).
            streaming: False (default) = everything that touches the corpus runs in
                memory on a list of documents. True = the corpus is processed as a
                STREAM, so it never has to fit in memory: ``fit()`` then accepts a
                list OR a zero-argument callable returning a fresh iterator of texts /
                (id, text) pairs (it is iterated several times: keyphrases, content
                words, and, if compute_doc_topic_probs, document vectors). What
                streams: anchor extraction (keyphrase_scope="flake" ->
                extract_flake_keyphrases_streaming; per-document algorithms such as
                KeyBERT -> batches of ``stream_batch_size`` documents through the same
                KeyphraseExtractor, keeping only a phrase->document-frequency
                Counter), content words (all four content_words_method values via
                streaming.extract_content_words_stream) and Step 8. Not supported
                with streaming=True: keyphrase_scope="global",
                contextual_anchor_embedding, FLaKE partitions. If False and a
                callable is passed, it is materialised into a list first.
            stream_batch_size: documents per KeyphraseExtractor call when streaming.
            stream_progress_every: print a progress line every N documents during the
                streaming KeyBERT/POS-vocabulary passes (0 = silent).
            stream_candidate_vocab: only for kp_algorithm="keybert" with
                keybert_candidate_pos_filter=True. KeyBERT's noun-final candidate vocabulary
                is corpus-wide (an n-gram is a candidate for every document if it is a noun
                phrase anywhere), so "global" (default) first streams one spaCy pass to build it,
                then extracts batch by batch against it - identical anchors to streaming=False.
                "batch" skips that pass and builds the vocabulary from each batch only: one pass
                cheaper, approximate (a document only sees candidates found in its own batch).
            compute_doc_topic_probs: False (default - CHANGES PRIOR BEHAVIOR,
                unlike every other flag in this class). When False, fit()
                skips Step 8 (encoding every document again and taking
                cosine similarity to each topic centroid) entirely -
                doc_topic_probs_ is left as an empty list, and fit_transform
                returns an all-zero array instead. topics_/get_topic_words
                are completely unaffected either way - Step 8 runs strictly
                after project_vocabulary (Step 7) has already fully
                determined every topic's word list, so nothing about the
                topics themselves depends on this flag. Measured directly on
                a real 250-document German novel at 34.5% of total fit()
                time (67.3s of 195.1s) - the second-largest cost in the
                whole pipeline after keyphrase extraction itself, for output
                (doc_topic_probs_, and models.py's run_katm's "top_docs"
                exemplar lookup built from it) that plain topic-word
                evaluation never reads. Set True to restore the original
                behavior - needed for anything that scores or uses
                per-document topic assignments (e.g. OCTIS-style evaluation
                harnesses, or pulling exemplar documents per topic).
                transform() (scoring NEW documents after fit()) is
                unaffected by this flag either way - it always recomputes
                fresh from the fitted centroids, independent of
                doc_topic_probs_.
        """
        self.kp_algorithm = kp_algorithm
        self.n_keyphrases = n_keyphrases
        self.embedding_model = embedding_model
        self.n_topics = n_topics
        self.soft_threshold = soft_threshold
        self.top_n_words = top_n_words
        self.min_df = min_df
        self.yake_use_position = yake_use_position
        self.tfidf_ngram_range = tfidf_ngram_range
        self.keybert_ngram_range = keybert_ngram_range
        self.keybert_candidate_pos_filter = keybert_candidate_pos_filter
        self.exclude_abstract_suffixes = exclude_abstract_suffixes
        self.content_words_method = content_words_method
        self.content_pos_filter = content_pos_filter
        self.normalize_embeddings = normalize_embeddings
        self.anchor_dedup_threshold = anchor_dedup_threshold
        self.min_anchor_df = min_anchor_df
        self.max_anchor_df_ratio = max_anchor_df_ratio
        self.mmr_diversity = mmr_diversity
        self.mmr_max_sim = mmr_max_sim
        self.lemma_dedup = lemma_dedup
        self.centroid_mode = centroid_mode
        self.centroid_blend_alpha = centroid_blend_alpha
        self.centroid_top_k = centroid_top_k
        self.exclusive_assignment = exclusive_assignment
        self.keybert_use_mmr = keybert_use_mmr
        self.keybert_diversity = keybert_diversity
        self.language = language
        self.n_topics_range = n_topics_range
        self.n_topics_method = n_topics_method
        self.n_topics_covariance_type = n_topics_covariance_type
        self.gmm_covariance_type = gmm_covariance_type
        self.keyphrase_scope = keyphrase_scope
        self.flake_kwargs = dict(flake_kwargs or {})
        self.gmm_random_state = gmm_random_state
        #  "gmm" (default, original GMMTopicClusterer) or one of
        # clustering_alt.METHODS ("gmm_ninit", "gmm_hub", "spherical_kmeans",
        # "overcluster_merge", "agglomerative").
        from .clustering_alt import METHODS as _CL_METHODS
        if clustering_method != "gmm" and clustering_method not in _CL_METHODS:
            raise ValueError(f"clustering_method must be 'gmm' or one of {_CL_METHODS}")
        self.clustering_method = clustering_method
        self.clustering_kwargs = dict(clustering_kwargs or {})
        # exploratory: "default" (unchanged behavior) or
        # "coverage" - seed GMM/k-means with coverage_seed_embeddings
        # (clustering.py) instead of the clusterer's own kmeans++/EM init.
        # Only takes effect for clustering_method in ("gmm" with
        # discard_hub_clusters=False, "spherical_kmeans", "gmm_ninit") AND
        # keyphrase_scope="flake" with no partitions (the only case where
        # the docs x anchors relevance graph FLaKE already built is
        # available) - every other combination silently keeps default init,
        # since there is no coverage graph to seed from. See
        # coverage_seed_embeddings's own docstring for the mechanism.
        self.clustering_init = clustering_init
        # exploratory: drop the lowest-anchor_isolation_
        # scores fraction of the anchor pool before clustering (not before
        # anything else - content words for the final topic word lists come
        # from a separate, independent extract_content_words() vocabulary,
        # not from anchor_phrases_, so this only removes points that would
        # otherwise help define a GMM/k-means centroid - see clustering.
        # filter_isolated_anchors's docstring). 0.0 (default) = unchanged
        # behavior (no anchor is ever dropped).
        self.anchor_isolation_drop_frac = anchor_isolation_drop_frac
        self.anchor_isolation_k = anchor_isolation_k
        # "concentration" (default): top-k-neighbor similarity minus
        # similarity to the rest of the pool - validated on 20NG's real
        # catch-all topic to separate it from genuine topic words.
        # "density" (first thing tried, kept for comparison): plain top-k
        # neighbor similarity - measured and rejected, see clustering.
        # anchor_isolation_scores's docstring.
        self.anchor_isolation_method = anchor_isolation_method
        self.word_scoring = word_scoring  #  see topic_assignment.project_vocabulary
        self.exclude_words = list(exclude_words or [])
        self.n_keyphrases_total = n_keyphrases_total
        self.global_top_k_docs = global_top_k_docs
        self.global_ranking_mode = global_ranking_mode
        self.global_membership_n_per_doc = global_membership_n_per_doc
        self.dim_reduction = dim_reduction
        self.dim_reduction_n_components = dim_reduction_n_components
        self.long_document_strategy = long_document_strategy
        self.n_keyphrases_per_chunk = n_keyphrases_per_chunk
        self.gsc_candidate_ranking = gsc_candidate_ranking
        self.gsc_candidate_pool = gsc_candidate_pool
        self.gsc_chunk_size = gsc_chunk_size
        self.gsc_ngram_range = gsc_ngram_range
        self.gsc_dedup = gsc_dedup
        self.gsc_method = gsc_method
        self.gsc_batch_size = gsc_batch_size
        self.contextual_anchor_embedding = contextual_anchor_embedding
        self.contextual_max_occurrences = contextual_max_occurrences
        self.discard_hub_clusters = discard_hub_clusters
        self.hub_overcluster_k = hub_overcluster_k
        self.compute_doc_topic_probs = compute_doc_topic_probs
        self.streaming = streaming
        self.stream_batch_size = stream_batch_size
        self.stream_candidate_vocab = stream_candidate_vocab
        self.stream_progress_every = stream_progress_every
        self.n_topics_selection_scores_: Optional[Dict[str, list]] = None

        # State set by fit()
        self.topics_: Optional[Dict[int, List[Tuple[str, float]]]] = None
        self.doc_topic_probs_: Optional[List[np.ndarray]] = None
        self._n_fitted_docs = 0
        self._embedder: Optional[SentenceEmbedder] = None
        self._clusterer: Optional[GMMTopicClusterer] = None
        self._reducer = None

        # Clustering-geometry evaluation artifacts (internal, label-free
        # metrics on the anchor embeddings GMM actually clustered - see
        # subtopic_discovery/clustering_eval.py). Populated by fit();
        # anchor_embeddings_/anchor_labels_ are None if fit() found no
        # anchors to cluster (e.g. empty vocabulary after filtering).
        self.anchor_embeddings_: Optional[np.ndarray] = None
        self.anchor_labels_: Optional[np.ndarray] = None
        self.anchor_phrases_: Optional[List[str]] = None
        self.gmm_: Optional[GaussianMixture] = None
        #  docs x anchors FLaKE relevance graph (sparse),
        # column-aligned with anchor_phrases_/anchor_embeddings_ - set during
        # fit() only when keyphrase_scope="flake" and not partitioned; None
        # otherwise. Used by clustering_init="coverage"; also kept around
        # (cheap - already computed) for inspection even when not used.
        self._flake_anchor_relevance_ = None
        self.coverage_init_used_: bool = False
        #  diagnostics from anchor_isolation_drop_frac - the
        # anchors dropped before clustering (phrase strings) and the raw
        # per-anchor coherence score for every anchor that SURVIVED the FLaKE
        # pool (pre-drop), for inspection. Both empty/None unless
        # anchor_isolation_drop_frac > 0.
        self.dropped_isolated_anchors_: List[str] = []
        self.anchor_isolation_scores_: Optional[np.ndarray] = None

    def fit(self, documents: List[str]) -> "KATM":
        """Run full KATM pipeline.

        Args:
            documents: List of document strings. If your raw data is posts,
                group them into documents before calling fit().

        Returns:
            self for chaining.
        """
        factory = None
        if self.streaming:
            from .streaming import pairs_factory
            if self.keyphrase_scope == "global" or self.contextual_anchor_embedding:
                raise NotImplementedError('streaming=True does not support keyphrase_scope="global" '
                                          "or contextual_anchor_embedding")
            factory = pairs_factory(documents)
            documents = None          # never materialised on this path
        else:
            if callable(documents):
                documents = [t if not isinstance(t, tuple) else t[1] for t in documents()]
            if not documents:
                self.topics_ = {}
                self.doc_topic_probs_ = []
                return self

        self._flake_anchor_relevance_ = None  # reset each fit() call
        self.coverage_init_used_ = False
        self.dropped_isolated_anchors_ = []
        self.anchor_isolation_scores_ = None

        # Step 1: Load the shared embedder first so KeyBERT can reuse it
        embedder = SentenceEmbedder(model_name=self.embedding_model)
        import os as _os2
        if _os2.environ.get("KATM_EMBED_CACHE"):
            # experiment-only: memoise encodings across fits in one process, so
            # repeated runs that differ only downstream (e.g. clustering) don't
            # re-encode the same documents / phrases. Off unless the env var is set.
            embedder = _CachedSentenceEmbedder(embedder)

        # Step 2/3: Extract keyphrases and build the doc-frequency-filtered
        # anchor pool. "global" only applies to kp_algorithm="keybert" (see
        # __init__ docstring) - RAKE/YAKE/etc. always run per-document.
        flake_anchor_embeddings = None
        if self.streaming:
            all_keyphrases, flake_anchor_embeddings = self._stream_anchors(factory, embedder)
        elif self.keyphrase_scope == "flake":
            # FLaKE: one corpus-level, coverage-aware selection of the whole
            # anchor pool (see flake/core.py).
            import os as _os, sys as _sys
            _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from ._flake import flake_core
            extract_flake_keyphrases = flake_core().extract_flake_keyphrases
            fk = dict(ngram_range=self.keybert_ngram_range, language=self.language,
                      min_df=self.min_anchor_df, max_df=self.max_anchor_df_ratio)
            fk.update(self.flake_kwargs)
            n_parts = int(fk.pop("partitions", 1))
            part_seed = int(fk.pop("partition_seed", 0))
            if n_parts <= 1:
                all_keyphrases, _finfo = extract_flake_keyphrases(
                    documents, embedder, n_keyphrases=self.n_keyphrases_total, return_details=True, **fk)
                self.flake_info_ = {"timings": _finfo.get("timings", {}),
                                    "doc_coverage": _finfo.get("doc_coverage"),
                                    "n_candidates": len(_finfo.get("candidates", []))}
                if all_keyphrases:
                    # reuse FLaKE's phrase embeddings (same model, L2-normalised)
                    flake_anchor_embeddings = np.asarray(_finfo["candidate_embeddings"])[_finfo["indices"]]
                    # docs x anchors relevance submatrix, column-aligned with
                    # all_keyphrases/flake_anchor_embeddings (no dedup/reorder
                    # happens between here and clustering for keyphrase_scope
                    # ="flake" - see Step 4b below) - kept for
                    # clustering_init="coverage"; cheap, it's already built.
                    self._flake_anchor_relevance_ = _finfo["relevance"][:, _finfo["indices"]]
            else:
                # Partitioned FLaKE: random split of the corpus into n_parts
                # sub-corpora, FLaKE on each (budget K/n_parts each, phrase
                # embeddings cached across parts so shared vocabulary is
                # embedded once), anchors merged round-robin until K unique.
                import time as _time
                cache = {}

                class _CachedEmb:
                    _model = embedder._model

                    def encode(self, texts, batch_size=32):
                        miss = [t for t in dict.fromkeys(texts) if t not in cache]
                        if miss:
                            for t_, v_ in zip(miss, embedder.encode(miss, batch_size=batch_size)):
                                cache[t_] = v_
                        return np.stack([cache[t_] for t_ in texts]).astype(np.float32)

                _t0 = _time.perf_counter()
                order = np.random.RandomState(part_seed).permutation(len(documents))
                per = int(np.ceil(self.n_keyphrases_total / n_parts * 1.5))
                lists = []
                for part in np.array_split(order, n_parts):
                    ph = extract_flake_keyphrases([documents[i] for i in part], _CachedEmb(),
                                                  n_keyphrases=per, **fk)
                    lists.append(ph)
                merged, seen = [], set()
                for rank in range(per):
                    for ph in lists:
                        if rank < len(ph) and ph[rank] not in seen:
                            seen.add(ph[rank]); merged.append(ph[rank])
                    if len(merged) >= self.n_keyphrases_total:
                        break
                all_keyphrases = merged[: self.n_keyphrases_total]
                self.flake_info_ = {"timings": {"partitioned_total": _time.perf_counter() - _t0},
                                    "n_parts": n_parts, "n_embedded_phrases": len(cache)}
                if all_keyphrases:
                    flake_anchor_embeddings = np.stack([cache[k_] for k_ in all_keyphrases])
        elif self.keyphrase_scope == "global" and self.kp_algorithm == "keybert":
            from .global_keyphrases import extract_global_keyphrases
            all_keyphrases = extract_global_keyphrases(
                documents, embedder, self.keybert_ngram_range, self.language,
                self.min_anchor_df, self.max_anchor_df_ratio, self.n_keyphrases_total,
                top_k_docs=self.global_top_k_docs,
                ranking_mode=self.global_ranking_mode,
                membership_n_per_doc=self.global_membership_n_per_doc,
                # Reuses the same flag the "per_document" path uses for its
                # own candidate-generation POS restriction (see
                # global_keyphrases.extract_global_keyphrases's
                # candidate_pos_filter docstring) - one flag, same meaning,
                # regardless of keyphrase_scope.
                candidate_pos_filter=self.keybert_candidate_pos_filter,
                exclude_abstract_suffixes=self.exclude_abstract_suffixes,
            )
        else:
            # Extract keyphrases (pass embedder._model to avoid a second load)
            from .utils import _stopwords_for as _kp_stopwords_for
            kp_stop_words = "english" if self.language == "english" else list(_kp_stopwords_for(self.language))
            kp_extractor = KeyphraseExtractor(
                algorithm=self.kp_algorithm,
                n_keyphrases=self.n_keyphrases,
                pretrained_model=embedder._model,
                yake_use_position=self.yake_use_position,
                tfidf_ngram_range=self.tfidf_ngram_range,
                keybert_ngram_range=self.keybert_ngram_range,
                keybert_candidate_pos_filter=self.keybert_candidate_pos_filter,
                keybert_use_mmr=self.keybert_use_mmr,
                keybert_diversity=self.keybert_diversity,
                keybert_stop_words=kp_stop_words,
                language=self.language,
                long_document_strategy=self.long_document_strategy,
                n_keyphrases_per_chunk=self.n_keyphrases_per_chunk,
                gsc_candidate_ranking=self.gsc_candidate_ranking,
                gsc_candidate_pool=self.gsc_candidate_pool,
                gsc_chunk_size=self.gsc_chunk_size,
                gsc_ngram_range=self.gsc_ngram_range,
                gsc_dedup=self.gsc_dedup,
                gsc_method=self.gsc_method,
                gsc_batch_size=self.gsc_batch_size,
            )
            doc_keyphrases = kp_extractor.extract(documents)

            # Flatten keyphrases, track doc-frequency per phrase.
            # Sorting by doc-frequency before semantic dedup ensures the most
            # representative (common) phrases survive as GMM anchors — rare
            # one-off phrases are removed as near-duplicates of common ones.
            from collections import Counter as _Counter
            doc_freq: _Counter = _Counter()
            for kps in doc_keyphrases:
                for kp in set(kps):   # count each phrase once per document
                    doc_freq[kp] += 1

            # Apply document-frequency filter before sorting.
            # min_anchor_df removes hapax phrases (too specific to one doc).
            # max_anchor_df_ratio removes ubiquitous phrases that appear in so many
            # docs they carry no class signal (e.g. "feel like" in 73 % of MH posts).
            n_docs = len(documents)
            max_df = int(self.max_anchor_df_ratio * n_docs) if self.max_anchor_df_ratio else n_docs
            all_keyphrases = [
                kp for kp, freq in sorted(
                    doc_freq.items(), key=lambda x: (-x[1], x[0])
                )
                if self.min_anchor_df <= freq <= max_df
            ]

        if not all_keyphrases:
            self.topics_ = {}
            self.doc_topic_probs_ = []
            return self

        # Step 4: Embed keyphrases. Default: bare phrase string, in
        # isolation, as before. contextual_anchor_embedding=True instead
        # embeds each phrase's actual occurrences-in-context and averages
        # them - see _contextual_embed_keyphrases's docstring for why.
        if flake_anchor_embeddings is not None and not self.contextual_anchor_embedding:
            keyphrase_embeddings = self._maybe_normalize(flake_anchor_embeddings)
        elif self.contextual_anchor_embedding:
            keyphrase_embeddings = self._maybe_normalize(
                _contextual_embed_keyphrases(
                    all_keyphrases, documents, embedder,
                    max_occurrences=self.contextual_max_occurrences,
                )
            )
        else:
            keyphrase_embeddings = self._maybe_normalize(
                embedder.encode(all_keyphrases, batch_size=32)
            )

        # Step 4b: Greedy semantic deduplication of the anchor pool.
        # Processing in doc-frequency order means common phrases are kept as
        # seeds; near-synonyms that are rarer get dropped. This removes the
        # tight synonym clouds that RAKE/YAKE produce (e.g. "panic attack",
        # "panic attacks", "panicking") and forces GMM components to spread
        # across the actual topic space.
        # S3: for n ≤ _DEDUP_MATRIX_MAX_N, one cosine_similarity call covers
        # all pairwise sims in a single BLAS op; above that threshold the
        # original per-row loop runs to avoid an oversized n×n allocation.
        if self.anchor_dedup_threshold is not None and self.keyphrase_scope != "flake":
            keyphrase_embeddings, all_keyphrases = _greedy_dedup(
                keyphrase_embeddings, all_keyphrases, self.anchor_dedup_threshold
            )

        # Step 4b-bis: optional isolation filter - drop anchors with no
        # strong relationship to any other anchor in the pool before they
        # ever reach clustering (see clustering.filter_isolated_anchors's
        # docstring for the mechanism). 0.0 (default) is a no-op - this
        # cannot change anything unless explicitly requested.
        if self.anchor_isolation_drop_frac > 0 and len(all_keyphrases) > 1:
            from .clustering import anchor_isolation_scores, filter_isolated_anchors
            self.anchor_isolation_scores_ = anchor_isolation_scores(
                keyphrase_embeddings, self.anchor_isolation_k, method=self.anchor_isolation_method
            )
            keep_mask = filter_isolated_anchors(
                keyphrase_embeddings, self.anchor_isolation_k, self.anchor_isolation_drop_frac,
                method=self.anchor_isolation_method,
            )
            self.dropped_isolated_anchors_ = [p for p, k in zip(all_keyphrases, keep_mask) if not k]
            all_keyphrases = [p for p, k in zip(all_keyphrases, keep_mask) if k]
            keyphrase_embeddings = keyphrase_embeddings[keep_mask]
            if self._flake_anchor_relevance_ is not None:
                self._flake_anchor_relevance_ = self._flake_anchor_relevance_[:, keep_mask]

        # Step 4c: Automatic cluster-count selection, if requested. Runs on
        # the same (deduped) anchor embeddings that will be clustered below,
        # so the choice of k reflects the actual anchor pool GMM sees, not
        # the raw pre-dedup one.
        if self.n_topics == "auto":
            from .clustering import select_n_topics
            k_min, k_max = self.n_topics_range
            best_k, scores = select_n_topics(
                keyphrase_embeddings, k_min=k_min, k_max=k_max,
                method=self.n_topics_method, random_state=42,
                covariance_type=self.n_topics_covariance_type,
            )
            self.n_topics_selection_scores_ = scores
            self.n_topics = best_k

        # Step 4d: Optional dimensionality reduction (dim_reduction param).
        # Fit once on the anchor embeddings; every downstream embedding
        # (content words, documents) goes through the same fitted reducer
        # via ReducedEmbedder, never refit. None (default) is a no-op -
        # cluster_input stays identical to keyphrase_embeddings and
        # proj_embedder behaves exactly like the plain embedder.
        cluster_input = keyphrase_embeddings
        if self.dim_reduction is not None:
            from .embedding_reduction import fit_reducer
            self._reducer = fit_reducer(
                self.dim_reduction, keyphrase_embeddings,
                self.dim_reduction_n_components, random_state=42,
            )
            cluster_input = self._reducer.transform(keyphrase_embeddings)

        # Step 4e: clustering_init="coverage" - seed GMM/k-means from FLaKE's
        # own relevance graph instead of embedding-space kmeans++/EM init
        # (see clustering.coverage_seed_embeddings's docstring for the
        # mechanism and motivation). Only attempted for the combinations it
        # supports; everywhere else coverage_seeds stays None and the
        # clusterer below falls back to its own default init exactly as
        # before - this parameter cannot change behavior unless explicitly
        # requested AND a coverage graph is actually available.
        coverage_seeds = None
        # coverage_over_factor: read from clustering_kwargs as a side-channel
        # (popped before clustering_kwargs reaches AltTopicClusterer's own
        # constructor below) - see coverage_seed_embeddings's over_factor
        # docstring. 1 (default) = unchanged plain coverage seeding.
        coverage_over_factor = int(self.clustering_kwargs.get("coverage_over_factor", 1))
        if self.clustering_init == "coverage" and self._flake_anchor_relevance_ is not None:
            supports_coverage_init = (
                (self.clustering_method == "gmm" and not self.discard_hub_clusters)
                or self.clustering_method in ("spherical_kmeans", "gmm_ninit")
            )
            if supports_coverage_init:
                from .clustering import coverage_seed_embeddings
                coverage_seeds = coverage_seed_embeddings(
                    self._flake_anchor_relevance_, cluster_input, self.n_topics,
                    over_factor=coverage_over_factor,
                )
        self.coverage_init_used_ = coverage_seeds is not None

        # Step 5: Cluster (possibly-reduced) keyphrase embeddings
        import time as _time
        _tc = _time.perf_counter()
        if self.clustering_method == "gmm":
            clusterer = GMMTopicClusterer(
                n_topics=self.n_topics,
                soft_threshold=self.soft_threshold,
                covariance_type=self.gmm_covariance_type,
                discard_hub_clusters=self.discard_hub_clusters,
                hub_overcluster_k=self.hub_overcluster_k,
                random_state=self.gmm_random_state,
                means_init=coverage_seeds,
            )
        else:
            from .clustering_alt import AltTopicClusterer
            _alt_kwargs = {k: v for k, v in self.clustering_kwargs.items() if k != "coverage_over_factor"}
            clusterer = AltTopicClusterer(n_topics=self.n_topics, method=self.clustering_method,
                                          random_state=self.gmm_random_state, init_centers=coverage_seeds,
                                          **_alt_kwargs)
        clusterer.fit(cluster_input)
        self.cluster_time_ = _time.perf_counter() - _tc
        self._embedder  = embedder
        self._clusterer = clusterer

        # Clustering-geometry evaluation artifacts (see clustering_eval.py).
        # cluster_input is exactly what clusterer.fit() above saw, so
        # silhouette/DB/CH/Dunn/BIC/AIC computed on these describe the real
        # fit, not the raw pre-dedup (or pre-reduction) pool.
        self.anchor_embeddings_ = cluster_input
        # clusterer.predict() (not clusterer._gmm.predict()) so the discard_
        # hub_clusters remap/reassignment is applied when that option is on;
        # a plain pass-through to self._gmm.predict() otherwise, byte-
        # identical to before.
        self.anchor_labels_ = clusterer.predict(cluster_input)
        self.anchor_phrases_ = all_keyphrases  # same order as anchor_embeddings_/anchor_labels_
        self.gmm_ = clusterer._gmm

        # Step 6: Extract content words from all documents
        if self.streaming:
            from .streaming import extract_content_words_stream
            content_words = extract_content_words_stream(factory, min_df=self.min_df,
                                                         method=self.content_words_method,
                                                         language=self.language,
                                                         pos_filter=self.content_pos_filter)
        else:
            content_words = extract_content_words(documents, min_df=self.min_df,
                                                   method=self.content_words_method,
                                                   language=self.language,
                                                   pos_filter=self.content_pos_filter)
        if self.exclude_words:   # (e.g. character / place names in fiction)
            _ex = set(self.exclude_words)
            content_words = [w for w in content_words if w not in _ex]
        self.content_words_ = content_words

        # proj_embedder composes embedder.encode() with the fitted reducer's
        # .transform() (or is a plain passthrough when dim_reduction is
        # None) behind the same .encode() interface, so WordTopicProjector
        # and the document-embedding step below need no changes to work in
        # the reduced space.
        from .embedding_reduction import ReducedEmbedder
        proj_embedder = ReducedEmbedder(embedder, self._reducer)

        # Step 7: Project vocabulary onto clusters
        if content_words:
            projector = WordTopicProjector(clusterer=clusterer, embedder=proj_embedder)
            self.topics_ = projector.project_vocabulary(
                content_words,
                min_prob=self.soft_threshold,
                mmr_diversity=self.mmr_diversity,
                mmr_max_sim=self.mmr_max_sim,
                normalize_embeddings=self.normalize_embeddings,
                lemma_dedup=self.lemma_dedup,
                language=self.language,
                centroid_mode=self.centroid_mode,
                anchor_embeddings=self.anchor_embeddings_,
                anchor_labels=self.anchor_labels_,
                centroid_blend_alpha=self.centroid_blend_alpha,
                centroid_top_k=self.centroid_top_k,
                exclusive_assignment=self.exclusive_assignment,
                word_scoring=self.word_scoring,
            )
            # Limit to top_n_words per topic
            for tid in self.topics_:
                self.topics_[tid] = self.topics_[tid][: self.top_n_words]
        else:
            self.topics_ = {tid: [] for tid in range(self.n_topics)}

        # Step 8: Compute document-level topic distributions by encoding full
        # document texts and computing cosine similarity to each GMM centroid,
        # then normalising to a probability distribution via softmax.
        # Using cosine similarity to centroids (rather than GMM predict_proba
        # or averaged keyphrase posteriors) is robust across text lengths and
        # keyphrase extractor quality: a document finds its natural position
        # relative to topic anchor regions regardless of what phrases were
        # extracted from it. GMM posteriors are unreliable when training
        # (short phrases) and prediction (long documents) distributions differ.
        #
        # Gated behind compute_doc_topic_probs (default False - changes prior
        # behavior, see __init__ docstring): this step touches nothing topics_
        # depends on (project_vocabulary, Step 7 above, already fully
        # determined self.topics_ before this point) - it exists only to
        # populate doc_topic_probs_ / support transform()'s style of lookup
        # for individual fitted documents. Measured directly (see project
        # notes: gemini/subtopic_discovery's KATM time-breakdown
        # investigation) at 34.5% of total fit() time on a real 250-document
        # German novel (67.3s of 195.1s) - by far the second-largest cost
        # after keyphrase extraction itself, for a value nothing in that
        # investigation's own qualitative analysis ever read (it only ever
        # consumed topics_/get_topic_words). Still a real, used feature
        # elsewhere (e.g. OCTIS-style evaluation harnesses that score
        # document-topic assignments, or models.py's run_katm wrapper, which
        # uses it to find exemplar documents per topic) - just not one every
        # caller needs paid for by default now.
        if self.compute_doc_topic_probs:
            centroids = clusterer.cluster_centers_

            def _probs(batch):
                emb_ = self._maybe_normalize(proj_embedder.encode(batch, batch_size=32))
                # Shape: (n_docs, n_topics)
                sim_to_centroids = cosine_similarity(emb_, centroids)
                # Softmax to convert similarities to a probability distribution
                exp_sim = np.exp(sim_to_centroids - sim_to_centroids.max(axis=1, keepdims=True))
                return exp_sim / exp_sim.sum(axis=1, keepdims=True)

            if self.streaming:
                from .streaming import iter_batches
                self.doc_topic_probs_ = [r for b in iter_batches(factory, 256) for r in _probs(b)]
            else:
                self.doc_topic_probs_ = list(_probs(documents))
        else:
            self.doc_topic_probs_ = []

        self._n_fitted_docs = self._stream_n_docs_ if self.streaming else len(documents)

        return self

    def _stream_anchors(self, factory, embedder):
        """Step 2/3 for streaming=True: returns (anchor phrases, anchor embeddings or None).
        Sets self._stream_n_docs_ (documents streamed)."""
        self._stream_n_docs_ = 0
        if self.keyphrase_scope == "flake":
            from ._flake import flake_stream
            extract_flake_keyphrases_streaming = flake_stream().extract_flake_keyphrases_streaming
            fk = dict(ngram_range=self.keybert_ngram_range, language=self.language,
                      min_df=self.min_anchor_df, max_df=self.max_anchor_df_ratio, segment_words=None)
            fk.update(self.flake_kwargs)
            if int(fk.pop("partitions", 1)) > 1:
                raise NotImplementedError("streaming=True does not support flake_kwargs['partitions']")
            fk.pop("partition_seed", None)
            fk.pop("contiguous_ngrams", None)   # streaming FLaKE is always contiguous
            phrases, info = extract_flake_keyphrases_streaming(
                factory, embedder, n_keyphrases=self.n_keyphrases_total, return_details=True, **fk)
            self._stream_n_docs_ = int(info["n_passages"])
            self.flake_info_ = {"timings": info.get("timings", {}), "n_candidates": len(info.get("candidates", [])),
                                "streaming": True}
            if not phrases:
                return [], None
            emb = np.asarray(info["candidate_embeddings"])[info["indices"]]
            if self.clustering_init == "coverage":   # the passage graph is only needed for coverage seeding
                self._flake_anchor_relevance_ = info["relevance"].tocsc()[:, info["indices"]].tocsr()
            return phrases, emb

        from .streaming import extract_perdoc_keyphrases_streaming
        from .utils import _stopwords_for as _kp_stopwords_for
        kp_stop_words = "english" if self.language == "english" else list(_kp_stopwords_for(self.language))
        extractor = KeyphraseExtractor(
            algorithm=self.kp_algorithm, n_keyphrases=self.n_keyphrases, pretrained_model=embedder._model,
            yake_use_position=self.yake_use_position, tfidf_ngram_range=self.tfidf_ngram_range,
            keybert_ngram_range=self.keybert_ngram_range,
            keybert_candidate_pos_filter=self.keybert_candidate_pos_filter,
            keybert_use_mmr=self.keybert_use_mmr, keybert_diversity=self.keybert_diversity,
            keybert_stop_words=kp_stop_words, language=self.language,
            long_document_strategy=self.long_document_strategy,
            n_keyphrases_per_chunk=self.n_keyphrases_per_chunk, gsc_candidate_ranking=self.gsc_candidate_ranking,
            gsc_candidate_pool=self.gsc_candidate_pool, gsc_chunk_size=self.gsc_chunk_size,
            gsc_ngram_range=self.gsc_ngram_range, gsc_dedup=self.gsc_dedup, gsc_method=self.gsc_method,
            gsc_batch_size=self.gsc_batch_size)
        if (self.kp_algorithm == "keybert" and self.keybert_candidate_pos_filter
                and self.stream_candidate_vocab == "global" and self.long_document_strategy not in ("chunk", "chunk_extract")):
            from .streaming import build_noun_final_vocab_streaming
            extractor._fixed_candidate_vocab = build_noun_final_vocab_streaming(
                factory, self.language, self.keybert_ngram_range, progress_every=self.stream_progress_every)
        phrases, info = extract_perdoc_keyphrases_streaming(
            factory, extractor, min_df=self.min_anchor_df, max_df_ratio=self.max_anchor_df_ratio,
            batch_size=self.stream_batch_size, progress_every=self.stream_progress_every, return_info=True)
        self._stream_n_docs_ = info["n_docs"]
        self.keyphrase_stream_info_ = {k: v for k, v in info.items() if k != "doc_freq"}
        return phrases, None

    def _maybe_normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize_embeddings:
            return x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x / norms

    def transform(self, documents: List[str]) -> np.ndarray:
        """Return topic distributions for new documents without re-fitting.

        Encodes each document with the fitted embedder, computes cosine
        similarity to the fitted GMM centroids, and applies softmax to
        produce a proper probability distribution over topics.

        Args:
            documents: List of document strings.

        Returns:
            numpy.ndarray of shape (n_docs, n_topics).

        Raises:
            RuntimeError: If called before fit().
        """
        if self._clusterer is None or not self._clusterer.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        if not documents:
            return np.zeros((0, self.n_topics))

        from .embedding_reduction import ReducedEmbedder
        proj_embedder = ReducedEmbedder(self._embedder, self._reducer)
        doc_embeddings = self._maybe_normalize(
            proj_embedder.encode(documents, batch_size=32)
        )
        centroids = self._clusterer.cluster_centers_
        sim = cosine_similarity(doc_embeddings, centroids)
        exp_sim = np.exp(sim - sim.max(axis=1, keepdims=True))
        return exp_sim / exp_sim.sum(axis=1, keepdims=True)

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the model and return topic distributions for the training documents.

        Equivalent to calling fit(documents) then retrieving doc_topic_probs_
        as a numpy array. Provided for sklearn-style API compatibility.

        Args:
            documents: List of document strings.

        Returns:
            numpy.ndarray of shape (n_docs, n_topics).
        """
        self.fit(documents)
        if not self.doc_topic_probs_:
            # Two different reasons doc_topic_probs_ can be empty: no
            # documents were fit at all (shape (0, n_topics) is correct), or
            # documents WERE fit but compute_doc_topic_probs=False skipped
            # Step 8 (the shape should still reflect the real n_docs, filled
            # with zeros, matching this method's own documented return
            # shape (n_docs, n_topics) - not silently collapse to 0 rows).
            return np.zeros((self._n_fitted_docs, self.n_topics))
        return np.array(self.doc_topic_probs_)

    def get_topic_words(self, topic_id: int, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get top words for a specific topic.

        Args:
            topic_id: Topic ID.
            n: Number of words to return (None for all).

        Returns:
            List of (word, probability) tuples.
        """
        if self.topics_ is None:
            return []

        if topic_id not in self.topics_:
            return []

        words = self.topics_[topic_id]
        if n is not None:
            return words[:n]
        return words

    def get_document_topics(self, doc_idx: int) -> Dict[int, float]:
        """Get topic distribution for a document by index.

        Args:
            doc_idx: Index of the document in the list passed to fit().

        Returns:
            Dictionary mapping topic_id to probability.
        """
        if self.doc_topic_probs_ is None or doc_idx < 0 or doc_idx >= len(self.doc_topic_probs_):
            return {}

        probs = self.doc_topic_probs_[doc_idx]

        # Convert to dict, filtering out zero probabilities
        return {i: float(probs[i]) for i in range(len(probs)) if probs[i] > 0}

    def print_topics(self, n_words: int = 10):
        """Pretty-print top words per topic.

        Args:
            n_words: Number of top words to print per topic.
        """
        if self.topics_ is None:
            print("Model has not been fitted.")
            return

        print(f"\n{'='*60}")
        print(f"KATM Topics (n_topics={self.n_topics})")
        print(f"{'='*60}")

        for topic_id in sorted(self.topics_.keys()):
            words = self.get_topic_words(topic_id, n=n_words)
            if words:
                word_str = ", ".join([f"{w} ({p:.3f})" for w, p in words])
                print(f"\nTopic {topic_id}:")
                print(f"  {word_str}")
            else:
                print(f"\nTopic {topic_id}: (no words above threshold)")

        print(f"\n{'='*60}\n")


# ── S3: vectorized anchor deduplication ──────────────────────────────────────

_DEDUP_MATRIX_MAX_N = 8000  # n×n float32 at this size ≈ 244 MB


def _greedy_dedup(
    embeddings: np.ndarray,
    phrases: List[str],
    threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    """Greedy semantic dedup of the anchor pool.

    Processes candidates in doc-frequency order (already sorted by caller);
    keeps a phrase iff its max cosine similarity to all previously kept phrases
    is below `threshold`. Logic is identical to the original per-row loop;
    only the implementation changes:

    - n ≤ _DEDUP_MATRIX_MAX_N: one cosine_similarity call for the full n×n
      pairwise matrix (single BLAS op), then a greedy pass over boolean-indexed
      rows — replaces n separate sklearn calls with one.
    - n > _DEDUP_MATRIX_MAX_N: falls back to the original per-row loop to avoid
      allocating an oversized matrix.
    """
    n = len(embeddings)
    if n == 0:
        return embeddings, phrases

    if n > _DEDUP_MATRIX_MAX_N:
        kept_indices = [0]
        kept_embs = [embeddings[0]]
        for i in range(1, n):
            if float(np.max(cosine_similarity([embeddings[i]], kept_embs)[0])) < threshold:
                kept_indices.append(i)
                kept_embs.append(embeddings[i])
        return np.array(kept_embs), [phrases[i] for i in kept_indices]

    sim_matrix = cosine_similarity(embeddings)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[0] = True
    for i in range(1, n):
        if float(sim_matrix[i, kept_mask].max()) < threshold:
            kept_mask[i] = True

    idx = np.where(kept_mask)[0]
    return embeddings[idx], [phrases[i] for i in idx]


def _contextual_embed_keyphrases(
    phrases: List[str],
    documents: List[str],
    embedder,
    max_occurrences: int = 20,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed each anchor phrase from its actual occurrences in `documents`
    (the containing sentence of each occurrence, averaged) instead of from
    the bare phrase string in isolation.

    Motivation, found by direct qualitative inspection (see project memory,
    the book-summary-comparison investigation): embedding a short polysemous
    phrase with no surrounding context pulls it toward that phrase's most
    common/default sense - e.g. bare "kitty" embeds near "cat", bare "monkey"
    embeds near "animal" - regardless of how the phrase is actually used in
    the fitted corpus (a character's nickname; half of the place name
    "Monkey Island"). That default-sense embedding is exactly what can merge
    two semantically unrelated anchors into the same GMM topic. Embedding
    real occurrences-in-context instead lets the surrounding sentence
    disambiguate the sense the same way a human reader would.

    Mechanism: for each phrase, find every sentence (across every document)
    containing it as a whole-word match, embed each occurrence sentence
    (capped at `max_occurrences`, first-found order - deterministic, not a
    random sample), and average those embeddings. A phrase whose exact
    surface form isn't found anywhere (should be rare - only possible if an
    upstream extraction step normalized it away from its literal source
    text) falls back to embedding the bare phrase string, so every anchor
    still gets *some* embedding.

    All occurrence sentences across all phrases are batched into ONE
    embedder.encode() call (plus a second, smaller call for any bare-phrase
    fallbacks), not one call per phrase - the expensive part is the O(n_anchors
    x n_docs) substring search, done in pure Python, not the embedding pass.

    Returns:
        (len(phrases), embedding_dim) array, same row order as `phrases` -
        a drop-in replacement for embedder.encode(phrases).
    """
    import re
    from nltk.tokenize import sent_tokenize

    doc_sentences = [sent_tokenize(doc) for doc in documents]

    flat_snippets: List[str] = []
    snippet_owner: List[int] = []  # index into `phrases`, parallel to flat_snippets
    n_found = [0] * len(phrases)

    for pi, phrase in enumerate(phrases):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for sents in doc_sentences:
            if n_found[pi] >= max_occurrences:
                break
            for s in sents:
                if n_found[pi] >= max_occurrences:
                    break
                if pattern.search(s):
                    flat_snippets.append(s)
                    snippet_owner.append(pi)
                    n_found[pi] += 1

    fallback_indices = [pi for pi in range(len(phrases)) if n_found[pi] == 0]

    snippet_embs = embedder.encode(flat_snippets, batch_size=batch_size) if flat_snippets else None
    fallback_embs = (
        embedder.encode([phrases[pi] for pi in fallback_indices], batch_size=batch_size)
        if fallback_indices else None
    )

    dim = (
        snippet_embs.shape[1] if snippet_embs is not None
        else fallback_embs.shape[1] if fallback_embs is not None
        else 0
    )
    out = np.zeros((len(phrases), dim))

    if snippet_embs is not None:
        sums = np.zeros((len(phrases), dim))
        for row_emb, pi in zip(snippet_embs, snippet_owner):
            sums[pi] += row_emb
        for pi in range(len(phrases)):
            if n_found[pi] > 0:
                out[pi] = sums[pi] / n_found[pi]

    if fallback_embs is not None:
        for row_emb, pi in zip(fallback_embs, fallback_indices):
            out[pi] = row_emb

    return out