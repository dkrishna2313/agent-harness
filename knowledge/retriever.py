"""EvidenceRetriever — lexical retrieval over the persistent Knowledge Base.

J8.3 introduces the retrieval abstraction that sits between the reasoning
pipeline and the KnowledgeStore.  Today the implementation is lexical
(keyword matching + metadata ranking).  The interface is stable: J8.4 will
add semantic / vector retrieval without changing the caller contract.

Architecture:
    Planner / EvidenceAgent
        ↓
    EvidenceRetriever.retrieve(query, ...)
        ↓
    KnowledgeStore (JSONL)   ← today
    / EmbeddingIndex (Qdrant) ← J8.4
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import Evidence, EvidenceType, KnowledgeMetadata, Source
from .store import KnowledgeStore

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

RETRIEVAL_METHOD_LEXICAL = "lexical-v1"

# ---------------------------------------------------------------------------
# Retrieval modes (J8.4)
# ---------------------------------------------------------------------------

RETRIEVAL_MODE_LEXICAL = "lexical"
RETRIEVAL_MODE_SEMANTIC = "semantic"
RETRIEVAL_MODE_HYBRID = "hybrid"

# Hybrid weighted combination: final_score = (w_lex * lex_rel + w_sem * sem) * metadata_factor
# Weights reflect semantic search's higher recall at the cost of precision;
# lexical anchors the signal to exact query-term coverage.
_HYBRID_LEXICAL_WEIGHT = 0.4
_HYBRID_SEMANTIC_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------

_INTENT_RISK = "RISK"

# Query terms that activate an intent
_INTENT_TRIGGERS: dict[str, frozenset[str]] = {
    _INTENT_RISK: frozenset([
        "risk", "risks", "barrier", "barriers", "challenge", "challenges",
        "uncertainty", "uncertain", "constraint", "constraints", "obstacle",
        "obstacles", "delay", "delays", "licens", "licensing", "regulatory",
        "regulation", "approval", "approvals", "cost", "costs", "economics",
        "haleu", "fuel", "financing", "schedule", "concern", "concerns",
        "problem", "problems", "difficulty", "issue", "issues",
    ]),
}

# Statement vocabulary that earns a boost when intent matches.
# Uses substring matching: "licens" catches license/licensing/licensed.
_INTENT_BOOST_VOCAB: dict[str, frozenset[str]] = {
    _INTENT_RISK: frozenset([
        # Risk / barrier
        "risk", "risks", "barrier", "barriers", "challenge", "challenges",
        "uncertainty", "uncertain", "constraint", "constraints",
        "obstacle", "obstacles", "concern", "concerns",
        # Regulatory
        "licens", "regulat", "approval", "nrc", "certif", "permit",
        # Fuel / supply chain
        "haleu", "fuel", "supply chain", "manufacturing", "production",
        "availability", "shortage",
        # Economics / financing
        "cost", "capital", "financ", "econom", "expensive", "investment",
        "foak", "first-of-a-kind", "affordab",
        # Construction / schedule
        "construction", "schedule", "delay", "timeline",
        # Other barriers
        "viab", "public acceptance", "limited", "lack", "insufficient", "unproven",
    ]),
}

# Additive boost magnitude and saturation count.
# vocab_boost = min(_INTENT_BOOST_MAX, n_matched / _INTENT_BOOST_SATURATION * _INTENT_BOOST_MAX)
_INTENT_BOOST_MAX = 0.45
_INTENT_BOOST_SATURATION = 5


def detect_intent(query_terms: list[str]) -> str | None:
    """Identify the primary retrieval intent from tokenised query terms.

    Returns an intent constant (e.g. ``_INTENT_RISK``) or ``None`` if the
    query has no detectable intent.  Only RISK intent is implemented in J8.3a;
    the hook is stable for future intents.
    """
    for intent, triggers in _INTENT_TRIGGERS.items():
        if any(t in triggers or any(t.startswith(tr) for tr in triggers) for t in query_terms):
            return intent
    return None


def _compute_vocab_boost(statement: str, intent: str | None) -> float:
    """Return the intent-vocabulary boost for one statement (0 ≤ result ≤ _INTENT_BOOST_MAX)."""
    if intent is None:
        return 0.0
    vocab = _INTENT_BOOST_VOCAB.get(intent)
    if not vocab:
        return 0.0
    s = statement.lower()
    n_matched = sum(1 for term in vocab if term in s)
    if n_matched == 0:
        return 0.0
    return min(_INTENT_BOOST_MAX, n_matched / _INTENT_BOOST_SATURATION * _INTENT_BOOST_MAX)


# ---------------------------------------------------------------------------
# Query tokenisation
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "for", "of", "in", "on",
    "at", "to", "from", "by", "with", "and", "or", "but", "not", "that",
    "this", "these", "those", "what", "which", "who", "how", "why", "when",
    "where", "their", "there", "they", "than", "then", "so", "if",
])

_TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]*")


def tokenize_query(query: str) -> list[str]:
    """Lowercase, tokenise, remove stopwords. Preserves domain terms like 'HALEU'."""
    tokens = _TOKEN_RE.findall(query.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedEvidence:
    """A single evidence item with its retrieval score and associated metadata."""

    evidence: Evidence
    metadata: KnowledgeMetadata
    score: float
    rank: int
    source: Source | None = field(default=None, repr=False)
    lexical_score: float = field(default=0.0, repr=False)
    semantic_score: float = field(default=0.0, repr=False)

    def load_source(self, store: KnowledgeStore) -> Source | None:
        """Lazy-load the primary Source for this evidence item."""
        if self.source is not None:
            return self.source
        if not self.evidence.supporting_source_ids:
            return None
        src = store.find_source(self.evidence.supporting_source_ids[0])
        self.source = src
        return src

    @property
    def statement(self) -> str:
        return self.evidence.statement

    @property
    def evidence_type(self) -> EvidenceType:
        return self.evidence.evidence_type


@dataclass
class RetrievalResult:
    """The complete output of a single EvidenceRetriever.retrieve() call."""

    query: str
    items: list[RetrievedEvidence]
    domains_searched: list[str]
    total_candidates: int
    matched_candidates: int
    retrieval_method: str
    latency_ms: float
    # J8.4 hybrid trace fields (optional, default to lexical-only values)
    mode: str = field(default=RETRIEVAL_MODE_LEXICAL)
    lexical_candidates: int = field(default=0)
    semantic_candidates: int = field(default=0)
    merged_candidates: int = field(default=0)
    duplicates_removed: int = field(default=0)
    semantic_model: str | None = field(default=None)

    def print_summary(self, *, show_source: bool = False) -> None:
        """Print a formatted retrieval summary to stdout."""
        mode_tag = f"mode: {self.mode}"
        if self.semantic_model:
            mode_tag += f" ({self.semantic_model})"

        header = (
            f"\nQuery:   {self.query!r}\n"
            f"Domains: {', '.join(self.domains_searched)}  |  "
            f"Method: {self.retrieval_method}  |  {mode_tag}  |  "
            f"Latency: {self.latency_ms:.0f}ms\n"
        )
        if self.mode == RETRIEVAL_MODE_HYBRID:
            header += (
                f"Lexical: {self.lexical_candidates}  |  "
                f"Semantic: {self.semantic_candidates}  |  "
                f"Merged: {self.merged_candidates}  |  "
                f"Dedup: {self.duplicates_removed}  |  "
                f"Returned: {len(self.items)}\n"
            )
        else:
            header += (
                f"Candidates: {self.total_candidates}  |  "
                f"Matched: {self.matched_candidates}  |  "
                f"Returned: {len(self.items)}\n"
            )
        print(header)

        if not self.items:
            print("  (no results)\n")
            return

        col_w = 88
        show_component = self.mode == RETRIEVAL_MODE_HYBRID
        if show_component:
            print(f"  {'#':>3}  {'Score':>6}  {'Lex':>5}  {'Sem':>5}  {'Type':<15}  Statement")
        else:
            print(f"  {'#':>3}  {'Score':>6}  {'Type':<15}  Statement")
        print("  " + "─" * (col_w + (18 if show_component else 0)))

        for item in self.items:
            stmt = item.statement
            if len(stmt) > col_w:
                stmt = stmt[:col_w - 1] + "…"
            if show_component:
                print(
                    f"  {item.rank:>3}  {item.score:>6.3f}  "
                    f"{item.lexical_score:>5.2f}  {item.semantic_score:>5.2f}  "
                    f"{item.evidence_type:<15}  {stmt}"
                )
            else:
                print(f"  {item.rank:>3}  {item.score:>6.3f}  {item.evidence_type:<15}  {stmt}")
            if show_source and item.source:
                src = item.source
                src_info = f"         └─ {src.title}"
                if src.organization:
                    src_info += f" · {src.organization}"
                if src.document_version:
                    src_info += f" · {src.document_version}"
                print(src_info)
        print()


# ---------------------------------------------------------------------------
# EvidenceRetriever
# ---------------------------------------------------------------------------


class EvidenceRetriever:
    """Retrieves Evidence from the KnowledgeStore against a natural-language query.

    Supports three retrieval modes (J8.4):
      - lexical  (default): keyword coverage + intent vocabulary boost
      - semantic:           cosine similarity over pre-computed embeddings
      - hybrid:             weighted sum of lexical relevance and semantic similarity,
                            then metadata re-ranking

    Hybrid scoring formula::

        final_score = (0.4 * lex_relevance + 0.6 * sem_similarity) * metadata_factor

    Where::

        lex_relevance  = (term_coverage + vocab_intent_boost)  ∈ [0, 1.45]
        sem_similarity = cosine_similarity(query, statement)   ∈ [0, 1]
        metadata_factor = quality × priority × strategic        ∈ [0.65, 1.45]

    Parameters
    ----------
    store:
        KnowledgeStore instance. Defaults to ``knowledge_store/`` in CWD.
    provider:
        EmbeddingProvider for semantic/hybrid modes.  Required when using
        those modes; ignored in lexical mode.
    """

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        provider: object | None = None,  # EmbeddingProvider — avoid circular import
    ) -> None:
        self.store = store or KnowledgeStore()
        self.provider = provider

    def retrieve(
        self,
        query: str,
        *,
        mode: str = RETRIEVAL_MODE_LEXICAL,
        domain: str | None = None,
        profile: str | None = None,
        top_k: int = 20,
        evidence_types: list[str] | None = None,
        retrieval_enabled_only: bool = True,
        min_score: float = 0.01,
        load_sources: bool = False,
    ) -> RetrievalResult:
        """Retrieve Evidence matching a query.

        Parameters
        ----------
        query:
            Natural-language retrieval query.
        mode:
            Retrieval mode: ``"lexical"`` (default), ``"semantic"``, or ``"hybrid"``.
        domain:
            Restrict to a specific domain (e.g. ``"smr"``). If None, all
            available domains are searched.
        profile:
            Restrict to evidence tagged with this profile_id.
        top_k:
            Maximum number of results to return.
        evidence_types:
            Filter to specific EvidenceType values, e.g.
            ``["STRATEGIC", "TECHNICAL"]``. None means all types.
        retrieval_enabled_only:
            If True (default), exclude evidence where
            KnowledgeMetadata.retrieval_enabled is False.
        min_score:
            Minimum relevance score threshold.  Items below this are excluded.
        load_sources:
            If True, load the primary Source record for each result.
        """
        t0 = time.monotonic()

        if mode not in (RETRIEVAL_MODE_LEXICAL, RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID):
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

        query_terms = tokenize_query(query)
        intent = detect_intent(query_terms) if query_terms else None
        LOGGER.debug("retriever: query=%r  mode=%s  intent=%s", query, mode, intent)

        # Semantic / hybrid require a query embedding
        query_vec: list[float] | None = None
        semantic_model: str | None = None
        if mode in (RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID):
            if self.provider is None:
                raise ValueError(
                    f"mode={mode!r} requires an EmbeddingProvider. "
                    "Pass provider= to EvidenceRetriever or use mode='lexical'."
                )
            query_vec = self.provider.embed_one(query)  # type: ignore[attr-defined]
            semantic_model = self.provider.model_name  # type: ignore[attr-defined]

        # Lexical-only: guard against all-stopword query
        if mode == RETRIEVAL_MODE_LEXICAL and not query_terms:
            LOGGER.warning("retriever: empty query after tokenisation — returning empty result")
            return RetrievalResult(
                query=query, items=[], domains_searched=[],
                total_candidates=0, matched_candidates=0,
                retrieval_method=RETRIEVAL_METHOD_LEXICAL,
                latency_ms=0.0,
                mode=mode,
            )

        domains = [domain] if domain else self.store.available_domains()

        # Prepare numpy query vector (normalised) for semantic paths
        q_norm = None
        if query_vec is not None:
            import numpy as np
            q_arr = np.array(query_vec, dtype=np.float32)
            q_norm = q_arr / (float(np.linalg.norm(q_arr)) + 1e-9)

        total_candidates = 0
        matched_candidates = 0
        # (final_score, lex_rel, sem_score, ev, meta)
        scored: list[tuple[float, float, float, Evidence, KnowledgeMetadata]] = []
        lex_count = 0
        sem_count = 0
        both_count = 0

        for dom in domains:
            meta_index = self._load_metadata_index(dom)

            for ev in self.store.iter_evidence(dom):
                total_candidates += 1
                meta = meta_index.get(ev.evidence_id)
                if meta is None:
                    continue

                if retrieval_enabled_only and not meta.retrieval_enabled:
                    continue
                if evidence_types and ev.evidence_type not in evidence_types:
                    continue
                if profile and profile not in ev.profile_ids:
                    continue

                if mode == RETRIEVAL_MODE_LEXICAL:
                    score = self._score(ev.statement, query_terms, meta, intent)
                    if score < min_score:
                        continue
                    matched_candidates += 1
                    scored.append((score, score, 0.0, ev, meta))

                elif mode == RETRIEVAL_MODE_SEMANTIC:
                    sem = self._cosine(ev.evidence_id, q_norm)
                    if sem <= 0.0:
                        continue
                    score = round(sem * self._metadata_factor(meta), 4)
                    if score < min_score:
                        continue
                    matched_candidates += 1
                    scored.append((score, 0.0, sem, ev, meta))

                else:  # hybrid
                    lex_rel = self._relevance_score(ev.statement, query_terms, intent)
                    sem = self._cosine(ev.evidence_id, q_norm)
                    sem = max(0.0, sem)

                    has_lex = lex_rel > 0.0
                    has_sem = sem > 0.0
                    if not has_lex and not has_sem:
                        continue

                    if has_lex:
                        lex_count += 1
                    if has_sem:
                        sem_count += 1
                    if has_lex and has_sem:
                        both_count += 1

                    combined = _HYBRID_LEXICAL_WEIGHT * lex_rel + _HYBRID_SEMANTIC_WEIGHT * sem
                    score = round(combined * self._metadata_factor(meta), 4)
                    if score < min_score:
                        continue
                    matched_candidates += 1
                    scored.append((score, lex_rel, sem, ev, meta))

        # Sort by combined score; tiebreak on overall_score
        scored.sort(key=lambda x: (x[0], x[4].overall_score), reverse=True)
        top = scored[:top_k]

        items: list[RetrievedEvidence] = []
        for rank, (score, lex_rel, sem, ev, meta) in enumerate(top, start=1):
            item = RetrievedEvidence(
                evidence=ev,
                metadata=meta,
                score=score,
                rank=rank,
                lexical_score=round(lex_rel, 4),
                semantic_score=round(sem, 4),
            )
            if load_sources:
                item.load_source(self.store)
            items.append(item)

        latency_ms = (time.monotonic() - t0) * 1000

        # Hybrid observability
        merged = lex_count + sem_count - both_count if mode == RETRIEVAL_MODE_HYBRID else 0

        LOGGER.info(
            "retriever: query=%r  mode=%s  domains=%s  candidates=%d  matched=%d  returned=%d  latency=%.0fms",
            query, mode, domains, total_candidates, matched_candidates, len(items), latency_ms,
        )

        return RetrievalResult(
            query=query,
            items=items,
            domains_searched=domains,
            total_candidates=total_candidates,
            matched_candidates=matched_candidates,
            retrieval_method=RETRIEVAL_METHOD_LEXICAL,
            latency_ms=latency_ms,
            mode=mode,
            lexical_candidates=lex_count if mode == RETRIEVAL_MODE_HYBRID else matched_candidates,
            semantic_candidates=sem_count if mode == RETRIEVAL_MODE_HYBRID else 0,
            merged_candidates=merged,
            duplicates_removed=both_count,
            semantic_model=semantic_model,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_metadata_index(self, domain: str) -> dict[str, KnowledgeMetadata]:
        return {m.evidence_id: m for m in self.store.iter_metadata(domain)}

    def _cosine(self, evidence_id: str, q_norm) -> float:  # type: ignore[no-untyped-def]
        """Load one evidence embedding and return cosine similarity with q_norm."""
        vec = self.store.read_embedding(evidence_id)
        if vec is None:
            return 0.0
        import numpy as np
        v = np.array(vec, dtype=np.float32)
        v_norm = v / (float(np.linalg.norm(v)) + 1e-9)
        return float(np.dot(q_norm, v_norm))

    @staticmethod
    def _relevance_score(
        statement: str,
        query_terms: list[str],
        intent: str | None = None,
    ) -> float:
        """Pure query-relevance signal: term_coverage + vocab_boost.

        No metadata factors — those are applied separately so hybrid mode can
        combine lexical and semantic relevance before multiplying by metadata.

        Returns 0.0 when no query term is found in the statement.
        """
        s = statement.lower()
        matched = sum(1 for term in query_terms if term in s)
        if matched == 0:
            return 0.0
        coverage = matched / len(query_terms)
        vocab_boost = _compute_vocab_boost(statement, intent)
        return coverage + vocab_boost

    @staticmethod
    def _metadata_factor(meta: KnowledgeMetadata) -> float:
        """Combined metadata quality multiplier ∈ [0.65, 1.45].

        Ranges are narrow so metadata breaks ties without overriding relevance.

        overall_score (1–5)       → quality_factor   ∈ [0.8, 1.2]
        retrieval_priority (1–5)  → priority_factor  ∈ [0.9, 1.1]
        strategic_value (0–1)     → strategic_factor ∈ [0.9, 1.1]
        """
        quality_factor   = 0.8 + (meta.overall_score - 1.0) / 4.0 * 0.4   # [0.8, 1.2]
        priority_factor  = 0.9 + (meta.retrieval_priority - 1) / 4.0 * 0.2  # [0.9, 1.1]
        strategic_factor = 0.9 + meta.strategic_value * 0.2                  # [0.9, 1.1]
        return quality_factor * priority_factor * strategic_factor

    @staticmethod
    def _score(
        statement: str,
        query_terms: list[str],
        meta: KnowledgeMetadata,
        intent: str | None = None,
    ) -> float:
        """Full lexical score: _relevance_score × _metadata_factor.

        Used by lexical-only mode.  Hybrid mode calls _relevance_score and
        _metadata_factor separately to interleave semantic scoring between them.
        """
        rel = EvidenceRetriever._relevance_score(statement, query_terms, intent)
        if rel == 0.0:
            return 0.0
        return round(rel * EvidenceRetriever._metadata_factor(meta), 4)
