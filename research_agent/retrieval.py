"""Question-aware retrieval scoring for chunk selection."""

from __future__ import annotations

import re

from .chunker import _extract_question_terms, score_chunk_relevance
from .schemas import Chunk, RetrievalScore, SourceQuality
from .source_quality import classify_source_quality

# Scoring weights (must sum to 1.0)
KEYWORD_WEIGHT = 0.40
TOPIC_WEIGHT = 0.35
SOURCE_QUALITY_WEIGHT = 0.25  # formerly DOC_PRIORITY_WEIGHT
# Backward-compat alias
DOC_PRIORITY_WEIGHT = SOURCE_QUALITY_WEIGHT

DEFAULT_TOP_CHUNKS = 15

TOPIC_KEYWORDS: dict[str, set[str]] = {
    "power": {"power", "watt", "kw", "mw", "energy", "pdu", "ups", "bess", "voltage", "current"},
    "cooling": {"cool", "thermal", "liquid", "dlc", "cdu", "heat", "temperature", "water"},
    "networking": {"network", "ethernet", "infiniband", "nvlink", "bandwidth", "switch", "fabric"},
    "rack architecture": {"rack", "chassis", "mgx", "nvl", "tray", "density", "form factor"},
    "operations": {"operat", "manag", "monitor", "mission control", "deploy", "mainten"},
    "resiliency": {"resilient", "redundan", "failover", "uptime", "bess", "buffer"},
}


def classify_document_priority(document_name: str) -> float:
    """Return normalized document priority (0–1) via source quality classifier.

    Kept for backward compatibility; new code should use
    ``classify_source_quality`` directly.
    """
    return classify_source_quality(document_name).source_quality_score / 5.0


def _detect_question_topics(question: str) -> set[str]:
    """Return topic names whose keywords appear in the question."""
    normalized = question.lower()
    detected: set[str] = set()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            detected.add(topic)
    return detected


def _topic_match_score(chunk: Chunk, detected_topics: set[str]) -> float:
    """Return fraction of detected topics that have at least one keyword in chunk text."""
    if not detected_topics:
        return 0.0
    text_lower = chunk.text.lower()
    matched = sum(
        1
        for topic in detected_topics
        if any(kw in text_lower for kw in TOPIC_KEYWORDS.get(topic, set()))
    )
    return round(matched / len(detected_topics), 4)


def score_retrieval(
    chunk: Chunk,
    question: str,
    question_terms: set[str],
    detected_topics: set[str],
    source_quality_map: dict[str, SourceQuality] | None = None,
) -> RetrievalScore:
    """Score a chunk against the question using keyword, topic, and source-quality signals."""
    keyword_score = score_chunk_relevance(chunk, question_terms)
    topic_score = _topic_match_score(chunk, detected_topics)

    # Source quality: use pre-built map when available, else classify on-the-fly
    if source_quality_map and chunk.document_name in source_quality_map:
        sq = source_quality_map[chunk.document_name]
    else:
        sq = classify_source_quality(chunk.document_name)
    quality_score_raw = sq.source_quality_score
    quality_score_norm = quality_score_raw / 5.0  # normalize to [0.2, 1.0]

    overall = round(
        KEYWORD_WEIGHT * keyword_score
        + TOPIC_WEIGHT * topic_score
        + SOURCE_QUALITY_WEIGHT * quality_score_norm,
        4,
    )

    return RetrievalScore(
        chunk_id=chunk.chunk_id,
        document_name=chunk.document_name,
        keyword_score=keyword_score,
        topic_match_score=topic_score,
        document_priority_score=quality_score_norm,  # normalized quality as priority
        source_quality_score=quality_score_raw,
        overall_retrieval_score=overall,
    )


# Minimum retrieval score a chunk must reach to be eligible for coverage guarantee.
# Chunks scoring below this have no meaningful overlap with the question.
MIN_COVERAGE_SCORE = 0.30


def select_top_chunks_multi(
    chunks: list[Chunk],
    queries: list[str],
    *,
    top_n: int = DEFAULT_TOP_CHUNKS,
    source_quality_map: dict[str, SourceQuality] | None = None,
) -> tuple[list[Chunk], list[RetrievalScore], dict]:
    """Run ``select_top_chunks`` for each query and merge results.

    Chunks are deduplicated by ``chunk_id``.  The best ``RetrievalScore``
    for each chunk (highest ``overall_retrieval_score``) is retained.

    Returns
    -------
    selected:
        Merged, deduplicated chunks in document/chunk order.
    retrieval_scores:
        Best score per chunk across all queries, sorted by score descending.
    retrieval_stats:
        Diagnostic dict: queries_generated, queries_executed, chunks_retrieved,
        unique_sources.
    """
    if not queries:
        return [], [], {
            "queries_generated": 0, "queries_executed": 0,
            "chunks_retrieved": 0, "unique_sources": 0,
        }

    best_score_by_id: dict[str, RetrievalScore] = {}
    best_chunk_by_id: dict[str, Chunk] = {}

    for query in queries:
        selected, scores = select_top_chunks(
            chunks, query, top_n=top_n, source_quality_map=source_quality_map
        )
        # Map chunk_id → Chunk for selected set
        selected_ids = {c.chunk_id for c in selected}
        for rs in scores:
            if rs.chunk_id not in selected_ids:
                continue
            existing = best_score_by_id.get(rs.chunk_id)
            if existing is None or rs.overall_retrieval_score > existing.overall_retrieval_score:
                best_score_by_id[rs.chunk_id] = rs
        for chunk in selected:
            if chunk.chunk_id not in best_chunk_by_id:
                best_chunk_by_id[chunk.chunk_id] = chunk

    merged_chunks = list(best_chunk_by_id.values())
    merged_chunks.sort(key=lambda c: (c.document_name, c.chunk_number))

    merged_scores = sorted(
        best_score_by_id.values(),
        key=lambda rs: -rs.overall_retrieval_score,
    )

    stats = {
        "queries_generated": len(queries),
        "queries_executed": len(queries),
        "chunks_retrieved": len(merged_chunks),
        "unique_sources": len({c.document_name for c in merged_chunks}),
    }
    return merged_chunks, merged_scores, stats


def select_top_chunks(
    chunks: list[Chunk],
    question: str,
    *,
    top_n: int = DEFAULT_TOP_CHUNKS,
    source_quality_map: dict[str, SourceQuality] | None = None,
) -> tuple[list[Chunk], list[RetrievalScore]]:
    """Score all chunks and return the top N by retrieval score plus all scores.

    Every document whose best chunk scores at or above MIN_COVERAGE_SCORE is
    guaranteed to have at least one chunk in the selection, even if its chunks
    would otherwise be displaced by higher-scoring chunks from larger documents.
    This prevents single-chunk or short synthetic documents from being silently
    excluded when they are genuinely relevant to the question.

    Returns
    -------
    selected:
        Top ``top_n`` chunks sorted back into document/chunk order.
    retrieval_scores:
        RetrievalScore for every chunk (not just selected), sorted by score
        descending.
    """
    question_terms = _extract_question_terms(question)
    detected_topics = _detect_question_topics(question)

    scored_pairs: list[tuple[Chunk, RetrievalScore]] = []
    for chunk in chunks:
        rs = score_retrieval(chunk, question, question_terms, detected_topics, source_quality_map)
        scored_pairs.append((chunk, rs))

    # Sort by overall score descending; tie-break by document/chunk order
    scored_pairs.sort(
        key=lambda pair: (-pair[1].overall_retrieval_score, pair[0].document_name, pair[0].chunk_number)
    )

    all_scores = [rs for _, rs in scored_pairs]

    # --- Phase 1: standard top-N selection ---
    selected_ids: set[str] = set()
    selected_pairs: list[tuple[Chunk, RetrievalScore]] = []
    for pair in scored_pairs[:top_n]:
        selected_ids.add(pair[0].chunk_id)
        selected_pairs.append(pair)

    # --- Phase 2: coverage guarantee ---
    # For each document not yet represented, promote its best-scoring chunk
    # if that chunk clears the minimum threshold.
    covered_docs: set[str] = {chunk.document_name for chunk, _ in selected_pairs}
    for chunk, rs in scored_pairs:
        if rs.overall_retrieval_score < MIN_COVERAGE_SCORE:
            break  # list is sorted; nothing below this can qualify
        if chunk.document_name not in covered_docs and chunk.chunk_id not in selected_ids:
            selected_ids.add(chunk.chunk_id)
            selected_pairs.append((chunk, rs))
            covered_docs.add(chunk.document_name)

    # Restore reading order
    selected_pairs.sort(key=lambda pair: (pair[0].document_name, pair[0].chunk_number))
    top_chunks_ordered = [chunk for chunk, _ in selected_pairs]

    return top_chunks_ordered, all_scores
