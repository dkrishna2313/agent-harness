"""Document chunking for the research workflow."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .chunk_classifier import ChunkClassification, classify_chunk, PRIORITY_BOOST
from .schemas import Chunk, ChunkDiagnostic, EvidenceItem, SourceDocument

CHUNK_TARGET = 7_000
CHUNK_MAX = 8_000

# Characters sent to Claude per extraction call.  Generous enough to cover
# ~30 chunks of 7 000 chars each without hitting context limits.
CHUNK_SELECTION_BUDGET = 200_000

_QUESTION_STOPWORDS = {
    "a", "an", "are", "as", "at", "be", "by", "do", "does", "for",
    "how", "in", "is", "it", "its", "of", "on", "or", "the", "to",
    "what", "which", "with",
}


def select_relevant_chunks(
    chunks: list[Chunk],
    question: str,
    *,
    max_total_chars: int = CHUNK_SELECTION_BUDGET,
) -> tuple[list[Chunk], dict[str, tuple[float, int]]]:
    """Select chunks by keyword relevance to the question.

    JH1: chunk classification is applied before selection.  Boilerplate and
    reference chunks are excluded; evidence-dense chunks receive a priority
    boost; candidate signals contribute an additional score term.

    Returns
    -------
    selected:
        Chunks chosen for extraction, sorted back into document order so
        Claude receives them in a coherent reading sequence.
    scores:
        Mapping of ``chunk_id -> (relevance_score, evidence_candidate_count)``
        for *every* chunk (selected or not), used to build diagnostics.
    """
    question_terms = _extract_question_terms(question)

    # ── JH1: classify every chunk upfront ───────────────────────────────────
    classifications: dict[str, ChunkClassification] = {
        chunk.chunk_id: classify_chunk(chunk.chunk_id, chunk.text)
        for chunk in chunks
    }

    scored: list[tuple[Chunk, float, int]] = []
    scores: dict[str, tuple[float, int]] = {}
    for chunk in chunks:
        rel = score_chunk_relevance(chunk, question_terms)
        candidates = count_evidence_candidates(chunk, question_terms)
        scores[chunk.chunk_id] = (rel, candidates)

        clf = classifications[chunk.chunk_id]
        # Skip boilerplate/reference entirely — never select them
        if clf.extraction_priority == "skip":
            scored.append((chunk, -1.0, candidates))
            continue

        # Combine relevance, priority boost, and signal score
        priority_boost = PRIORITY_BOOST.get(clf.extraction_priority, 0.0)
        signal_boost = clf.candidate_signals.signal_score * 0.25
        combined = rel + priority_boost * 0.30 + signal_boost
        scored.append((chunk, combined, candidates))

    # Sort by combined score descending; tie-break by document/chunk order.
    scored.sort(key=lambda x: (-x[1], x[0].document_name, x[0].chunk_number))

    selected: list[Chunk] = []
    total = 0
    for chunk, combined_rel, _cand in scored:
        if combined_rel < 0:
            continue   # skip boilerplate
        if total + chunk.char_count <= max_total_chars:
            selected.append(chunk)
            total += chunk.char_count

    # Restore reading order for Claude.
    selected.sort(key=lambda c: (c.document_name, c.chunk_number))
    return selected, scores


def score_chunk_relevance(chunk: Chunk, question_terms: set[str]) -> float:
    """Return 0–1 fraction of question terms present in chunk text."""
    if not question_terms:
        return 0.0
    text_lower = chunk.text.lower()
    matched = sum(1 for t in question_terms if t in text_lower)
    return round(matched / len(question_terms), 4)


def count_evidence_candidates(chunk: Chunk, question_terms: set[str]) -> int:
    """Count sentences in the chunk that contain at least one question term."""
    if not question_terms:
        return 0
    sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
    return sum(
        1 for s in sentences if any(t in s.lower() for t in question_terms)
    )


def compute_chunk_diagnostics(
    chunks: list[Chunk],
    selected_chunks: list[Chunk],
    evidence: list[EvidenceItem],
    scores: dict[str, tuple[float, int]],
) -> list[ChunkDiagnostic]:
    """Build a per-chunk diagnostic record for the trace (JH1: adds classification fields)."""
    selected_ids = {c.chunk_id for c in selected_chunks}

    evidence_per_chunk: dict[str, int] = {}
    for item in evidence:
        if item.source_chunk_id:
            evidence_per_chunk[item.source_chunk_id] = (
                evidence_per_chunk.get(item.source_chunk_id, 0) + 1
            )

    # JH1: classify all chunks for diagnostic enrichment
    classifications: dict[str, ChunkClassification] = {
        chunk.chunk_id: classify_chunk(chunk.chunk_id, chunk.text)
        for chunk in chunks
    }

    diagnostics: list[ChunkDiagnostic] = []
    for chunk in chunks:
        rel_score, candidate_count = scores.get(chunk.chunk_id, (0.0, 0))
        sent = chunk.chunk_id in selected_ids
        items_created = evidence_per_chunk.get(chunk.chunk_id, 0)
        clf = classifications.get(chunk.chunk_id)

        if not sent:
            if clf and clf.extraction_priority == "skip":
                decision = "not_sent"
                reason: str | None = f"skipped: {clf.chunk_type} ({clf.classification_reason[:80]})"
            elif rel_score == 0.0:
                decision = "not_sent"
                reason = "not relevant to question"
            else:
                decision = "not_sent"
                reason = "excluded by character budget"
        elif items_created > 0:
            decision = "accepted"
            reason = None
        else:
            decision = "rejected"
            reason = "no evidence extracted"

        diagnostics.append(
            ChunkDiagnostic(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                chunk_size=chunk.char_count,
                relevance_score=rel_score,
                evidence_candidate_count=candidate_count,
                sent_to_claude=sent,
                evidence_items_created=items_created,
                extraction_decision=decision,
                rejection_reason=reason,
                # JH1 fields
                chunk_type=clf.chunk_type if clf else "unknown",
                extraction_priority=clf.extraction_priority if clf else "medium",
                candidate_signals=clf.candidate_signals.to_dict() if clf else {},
                classification_reason=clf.classification_reason if clf else "",
            )
        )

    return diagnostics


def compute_evidence_yield_metrics(
    chunks: list[Chunk],
    selected_chunks: list[Chunk],
    evidence: list[EvidenceItem],
    documents_loaded: int = 0,
) -> dict:
    """Compute evidence yield metrics for the trace (JH1).

    Returns a dict with keys matching the JH1 spec:
        documents_loaded, chunks_total, chunks_selected, chunks_with_evidence,
        evidence_items_created, zero_evidence_chunks, zero_evidence_selected_chunks,
        skipped_boilerplate_chunks, yield_per_selected_chunk, yield_per_total_chunk.
    """
    chunks_total = len(chunks)
    chunks_selected = len(selected_chunks)
    evidence_items_created = len(evidence)

    # Chunks that produced at least one evidence item
    chunks_with_evidence = len({
        item.source_chunk_id for item in evidence if item.source_chunk_id
    })

    # Classify all chunks to count boilerplate
    skipped_boilerplate = sum(
        1 for chunk in chunks
        if classify_chunk(chunk.chunk_id, chunk.text).extraction_priority == "skip"
    )

    zero_evidence_chunks = chunks_total - chunks_with_evidence
    zero_evidence_selected_chunks = max(0, chunks_selected - chunks_with_evidence)

    yield_per_selected = (
        round(evidence_items_created / chunks_selected, 3)
        if chunks_selected > 0 else 0.0
    )
    yield_per_total = (
        round(evidence_items_created / chunks_total, 3)
        if chunks_total > 0 else 0.0
    )

    return {
        "documents_loaded":             documents_loaded,
        "chunks_total":                 chunks_total,
        "chunks_selected":              chunks_selected,
        "chunks_with_evidence":         chunks_with_evidence,
        "evidence_items_created":       evidence_items_created,
        "zero_evidence_chunks":         zero_evidence_chunks,
        "zero_evidence_selected_chunks": zero_evidence_selected_chunks,
        "skipped_boilerplate_chunks":   skipped_boilerplate,
        "yield_per_selected_chunk":     yield_per_selected,
        "yield_per_total_chunk":        yield_per_total,
    }


def _extract_question_terms(question: str) -> set[str]:
    """Extract meaningful single-word terms from a question."""
    words = re.findall(r"[a-z][a-z0-9-]{1,}", question.lower())
    return {w for w in words if w not in _QUESTION_STOPWORDS}


def chunk_documents(
    docs: Sequence[SourceDocument],
    *,
    target_size: int = CHUNK_TARGET,
    max_size: int = CHUNK_MAX,
) -> list[Chunk]:
    """Chunk all documents and return a flat list of Chunk objects."""
    result: list[Chunk] = []
    for doc in docs:
        result.extend(chunk_document(doc, target_size=target_size, max_size=max_size))
    return result


def chunk_document(
    doc: SourceDocument,
    *,
    target_size: int = CHUNK_TARGET,
    max_size: int = CHUNK_MAX,
) -> list[Chunk]:
    """Split one document into Chunk objects."""
    text = doc.text
    if not text:
        return []

    if len(text) <= max_size:
        return [
            Chunk(
                chunk_id=_chunk_id(doc.path.name, 1),
                document_name=doc.path.name,
                chunk_number=1,
                text=text,
                start_offset=0,
                end_offset=len(text),
            )
        ]

    sentence_ends = _find_sentence_ends(text)
    chunks: list[Chunk] = []
    pos = 0
    chunk_number = 1

    while pos < len(text):
        target_end = pos + target_size
        max_end = pos + max_size

        if max_end >= len(text):
            # Last chunk: take the rest
            chunk_text = text[pos:]
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.path.name, chunk_number),
                    document_name=doc.path.name,
                    chunk_number=chunk_number,
                    text=chunk_text,
                    start_offset=pos,
                    end_offset=len(text),
                )
            )
            break

        split = _best_sentence_end(sentence_ends, pos, target_end, max_end)
        if split is None:
            split = max_end

        chunk_text = text[pos:split]
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc.path.name, chunk_number),
                document_name=doc.path.name,
                chunk_number=chunk_number,
                text=chunk_text,
                start_offset=pos,
                end_offset=split,
            )
        )
        pos = split
        chunk_number += 1

    return chunks


def _chunk_id(document_name: str, chunk_number: int) -> str:
    """Generate a stable chunk ID from a document name and chunk number."""
    stem = re.sub(r"[^a-zA-Z0-9]", "_", document_name)[:24]
    return f"{stem}_C{chunk_number:03d}"


def _find_sentence_ends(text: str) -> list[int]:
    """Return a sorted list of positions just after sentence-ending punctuation."""
    return [m.start() for m in re.finditer(r"(?<=[.!?])\s+", text)]


def _best_sentence_end(
    sentence_ends: list[int],
    chunk_start: int,
    target_end: int,
    max_end: int,
) -> int | None:
    """Return the best split position within (chunk_start, max_end]."""
    # Prefer first sentence end in [target_end, max_end)
    for end in sentence_ends:
        if target_end <= end < max_end:
            return end

    # Fall back to last sentence end in (chunk_start, max_end)
    last = None
    for end in sentence_ends:
        if chunk_start < end < max_end:
            last = end
    return last
