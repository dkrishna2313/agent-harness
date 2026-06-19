"""Document chunking for the research workflow."""

from __future__ import annotations

import re
from collections.abc import Sequence

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

    scored: list[tuple[Chunk, float, int]] = []
    scores: dict[str, tuple[float, int]] = {}
    for chunk in chunks:
        rel = score_chunk_relevance(chunk, question_terms)
        candidates = count_evidence_candidates(chunk, question_terms)
        scored.append((chunk, rel, candidates))
        scores[chunk.chunk_id] = (rel, candidates)

    # Sort by relevance descending; break ties by document/chunk order so we
    # draw evenly from all documents rather than exhausting one first.
    scored.sort(key=lambda x: (-x[1], x[0].document_name, x[0].chunk_number))

    selected: list[Chunk] = []
    total = 0
    for chunk, _rel, _cand in scored:
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
    """Build a per-chunk diagnostic record for the trace."""
    selected_ids = {c.chunk_id for c in selected_chunks}

    evidence_per_chunk: dict[str, int] = {}
    for item in evidence:
        if item.source_chunk_id:
            evidence_per_chunk[item.source_chunk_id] = (
                evidence_per_chunk.get(item.source_chunk_id, 0) + 1
            )

    diagnostics: list[ChunkDiagnostic] = []
    for chunk in chunks:
        rel_score, candidate_count = scores.get(chunk.chunk_id, (0.0, 0))
        sent = chunk.chunk_id in selected_ids
        items_created = evidence_per_chunk.get(chunk.chunk_id, 0)

        if not sent:
            decision = "not_sent"
            reason: str | None = (
                "not relevant to question"
                if rel_score == 0.0
                else "excluded by character budget"
            )
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
            )
        )

    return diagnostics


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
