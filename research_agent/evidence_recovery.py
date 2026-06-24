"""Evidence recovery pass for high-signal zero-yield chunks (JH1a).

After normal evidence extraction, some selected evidence-dense chunks contain
no extracted evidence items — either because no topic keyword matched, or because
the extraction path operates at document level and never sets source_chunk_id.

This module provides two complementary fixes:

1.  ``attribute_evidence_to_chunks`` — after extraction, scan each evidence
    item's snippet against chunk texts to set source_chunk_id.  This makes yield
    metrics accurate even before any recovery.

2.  ``run_recovery_pass`` — for chunks that are still zero-evidence after
    attribution, run a signal-driven extraction using the candidate-signals regex
    patterns to pull out numeric claims, policy terms, timelines, and technical
    specs as evidence items marked ``recovered=True``.

Public API
----------
attribute_evidence_to_chunks(evidence, chunks) -> list[EvidenceItem]
find_recovery_eligible_chunks(chunks, evidence, diagnostics) -> list[Chunk]
recover_evidence_from_chunk(chunk, question, source_quality_map, profile) -> list[EvidenceItem]
run_recovery_pass(chunks, selected_chunks, evidence, diagnostics, ...) -> RecoveryResult
compute_zero_yield_documents(chunks, evidence, documents) -> list[dict]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

LOGGER = logging.getLogger(__name__)

# Minimum relevance_score (0–1) for a chunk to be eligible for recovery
_MIN_RELEVANCE_FOR_RECOVERY = 0.3

# Minimum overall_score for a recovered item to be included in final ranking
RECOVERY_MIN_OVERALL_SCORE = 3.5

# Maximum recovered items per chunk (keep strict to avoid noise)
_MAX_RECOVERED_PER_CHUNK = 4

# ---------------------------------------------------------------------------
# Sentence-level signal patterns (re-use compatible patterns from classifier)
# ---------------------------------------------------------------------------

_NUMERIC_SENTENCE_RE = re.compile(
    r"""
    (?:
        \$?\d[\d,]*\.?\d*\s*
        (?:%|MW|GW|kW|TWh|GWh|MWh|kWh|°C|°F|
           USD|GBP|EUR|billion|million|trillion|bn|mn|
           GB/s|TB/s|PB/s|
           PUE|CUE|WUE|
           years?|months?|days?|
           tons?|tonnes?)
    )
    """,
    re.VERBOSE | re.I,
)

_POLICY_SENTENCE_RE = re.compile(
    r"\b(?:FERC|NERC|ERCOT|PJM|MISO|CAISO|SPP|NYISO|ISO-NE|"
    r"IEEE|ASHRAE|NFPA|NEC|IEC|"
    r"DOE|EPA|CPUC|PUCT|NARUC|"
    r"Order\s+\d{3,4}|"
    r"interconnection\s+(?:queue|process|agreement)|"
    r"Inflation\s+Reduction\s+Act|IRA\b)",
    re.I,
)

_TIMELINE_RE = re.compile(
    r"\b(?:20\d{2}|19\d{2})\b"
    r"|Q[1-4]\s*20\d{2}"
    r"|\b\d+[\-–]\d+\s+(?:years?|months?|weeks?)\b",
    re.I,
)

_TECHNICAL_SPEC_RE = re.compile(
    r"\b(?:"
    r"\d+\s*(?:kW|MW|GW|kWh|MWh|GWh|TWh)"
    r"|\d+\s*(?:GB|TB|PB)/s"
    r"|\d+\s*(?:rack|cabinet|server|node|GPU|TPU)"
    r"|PUE\s*(?:of\s*)?[\d\.]+"
    r"|[A-Z]+\s*(?:standard|specification|requirement|compliance)"
    r")",
    re.I,
)

_MARKET_PROJECTION_RE = re.compile(
    r"\b(?:"
    r"(?:market|demand|capacity|revenue)\s+(?:will|expected|projected|forecast|estimated)\s+(?:to\s+)?(?:reach|grow|increase|exceed)"
    r"|(?:CAGR|compound\s+annual\s+growth\s+rate)"
    r"|(?:by\s+20\d{2})\s*(?:,\s*)?(?:the\s+)?(?:market|demand|capacity)"
    r")",
    re.I,
)

_GRID_CONSTRAINT_RE = re.compile(
    r"\b(?:"
    r"(?:transmission|grid|interconnection)\s+(?:constraint|limit|congestion|bottleneck|delay|backlog)"
    r"|(?:interconnection\s+queue\s+(?:of|at|exceeds|reached)\s+\d)"
    r"|(?:stranded\s+(?:asset|capacity|investment))"
    r"|(?:curtail(?:ment|ed))"
    r")",
    re.I,
)

_RECOVERY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_NUMERIC_SENTENCE_RE,    "numeric_claim"),
    (_POLICY_SENTENCE_RE,     "policy_claim"),
    (_TIMELINE_RE,            "timeline_claim"),
    (_TECHNICAL_SPEC_RE,      "technical_spec"),
    (_MARKET_PROJECTION_RE,   "market_projection"),
    (_GRID_CONSTRAINT_RE,     "grid_constraint"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RecoveryResult:
    """Output of run_recovery_pass()."""

    recovered_items: list[Any] = field(default_factory=list)      # list[EvidenceItem]
    missed_chunk_queue: list[dict] = field(default_factory=list)
    recovery_metrics: dict[str, Any] = field(default_factory=dict)
    yield_before: dict[str, Any] = field(default_factory=dict)
    yield_after: dict[str, Any] = field(default_factory=dict)
    category_normalization: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk attribution
# ---------------------------------------------------------------------------

def attribute_evidence_to_chunks(
    evidence: list,   # list[EvidenceItem]
    chunks: list,     # list[Chunk]
) -> list:
    """Set source_chunk_id on each evidence item by matching its snippet to chunks.

    This is a post-extraction attribution step that makes chunk-level yield
    metrics accurate even when extract_evidence() operates at document level.
    Items whose snippet appears verbatim (or as a substring) in a chunk get
    that chunk's chunk_id assigned.  Already-attributed items are not changed.

    Returns a new list of evidence items (does not mutate in-place).
    """
    # Build doc→chunks index for fast lookup
    doc_chunks: dict[str, list] = {}
    for chunk in chunks:
        doc_chunks.setdefault(chunk.document_name, []).append(chunk)

    attributed: list = []
    for item in evidence:
        if item.source_chunk_id:
            attributed.append(item)
            continue

        snippet = item.evidence_snippet.strip()[:200]
        found_id = ""

        # Look only at chunks from the same document
        for chunk in doc_chunks.get(item.source_document, []):
            if snippet and snippet in chunk.text:
                found_id = chunk.chunk_id
                break

        if found_id:
            attributed.append(item.model_copy(update={"source_chunk_id": found_id}))
        else:
            attributed.append(item)

    return attributed


# ---------------------------------------------------------------------------
# Recovery eligibility
# ---------------------------------------------------------------------------

def find_recovery_eligible_chunks(
    selected_chunks: list,    # list[Chunk]
    evidence: list,           # list[EvidenceItem]
    diagnostics: list,        # list[ChunkDiagnostic]
) -> list:
    """Return selected evidence-dense chunks that produced zero evidence items."""
    # Build set of chunk_ids that have at least one evidence item attributed
    chunks_with_evidence: set[str] = {
        item.source_chunk_id for item in evidence if item.source_chunk_id
    }

    # Build diagnostic lookup for relevance score and chunk_type
    diag_lookup: dict[str, Any] = {d.chunk_id: d for d in diagnostics}

    eligible: list = []
    for chunk in selected_chunks:
        if chunk.chunk_id in chunks_with_evidence:
            continue
        diag = diag_lookup.get(chunk.chunk_id)
        if diag is None:
            continue
        # Must be evidence_dense (not boilerplate/context) and have some relevance
        if diag.chunk_type not in ("evidence_dense",):
            continue
        if diag.relevance_score < _MIN_RELEVANCE_FOR_RECOVERY:
            continue
        eligible.append(chunk)

    return eligible


# ---------------------------------------------------------------------------
# Recovery extraction
# ---------------------------------------------------------------------------

def _classify_recovery_reason(sentence: str) -> str:
    """Return the first matching recovery reason for a sentence."""
    for pattern, reason in _RECOVERY_PATTERNS:
        if pattern.search(sentence):
            return reason
    return "other"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences of at least 40 chars."""
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", normalized):
        s = s.strip()
        if len(s) >= 40:
            sentences.append(s)
    return sentences


def recover_evidence_from_chunk(
    chunk: Any,          # Chunk
    question: str,
    source_quality_map: dict | None = None,
    profile: Any = None,
    out_normalizations: list | None = None,
) -> list:
    """Extract evidence from a chunk using signal-driven (not keyword) logic.

    Only explicit factual claims with measurable signals are returned.
    Inference, synthesis, and vague statements are excluded.
    Recovered items are marked recovered=True with a recovery_reason.
    """
    from .schemas import EvidenceItem
    from .source_quality import classify_source_quality
    from .agent import (
        _category_for_snippet,
        _claim_from_snippet,
        _relevance_for_snippet,
        _confidence_for_snippet,
        _keywords,
        _normalize_category,
    )

    query_terms = _keywords(question)
    sentences = _split_sentences(chunk.text)
    recovered: list[EvidenceItem] = []
    seen: set[str] = set()

    for sentence in sentences:
        # At least one signal pattern must match (explicit claim only)
        reason = _classify_recovery_reason(sentence)
        if reason == "other":
            continue  # no measurable signal — skip

        key = sentence[:100]
        if key in seen:
            continue
        seen.add(key)

        snippet = sentence[:500]
        raw_category = _category_for_snippet(snippet, profile)
        category = _normalize_category(raw_category)
        if out_normalizations is not None and raw_category != category:
            out_normalizations.append({"raw": raw_category, "normalized": category})
        score = sum(1 for t in query_terms if t in snippet.lower())

        if source_quality_map and chunk.document_name in source_quality_map:
            sq = source_quality_map[chunk.document_name]
            source_quality_score = sq.source_quality_score
        else:
            sq = classify_source_quality(chunk.document_name)
            source_quality_score = sq.source_quality_score

        recovered.append(
            EvidenceItem(
                claim=_claim_from_snippet(snippet, category),
                source_document=chunk.document_name,
                source_chunk_id=chunk.chunk_id,
                evidence_snippet=snippet,
                category=category,
                relevance=_relevance_for_snippet(snippet, question, score),
                confidence=_confidence_for_snippet(snippet, score),
                source_quality_score=source_quality_score,
                recovered=True,
                recovery_reason=reason,
            )
        )

        if len(recovered) >= _MAX_RECOVERED_PER_CHUNK:
            break

    return recovered


# ---------------------------------------------------------------------------
# Recovery pass
# ---------------------------------------------------------------------------

def run_recovery_pass(
    chunks: list,
    selected_chunks: list,
    evidence: list,           # list[EvidenceItem] — already attributed
    diagnostics: list,        # list[ChunkDiagnostic]
    *,
    question: str,
    source_quality_map: dict | None = None,
    profile: Any = None,
) -> RecoveryResult:
    """Run the full evidence recovery pipeline.

    1. Compute yield_before from current evidence + selected_chunks.
    2. Find eligible chunks (evidence_dense, zero-evidence, rel >= threshold).
    3. For each eligible chunk, run recovery extraction.
    4. Score and filter recovered items (overall_score >= RECOVERY_MIN_OVERALL_SCORE).
    5. Build missed_chunk_queue and recovery_metrics.
    6. Compute yield_after.
    """
    from .agent import score_evidence_items
    from .schemas import assign_evidence_ids
    from .chunker import compute_evidence_yield_metrics

    # Yield before recovery
    yield_before = compute_evidence_yield_metrics(
        chunks, selected_chunks, evidence, documents_loaded=0
    )
    yield_before_snapshot = {
        "chunks_selected":              yield_before["chunks_selected"],
        "chunks_with_evidence":         yield_before["chunks_with_evidence"],
        "evidence_items_created":       yield_before["evidence_items_created"],
        "zero_evidence_selected_chunks": yield_before["zero_evidence_selected_chunks"],
    }

    eligible = find_recovery_eligible_chunks(selected_chunks, evidence, diagnostics)
    LOGGER.debug("[RecoveryPass] %d eligible chunks for recovery", len(eligible))

    all_raw_recovered: list = []
    chunk_recovery_map: dict[str, int] = {}   # chunk_id → recovered count
    all_normalizations: list[dict[str, str]] = []

    for chunk in eligible:
        raw = recover_evidence_from_chunk(
            chunk, question,
            source_quality_map=source_quality_map,
            profile=profile,
            out_normalizations=all_normalizations,
        )
        chunk_recovery_map[chunk.chunk_id] = len(raw)
        all_raw_recovered.extend(raw)

    # Score recovered items
    if all_raw_recovered:
        scored = score_evidence_items(question, assign_evidence_ids(all_raw_recovered),
                                      source_quality_map, profile)
        # Filter: only include items above minimum quality threshold
        kept = [item for item in scored if item.overall_score >= RECOVERY_MIN_OVERALL_SCORE]
    else:
        kept = []

    LOGGER.debug(
        "[RecoveryPass] %d raw recovered → %d kept (score >= %.1f)",
        len(all_raw_recovered), len(kept), RECOVERY_MIN_OVERALL_SCORE,
    )

    # Build missed_chunk_queue
    diag_lookup = {d.chunk_id: d for d in diagnostics}
    missed_queue: list[dict] = []
    for chunk in eligible:
        diag = diag_lookup.get(chunk.chunk_id)
        count = chunk_recovery_map.get(chunk.chunk_id, 0)
        missed_queue.append({
            "chunk_id":               chunk.chunk_id,
            "document_name":          chunk.document_name,
            "relevance_score":        diag.relevance_score if diag else 0.0,
            "candidate_signals":      diag.candidate_signals if diag else {},
            "original_chunk_type":    diag.chunk_type if diag else "unknown",
            "recovery_attempted":     True,
            "recovered_evidence_count": count,
        })

    # Recovery metrics
    chunks_recovered = sum(1 for c in chunk_recovery_map.values() if c > 0)
    recovery_metrics = {
        "eligible_chunks":        len(eligible),
        "recovery_attempted":     len(eligible),
        "chunks_recovered":       chunks_recovered,
        "recovered_evidence_items": len(kept),
        "recovery_yield": round(len(kept) / len(eligible), 3) if eligible else 0.0,
    }

    # Yield after recovery (combine existing + recovered)
    combined = list(evidence) + kept
    yield_after_raw = compute_evidence_yield_metrics(
        chunks, selected_chunks, combined, documents_loaded=0
    )
    yield_after_snapshot = {
        "chunks_selected":              yield_after_raw["chunks_selected"],
        "chunks_with_evidence":         yield_after_raw["chunks_with_evidence"],
        "evidence_items_created":       yield_after_raw["evidence_items_created"],
        "zero_evidence_selected_chunks": yield_after_raw["zero_evidence_selected_chunks"],
    }

    return RecoveryResult(
        recovered_items=kept,
        missed_chunk_queue=missed_queue,
        recovery_metrics=recovery_metrics,
        yield_before=yield_before_snapshot,
        yield_after=yield_after_snapshot,
        category_normalization=all_normalizations,
    )


# ---------------------------------------------------------------------------
# Zero-yield document diagnostics
# ---------------------------------------------------------------------------

def compute_zero_yield_documents(
    chunks: list,
    evidence: list,
    documents: Sequence[Any],
) -> list[dict]:
    """Identify documents with many chunks but zero evidence items.

    Returns a list of dicts with document diagnostics and a recommendation.
    """
    # Count chunks per document
    chunks_per_doc: dict[str, int] = {}
    chars_per_doc: dict[str, int] = {}
    for chunk in chunks:
        chunks_per_doc[chunk.document_name] = chunks_per_doc.get(chunk.document_name, 0) + 1
        chars_per_doc[chunk.document_name] = chars_per_doc.get(chunk.document_name, 0) + chunk.char_count

    # Count evidence per document
    evidence_per_doc: dict[str, int] = {}
    for item in evidence:
        evidence_per_doc[item.source_document] = evidence_per_doc.get(item.source_document, 0) + 1

    zero_yield: list[dict] = []
    for doc_name, chunk_count in chunks_per_doc.items():
        ev_count = evidence_per_doc.get(doc_name, 0)
        if ev_count == 0:
            chars = chars_per_doc.get(doc_name, 0)
            if chunk_count >= 3:
                recommendation = "inspect_parser_output"
            elif chars > 50_000:
                recommendation = "recover_sections"
            else:
                recommendation = "lower_priority"

            zero_yield.append({
                "document_name":  doc_name,
                "chunks":         chunk_count,
                "characters":     chars,
                "evidence_items": 0,
                "recommendation": recommendation,
            })

    return sorted(zero_yield, key=lambda d: -d["chunks"])
