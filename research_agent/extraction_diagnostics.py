"""Failure pipeline diagnostics for evidence extraction (JH1b).

Answers the question: **where is evidence being lost?**

Architecture discovery
----------------------
In the mock (deterministic) path, extraction works at DOCUMENT level inside
``extract_evidence()`` — it is NOT per-chunk. The retrieval system selects
chunks for budget/relevance ranking, but those selected chunks are not passed
to ``extract_evidence()``. The function operates on ``document.text`` directly
using topic-keyword sentence matching.

Chunk-level ``evidence_items_created`` counts are populated by
``attribute_evidence_to_chunks()``, which tries to match each evidence
snippet back to a chunk via verbatim substring search. That search fails
because ``extract_evidence()`` compacts whitespace (via ``_compact_whitespace``)
before creating the snippet, while ``chunk.text`` preserves original whitespace.
This is the dominant root cause of zero-evidence readings for "sent" chunks.

Failure stages
--------------
ATTRIBUTION_FAILURE   Evidence items exist for this document; a snippet from
                      this doc matches the chunk when both are whitespace-
                      normalised, but the raw substring match in
                      attribute_evidence_to_chunks() fails → attribution bug.

CROSS_CHUNK           Evidence from this document is attributed to a different
                      chunk (snippet doesn't appear in this chunk even
                      normalised) → extraction happened elsewhere in the doc.

EMPTY_EXTRACTION      No topic keyword from the active profile matches any
                      sentence in this chunk's text. Nothing to extract here.

BUDGET_EXCLUSION      Chunk was not in selected_chunks; character budget was
                      exhausted before this chunk could be included.

PARSER_FAILURE        Chunk text is mostly non-ASCII or extremely sparse
                      relative to its byte size → PDF parser likely failed.

SCHEMA_VALIDATION_FAILURE  Real-path only: LLM items failed Pydantic validation.

QUALITY_THRESHOLD_REJECTION  Items found but discarded by score thresholds or
                              duplicate suppression inside extract_evidence().

POST_PROCESSING_REJECTION  Items sanitised away by evidence_filter.

NO_LLM_OUTPUT         Real-path only: LLM returned an empty payload.

UNKNOWN               Failure reason could not be determined.

Public API
----------
classify_chunk_failure(chunk, diag, doc_evidence, topic_term_sets) -> tuple[str, str]
build_failure_diagnostics(chunk_diags, chunks, evidence, topic_term_sets, ...) -> list[dict]
build_failure_summary(failure_diagnostics) -> dict[str, int]
compute_top_missed_chunks(failure_diagnostics, n=10) -> list[dict]
analyze_document_failures(doc_name, failure_diagnostics) -> dict
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------

ATTRIBUTION_FAILURE           = "ATTRIBUTION_FAILURE"
CROSS_CHUNK                   = "CROSS_CHUNK"
EMPTY_EXTRACTION              = "EMPTY_EXTRACTION"
BUDGET_EXCLUSION              = "BUDGET_EXCLUSION"
PARSER_FAILURE                = "PARSER_FAILURE"
SCHEMA_VALIDATION_FAILURE     = "SCHEMA_VALIDATION_FAILURE"
QUALITY_THRESHOLD_REJECTION   = "QUALITY_THRESHOLD_REJECTION"
DUPLICATE_SUPPRESSION         = "DUPLICATE_SUPPRESSION"
POST_PROCESSING_REJECTION     = "POST_PROCESSING_REJECTION"
NO_LLM_OUTPUT                 = "NO_LLM_OUTPUT"
UNKNOWN                       = "UNKNOWN"

# Spec-required stages — these must appear as keys in failure_summary output
SPEC_FAILURE_STAGES: tuple[str, ...] = (
    NO_LLM_OUTPUT,
    EMPTY_EXTRACTION,
    PARSER_FAILURE,
    SCHEMA_VALIDATION_FAILURE,
    QUALITY_THRESHOLD_REJECTION,
    DUPLICATE_SUPPRESSION,
    POST_PROCESSING_REJECTION,
    UNKNOWN,
)

# Full set including discovery stages added by JH1b analysis
ALL_FAILURE_STAGES: tuple[str, ...] = SPEC_FAILURE_STAGES + (
    ATTRIBUTION_FAILURE,
    CROSS_CHUNK,
    BUDGET_EXCLUSION,
)

# Signal threshold for parser failure detection
_MIN_ASCII_RATIO = 0.70
_MIN_PRINTABLE_CHARS = 50

# ---------------------------------------------------------------------------
# Whitespace normalisation (mirrors agent._compact_whitespace)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Parser-failure detection
# ---------------------------------------------------------------------------

def _is_garbled(text: str) -> bool:
    """Return True when the text appears to be binary/garbled PDF output."""
    if not text or len(text.strip()) < _MIN_PRINTABLE_CHARS:
        return True
    printable = sum(1 for c in text if c.isprintable())
    ratio = printable / max(1, len(text))
    return ratio < _MIN_ASCII_RATIO


# ---------------------------------------------------------------------------
# Topic-term matching on chunk text
# ---------------------------------------------------------------------------

def _chunk_has_topic_match(chunk_text: str, topic_term_sets: dict[str, set[str]]) -> bool:
    """Return True if any topic keyword from the profile appears in chunk_text."""
    lower = chunk_text.lower()
    return any(
        any(term in lower for term in terms)
        for terms in topic_term_sets.values()
    )


def _sentences_with_topic_match(
    chunk_text: str,
    topic_term_sets: dict[str, set[str]],
) -> list[str]:
    """Return sentences in chunk_text that contain at least one topic keyword."""
    sentences = re.split(r"(?<=[.!?])\s+", _normalise(chunk_text))
    matched = []
    for s in sentences:
        lower = s.lower()
        if any(any(term in lower for term in terms) for terms in topic_term_sets.values()):
            matched.append(s)
    return matched


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_chunk_failure(
    chunk: Any,            # Chunk schema object
    diag: Any,             # ChunkDiagnostic schema object OR dict
    doc_evidence: list,    # list[EvidenceItem] for this document (may be empty)
    topic_term_sets: dict[str, set[str]],
) -> tuple[str, str]:
    """Classify why a zero-evidence chunk produced no evidence.

    Returns (stage, reason) — both human-readable strings.
    """
    chunk_text = chunk.text if hasattr(chunk, "text") else ""
    sent = diag.sent_to_claude if hasattr(diag, "sent_to_claude") else diag.get("sent_to_claude", False)

    # 1. Budget exclusion — chunk never entered the extraction path
    if not sent:
        return BUDGET_EXCLUSION, "chunk excluded from selected_chunks by character budget"

    # 2. Parser failure — text is garbled
    if _is_garbled(chunk_text):
        return PARSER_FAILURE, "chunk text appears garbled or binary (PDF parser failure)"

    norm_chunk = _normalise(chunk_text)

    if doc_evidence:
        # 3. Attribution failure — evidence exists AND snippet matches normalized chunk
        for ev in doc_evidence:
            snippet = ev.evidence_snippet if hasattr(ev, "evidence_snippet") else ev.get("evidence_snippet", "")
            norm_snippet = _normalise(snippet.strip()[:300])
            if norm_snippet and norm_snippet in norm_chunk:
                return (
                    ATTRIBUTION_FAILURE,
                    f"evidence snippet matches chunk when whitespace-normalised but raw "
                    f"substring match fails — attribute_evidence_to_chunks() does not "
                    f"normalise before comparing (snippet[:60]={norm_snippet[:60]!r})",
                )

        # 4. Cross-chunk — evidence is in this doc but from a different chunk
        return (
            CROSS_CHUNK,
            f"evidence exists for this document ({len(doc_evidence)} items) "
            f"but none match this chunk's text — evidence was extracted from a "
            f"different section/chunk of the document",
        )

    # 5. No evidence from doc at all — check topic keyword match
    if not _chunk_has_topic_match(chunk_text, topic_term_sets):
        return (
            EMPTY_EXTRACTION,
            "no topic keyword from the active profile matches any sentence "
            "in this chunk's text",
        )

    # 6. Topic matches exist but no evidence was created — dedup / max-cap
    # In the mock path, _append_evidence_item suppresses items via (category, text)
    # dedup key or the per-document max_items cap.  Neither leaves a trace, so
    # we classify these as DUPLICATE_SUPPRESSION (the most likely cause when
    # topic keywords are present but the document already hit its item ceiling).
    return (
        DUPLICATE_SUPPRESSION,
        "topic keywords match in this chunk but no evidence was created — "
        "likely duplicate-key suppression or per-document max-items cap in "
        "_append_evidence_item(); a prior chunk from this document may have "
        "already extracted the same sentences",
    )


# ---------------------------------------------------------------------------
# Signal strength
# ---------------------------------------------------------------------------

def _signal_strength(diag: Any) -> int:
    """Compute composite signal strength from candidate_signals (0-100)."""
    signals: dict = (
        diag.candidate_signals if hasattr(diag, "candidate_signals")
        else diag.get("candidate_signals", {})
    )
    return min(
        100,
        signals.get("numeric_claim_count", 0)
        + signals.get("named_entity_count", 0)
        + signals.get("unit_count", 0) * 3
        + signals.get("policy_or_standard_terms", 0) * 2
        + signals.get("date_count", 0)
        + signals.get("comparative_claim_count", 0) * 2,
    )


# ---------------------------------------------------------------------------
# Build full failure diagnostics list
# ---------------------------------------------------------------------------

def build_failure_diagnostics(
    chunk_diags: list,        # list[ChunkDiagnostic] or list[dict]
    chunks: list,             # list[Chunk]
    evidence: list,           # list[EvidenceItem] — full attributed set
    topic_term_sets: dict[str, set[str]],
    *,
    is_mock: bool = True,
    llm_raw_responses: dict[str, str] | None = None,   # chunk_id → raw LLM text (real path)
    validated_raw: dict[str, list] | None = None,       # chunk_id → raw parsed items
    sanitize_rejected: dict[str, list] | None = None,  # chunk_id → rejected items
) -> list[dict]:
    """Produce one failure_diagnostic dict for every zero-evidence evidence-dense chunk.

    Parameters
    ----------
    chunk_diags:
        ChunkDiagnostic records for all chunks.
    chunks:
        Chunk objects used to read chunk text.
    evidence:
        Final attributed evidence items (used to classify failure stage).
    topic_term_sets:
        Profile topic → keyword set mapping.
    is_mock:
        True for the deterministic mock path.
    llm_raw_responses:
        Real-path only: raw text returned by LLM for each chunk_id.
    validated_raw / sanitize_rejected:
        Real-path only: intermediate extraction results.
    """
    # Build lookup maps
    chunk_map: dict[str, Any] = {
        (c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]): c
        for c in chunks
    }

    # Group evidence by document
    doc_evidence: dict[str, list] = defaultdict(list)
    for ev in evidence:
        doc_name = ev.source_document if hasattr(ev, "source_document") else ev.get("source_document", "")
        doc_evidence[doc_name].append(ev)

    llm_raw = llm_raw_responses or {}
    v_raw = validated_raw or {}
    s_rejected = sanitize_rejected or {}

    results: list[dict] = []

    for d in chunk_diags:
        # Work with both Pydantic objects and plain dicts
        cid = d.chunk_id if hasattr(d, "chunk_id") else d["chunk_id"]
        doc_name = d.document_name if hasattr(d, "document_name") else d["document_name"]
        chunk_type = d.chunk_type if hasattr(d, "chunk_type") else d.get("chunk_type", "unknown")
        ev_count = (
            d.evidence_items_created if hasattr(d, "evidence_items_created")
            else d.get("evidence_items_created", 0)
        )
        sent = d.sent_to_claude if hasattr(d, "sent_to_claude") else d.get("sent_to_claude", False)
        rel_score = d.relevance_score if hasattr(d, "relevance_score") else d.get("relevance_score", 0.0)
        signals = (
            dict(d.candidate_signals) if hasattr(d, "candidate_signals")
            else d.get("candidate_signals", {})
        )

        # Only diagnose zero-evidence evidence-dense chunks
        if ev_count > 0:
            continue
        if chunk_type != "evidence_dense":
            continue

        chunk_obj = chunk_map.get(cid)
        if chunk_obj is None:
            results.append({
                "chunk_id": cid,
                "document_name": doc_name,
                "relevance_score": rel_score,
                "candidate_signals": signals,
                "llm_invoked": False,
                "llm_response_received": False,
                "raw_extraction_count": 0,
                "parsed_extraction_count": 0,
                "validated_extraction_count": 0,
                "accepted_extraction_count": 0,
                "failure_stage": UNKNOWN,
                "failure_reason": "chunk object not found in chunk map",
                "raw_llm_extraction_response": "",
                "parser_output": [],
                "validation_results": {"passed": 0, "failed": 0, "reasons": []},
                "signal_strength": _signal_strength(d),
            })
            continue

        stage, reason = classify_chunk_failure(
            chunk_obj,
            d,
            doc_evidence.get(doc_name, []),
            topic_term_sets,
        )

        # Parser output: sentences in this chunk that match topic terms
        chunk_text = chunk_obj.text if hasattr(chunk_obj, "text") else chunk_obj.get("text", "")
        parser_sentences = _sentences_with_topic_match(chunk_text, topic_term_sets)

        # Real-path LLM data (may be empty in mock mode)
        raw_llm = llm_raw.get(cid, (
            "[mock: document-level deterministic extraction — no per-chunk LLM call; "
            "extraction runs on document.text via _find_topic_chunks_multi()]"
        ) if is_mock else "")

        raw_items = v_raw.get(cid, [])
        rejected_items = s_rejected.get(cid, [])
        validated_count = len(raw_items)
        sanitize_fail_reasons = [str(r) for r in rejected_items]

        results.append({
            "chunk_id": cid,
            "document_name": doc_name,
            "relevance_score": rel_score,
            "candidate_signals": signals,
            "llm_invoked": sent and not is_mock,
            "llm_response_received": bool(raw_llm) and not is_mock,
            "raw_extraction_count": len(raw_items),
            "parsed_extraction_count": len(parser_sentences),
            "validated_extraction_count": validated_count,
            "accepted_extraction_count": 0,
            "failure_stage": stage,
            "failure_reason": reason,
            "raw_llm_extraction_response": raw_llm[:2000] if raw_llm else "",
            "parser_output": parser_sentences[:10],
            "validation_results": {
                "passed": validated_count,
                "failed": len(rejected_items),
                "reasons": sanitize_fail_reasons,
            },
            "signal_strength": _signal_strength(d),
        })

    return results


# ---------------------------------------------------------------------------
# Summary and ranking
# ---------------------------------------------------------------------------

def build_failure_summary(failure_diagnostics: list[dict]) -> dict[str, int]:
    """Return count of each failure stage across all diagnosed chunks.

    Spec-required stages always appear first (keyed by their exact names).
    Discovery stages found by JH1b (ATTRIBUTION_FAILURE, CROSS_CHUNK,
    BUDGET_EXCLUSION) appear after, so spec consumers can slice [:8].
    All stages initialise to 0 so callers never get KeyError.
    """
    # Spec-required keys first, then discovery keys
    summary: dict[str, int] = {stage: 0 for stage in ALL_FAILURE_STAGES}
    for item in failure_diagnostics:
        stage = item.get("failure_stage", UNKNOWN)
        summary[stage] = summary.get(stage, 0) + 1
    return summary


def compute_top_missed_chunks(
    failure_diagnostics: list[dict],
    n: int = 10,
) -> list[dict]:
    """Return the top-N missed chunks sorted by signal_strength descending."""
    sorted_items = sorted(
        failure_diagnostics,
        key=lambda x: x.get("signal_strength", 0),
        reverse=True,
    )
    return [
        {
            "chunk_id":       item["chunk_id"],
            "document_name":  item["document_name"],
            "signal_strength": item.get("signal_strength", 0),
            "failure_stage":  item.get("failure_stage", UNKNOWN),
            "failure_reason": item.get("failure_reason", ""),
            "relevance_score": item.get("relevance_score", 0.0),
            "candidate_signals": item.get("candidate_signals", {}),
        }
        for item in sorted_items[:n]
    ]


# ---------------------------------------------------------------------------
# Per-document failure analysis
# ---------------------------------------------------------------------------

def analyze_document_failures(
    doc_name: str,
    failure_diagnostics: list[dict],
    chunk_diags: list | None = None,
    evidence: list | None = None,
) -> dict:
    """Summarise failure patterns for a single document.

    Returns a dict with chunk count, evidence_created, most common stage/reason.
    """
    doc_items = [d for d in failure_diagnostics if d.get("document_name") == doc_name]

    stage_counter: Counter[str] = Counter(d.get("failure_stage", UNKNOWN) for d in doc_items)
    reason_counter: Counter[str] = Counter(
        d.get("failure_reason", "")[:80] for d in doc_items
    )

    # Total chunks for this doc
    total_chunks = 0
    if chunk_diags is not None:
        total_chunks = sum(
            1 for c in chunk_diags
            if (c.document_name if hasattr(c, "document_name") else c.get("document_name")) == doc_name
        )

    # Evidence created for this doc
    evidence_created = 0
    if evidence is not None:
        evidence_created = sum(
            1 for e in evidence
            if (e.source_document if hasattr(e, "source_document") else e.get("source_document")) == doc_name
        )

    most_common_stage, _ = stage_counter.most_common(1)[0] if stage_counter else (UNKNOWN, 0)
    most_common_reason, _ = reason_counter.most_common(1)[0] if reason_counter else ("", 0)

    return {
        "document_name":              doc_name,
        "chunks":                     total_chunks,
        "failed_chunks":              len(doc_items),
        "evidence_created":           evidence_created,
        "stage_breakdown":            dict(stage_counter),
        "most_common_failure_stage":  most_common_stage,
        "most_common_failure_reason": most_common_reason,
    }
