"""Trace JSON generation for research harness runs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .evaluator import classify_question_topics
from .schemas import ResearchMemo, SourceDocument

if TYPE_CHECKING:
    from .profile import DomainProfile

MEMO_SECTIONS = [
    "Executive Summary",
    "Confirmed Facts",
    "Inferences",
    "Power Implications",
    "Cooling Implications",
    "Networking Implications",
    "Rack Architecture Implications",
    "Potential Contradictions",
    "Research Gaps",
    "Coverage Matrix",
    "Open Questions",
    "Source Notes",
    "Evaluation Warnings",
]


def trace_path_for_output(output_path: str | Path) -> Path:
    """Return the trace path adjacent to a Markdown output path."""

    return Path(output_path).with_suffix(".trace.json")


def build_trace(
    *,
    question: str,
    source_directory: str | Path,
    output_path: str | Path,
    documents: list[SourceDocument],
    memo: ResearchMemo,
    mock_mode: bool,
    profile: "DomainProfile | None" = None,
) -> dict[str, Any]:
    """Build a serializable trace payload for one harness run."""

    evidence_items = memo.source_notes or memo.evidence
    evidence_counts = Counter(item.source_document for item in evidence_items)
    ranked_evidence = sorted(
        evidence_items,
        key=lambda item: (
            -item.overall_score,
            -item.relevance_score,
            -item.specificity_score,
            item.source_document.lower(),
            item.evidence_id or item.claim,
        ),
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "question_topics_detected": sorted(classify_question_topics(question, profile)),
        "source_directory": str(source_directory),
        "output_path": str(output_path),
        "documents_loaded": len(documents),
        "total_characters_extracted": sum(document.char_count for document in documents),
        "documents": [
            {
                "filename": document.path.name,
                "path": str(document.path),
                "character_count": document.char_count,
                "evidence_item_count": evidence_counts.get(document.path.name, 0),
            }
            for document in documents
        ],
        "evidence_items": [item.model_dump() for item in evidence_items],
        "top_evidence_limit": memo.metadata.get("top_evidence_limit"),
        "synthesis_input_tokens": _synthesis_input_tokens(memo),
        "evidence_passed_to_synthesis": memo.metadata.get("evidence_passed_to_synthesis", 0),
        "contradictions_passed_to_synthesis": memo.metadata.get("contradictions_passed_to_synthesis", 0),
        "research_gaps_passed_to_synthesis": memo.metadata.get("research_gaps_passed_to_synthesis", 0),
        "evidence_items_used_for_synthesis": memo.metadata.get(
            "evidence_items_used_for_synthesis",
            len(evidence_items),
        ),
        "evidence_ranking": [
            {
                "rank": index,
                "evidence_id": item.evidence_id,
                "source_document": item.source_document,
                "overall_score": item.overall_score,
                "relevance_score": item.relevance_score,
                "source_quality_score": item.source_quality_score,
                "specificity_score": item.specificity_score,
            }
            for index, item in enumerate(ranked_evidence, start=1)
        ],
        "memo_sections": list(MEMO_SECTIONS),
        "evaluation_warnings": list(memo.evaluation_warnings),
        "chunk_count": memo.metadata.get("chunk_count", 0),
        "chunks_selected": memo.metadata.get("chunks_selected", 0),
        "chunks_per_document": memo.metadata.get("chunks_per_document", {}),
        "evidence_per_chunk": memo.metadata.get("evidence_per_chunk", {}),
        "avg_chunk_size": memo.metadata.get("avg_chunk_size", 0),
        "chunk_diagnostics": memo.metadata.get("chunk_diagnostics", []),
        "retrieval_ranking": memo.metadata.get("retrieval_ranking", []),
        "selected_chunk_ids": memo.metadata.get("selected_chunk_ids", []),
        "rejected_chunk_ids": memo.metadata.get("rejected_chunk_ids", []),
        "contradictions_detected": memo.metadata.get("contradictions", []),
        "research_gaps": memo.metadata.get("research_gaps", []),
        "domain_profile": memo.metadata.get("domain_profile", {}),
        "coverage_matrix": memo.metadata.get("coverage_matrix", []),
        "source_quality_map": {
            name: sq["source_quality_score"]
            for name, sq in memo.metadata.get("source_quality_map", {}).items()
        },
        "source_quality_details": memo.metadata.get("source_quality_map", {}),
        "suppressed_comparisons": memo.metadata.get("suppressed_comparisons", []),
        "extraction_stats": memo.metadata.get("extraction_stats", {}),
        "web_search": memo.metadata.get("web_search"),
        "mock_mode": mock_mode,
        "model_name": memo.claude_model_name,
        "claude_request_timestamp": memo.claude_request_timestamp,
        "claude_response_success": memo.claude_response_success,
        "claude_token_usage": memo.claude_token_usage,
        "claude_calls": [trace.model_dump() for trace in memo.claude_call_traces],
    }


def _synthesis_input_tokens(memo: ResearchMemo) -> int:
    """Return actual synthesis input tokens from call traces, or the estimate."""
    for call in memo.claude_call_traces:
        if call.operation == "synthesize_memo":
            return call.token_usage.get("input_tokens", 0)
    return memo.metadata.get("synthesis_input_tokens_estimate", 0)


def write_trace(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write a trace payload next to the Markdown output path."""

    trace_path = trace_path_for_output(output_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return trace_path
