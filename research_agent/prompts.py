"""Prompt builders for the research agent."""

from __future__ import annotations

from collections.abc import Sequence

from .schemas import SourceDocument

SYSTEM_PROMPT = """You are a careful infrastructure research analyst.
Use only the supplied local sources. Distinguish confirmed facts from
reasoned inferences and speculation. Surface uncertainty as warnings."""

REQUIRED_MEMO_SECTIONS = [
    "Executive Summary",
    "Confirmed Facts",
    "Inferences",
    "Power Implications",
    "Cooling Implications",
    "Open Questions",
    "Source Notes",
    "Evaluation Warnings",
]


def build_research_prompt(
    question: str,
    documents: Sequence[SourceDocument],
    *,
    max_context_chars: int = 60_000,
) -> str:
    """Build the single prompt sent to the LLM client."""

    remaining = max_context_chars
    source_blocks: list[str] = []

    for document in documents:
        if remaining <= 0:
            break
        snippet = document.text[:remaining]
        remaining -= len(snippet)
        source_blocks.append(
            "\n".join(
                [
                    f"Source: {document.title}",
                    f"Path: {document.path}",
                    "Text:",
                    snippet,
                ]
            )
        )

    sections = "\n".join(f"- {section}" for section in REQUIRED_MEMO_SECTIONS)
    sources = "\n\n---\n\n".join(source_blocks) if source_blocks else "No source text was loaded."

    return f"""Question:
{question}

Required memo sections:
{sections}

Return Markdown only. Use the section headings exactly as listed.

Local source text:
{sources}
"""
