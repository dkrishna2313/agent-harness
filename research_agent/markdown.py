"""Markdown rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .schemas import Contradiction, CoverageArea, EvidenceItem, ResearchGap, ResearchMemo


def memo_to_markdown(memo: ResearchMemo) -> str:
    """Render a structured research memo as Markdown."""

    sections = [
        f"# {memo.title}",
        "",
        f"**Question:** {memo.question}",
        "",
        "## Executive Summary",
        "",
        memo.executive_summary or "No executive summary generated.",
        "",
        _list_section("Confirmed Facts", memo.confirmed_facts),
        "",
        _list_section("Inferences", memo.inferences),
        "",
        _list_section("Power Implications", memo.power_implications),
        "",
        _list_section("Cooling Implications", memo.cooling_implications),
        "",
        _list_section("Networking Implications", memo.networking_implications),
        "",
        _list_section("Rack Architecture Implications", memo.rack_architecture_implications),
        "",
        _contradictions_section(memo.contradictions),
        "",
        _research_gaps_section(memo.research_gaps),
        "",
        _coverage_matrix_section(memo.coverage_matrix),
        "",
        _list_section("Open Questions", memo.open_questions),
        "",
        _source_notes_section(memo.source_notes or memo.evidence),
        "",
        _list_section("Evaluation Warnings", memo.evaluation_warnings),
        "",
    ]
    return "\n".join(sections).rstrip() + "\n"


def write_markdown(content: str, out_path: str | Path) -> Path:
    """Write Markdown content to disk, creating the parent directory."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _list_section(title: str, items: Sequence[str]) -> str:
    lines = [f"## {title}", ""]
    if not items:
        lines.append("- None.")
    else:
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _research_gaps_section(gaps: Sequence[ResearchGap]) -> str:
    lines = ["## Research Gaps", ""]
    if not gaps:
        lines.append("No research gaps identified.")
        return "\n".join(lines)

    for priority_label in ("high", "medium", "low"):
        tier = [g for g in gaps if g.priority == priority_label]
        if not tier:
            continue
        lines.append(f"**{priority_label.capitalize()} Priority**")
        lines.append("")
        for g in tier:
            lines.append(f"- **{g.gap_id}** {g.topic}: {g.description}")
            lines.append(f"  *{g.rationale}*")
        lines.append("")

    return "\n".join(lines).rstrip()


def _contradictions_section(contradictions: Sequence[Contradiction]) -> str:
    lines = ["## Potential Contradictions", ""]
    if not contradictions:
        lines.append("No significant contradictions detected.")
        return "\n".join(lines)
    for c in contradictions:
        lines.append(
            f"- **{c.contradiction_id}** [{c.severity.upper()}] **{c.topic}**: "
            f"{c.evidence_a_id} (*{c.evidence_a_source}*) vs "
            f"{c.evidence_b_id} (*{c.evidence_b_source}*) — {c.explanation}"
        )
    return "\n".join(lines)


def _coverage_matrix_section(areas: Sequence[CoverageArea]) -> str:
    lines = ["## Coverage Matrix", ""]
    if not areas:
        lines.append("No topic coverage data available.")
        return "\n".join(lines)

    # Group by coverage level for a scannable layout
    level_order = ("strong", "moderate", "weak", "none")
    level_emoji = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak", "none": "None"}

    for area in areas:
        lines.append(f"### {area.topic.title()}")
        lines.append("")
        lines.append(f"- Coverage: {level_emoji[area.coverage_level]}")
        lines.append(f"- Evidence Items: {area.evidence_count}")
        lines.append(f"- Sources: {area.source_count}")
        lines.append(f"- *{area.rationale}*")
        lines.append("")

    return "\n".join(lines).rstrip()


def _source_notes_section(items: Sequence[EvidenceItem]) -> str:
    lines = ["## Source Notes", ""]
    if not items:
        lines.append("- None.")
        return "\n".join(lines)

    grouped: dict[str, list[EvidenceItem]] = {}
    for item in items:
        grouped.setdefault(item.source_document, []).append(item)

    for document, notes in grouped.items():
        lines.append(f"### {document}")
        lines.append("")
        for note in notes:
            snippet = note.evidence_snippet.strip() or "No evidence snippet extracted."
            lines.extend(
                [
                    f"- **Evidence ID:** {note.evidence_id or 'unassigned'}",
                    f"  **Claim:** {note.claim}",
                    f"  **Evidence:** \"{snippet}\"",
                    f"  **Category:** {note.category}",
                    f"  **Relevance:** {note.relevance}",
                    f"  **Confidence:** {note.confidence}",
                ]
            )
        lines.append("")

    return "\n".join(lines).rstrip()
