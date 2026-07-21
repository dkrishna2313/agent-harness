"""AppendixWriter — PH6.10 Editorial Writer.

Consumes EditorialBrief.appendix and populates EditorialManuscript.appendix
with a factual supporting evidence summary, topic table, and selected citations.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)


class AppendixWriter(EditorialWriter):
    """Writer for EditorialManuscript.appendix."""

    section_name = "appendix"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def write(self, brief: EditorialBrief, manuscript: EditorialManuscript) -> EditorialManuscript:
        app = brief.appendix

        prose = self._generate(
            question=brief.metadata.question,
            research_object_id=app.research_object_id,
            total_evidence_items=app.total_evidence_items,
            citation_count=app.citation_count,
            profiles=app.profiles,
            evidence_topics=app.evidence_topics,
            citations=app.citations,
        )

        manuscript.appendix.paragraphs = prose.paragraphs
        manuscript.appendix.bullet_groups = prose.bullet_groups
        manuscript.appendix.subtitle = self._subtitle(app)
        manuscript.appendix.tables = self._build_tables(app)
        return manuscript

    def _subtitle(self, app) -> str:
        n = app.total_evidence_items
        n_topics = len(app.evidence_topics)
        if n and n_topics:
            return f"{n} evidence item{'s' if n != 1 else ''} across {n_topics} topic{'s' if n_topics != 1 else ''}"
        if n:
            return f"{n} evidence item{'s' if n != 1 else ''}"
        return ""

    def _build_tables(self, app) -> list[dict[str, Any]]:
        tables = []
        if app.evidence_topics:
            sorted_topics = sorted(app.evidence_topics.items(), key=lambda x: x[1], reverse=True)
            tables.append({
                "title": "Evidence by Topic",
                "headers": ["Topic", "Items"],
                "rows": [[t, str(n)] for t, n in sorted_topics],
                "notes": "",
            })
        if app.citations:
            tables.append({
                "title": "Selected Citations",
                "headers": ["#", "Citation"],
                "rows": [[str(i + 1), c[:150]] for i, c in enumerate(app.citations[:20])],
                "notes": f"Showing {min(20, len(app.citations))} of {len(app.citations)} citation(s).",
            })
        return tables

    def _generate(self, **kwargs):
        if self._client is None or getattr(self._client, "is_mock", False):
            return self._mock_generate(**kwargs)
        if not hasattr(self._client, "generate_appendix_prose"):
            LOGGER.warning("[AppendixWriter] client lacks generate_appendix_prose — using mock")
            return self._mock_generate(**kwargs)
        try:
            return self._client.generate_appendix_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning("[AppendixWriter] LLM call failed (%s: %s) — using mock", type(exc).__name__, exc)
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(
        *,
        question: str,
        research_object_id: str,
        total_evidence_items: int,
        citation_count: int,
        profiles: list[str],
        evidence_topics: dict,
        citations: list[str],
    ):
        from research_agent.claude_client import AppendixProsePayload

        n_topics = len(evidence_topics)
        profile_str = ", ".join(profiles) if profiles else "default"
        top_topics = sorted(evidence_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_summary = (
            ", ".join(f"{t} ({n} items)" for t, n in top_topics)
            if top_topics else "general research"
        )
        cite_bullets = [c[:120] for c in citations[:10]]

        paragraphs = [
            (
                f"This analysis draws on {total_evidence_items} evidence item(s) "
                f"with {citation_count} citation(s), sourced via the {profile_str} knowledge profile(s)."
            ),
            (
                f"Evidence spans {n_topics} topic area(s). "
                f"Leading topics include: {topic_summary}. "
                "Areas with limited coverage are flagged in the confidence assessment."
            ),
        ]
        if citations:
            paragraphs.append(
                f"{len(citations)} primary reference(s) are listed below. "
                "These citations were extracted and validated during the evidence pipeline."
            )

        return AppendixProsePayload(
            paragraphs=paragraphs,
            bullet_groups=[cite_bullets] if cite_bullets else [],
        )
