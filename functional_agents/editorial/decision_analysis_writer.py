"""DecisionAnalysisWriter — PH6.5 Editorial Writer.

Consumes EditorialBrief.decision_analysis (plus strategic_options via provenance
links) and populates EditorialManuscript.decision_analysis with authored prose
and a structured decision matrix table.

Design constraints:
- Improves communication, never reasoning.
- Does not change rankings, tradeoffs, or uncertainties.
- Does not read ReportAgent, Markdown, or DOCX/PPTX output.
- Only decision_analysis section of the manuscript is modified.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)


class DecisionAnalysisWriter(EditorialWriter):
    """Writer for EditorialManuscript.decision_analysis.

    Call write(brief, manuscript) → populated EditorialManuscript.
    When client is None or mock, a deterministic fallback is used.
    """

    section_name = "decision_analysis"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API (EditorialWriter interface)
    # ------------------------------------------------------------------

    def write(
        self,
        brief: EditorialBrief,
        manuscript: EditorialManuscript,
    ) -> EditorialManuscript:
        """Populate manuscript.decision_analysis from brief. Returns manuscript."""
        sec = brief.decision_analysis
        ranked_options = self._ranked_options(brief)
        recommended_title = self._recommended_title(brief, sec.recommended_option_id)

        prose = self._generate(
            question=brief.metadata.question,
            recommended_option_title=recommended_title,
            ranked_option_titles=[o["title"] for o in ranked_options],
            comparison_dimensions=list(sec.comparison_dimensions),
            key_tradeoffs=list(sec.key_tradeoffs),
            key_uncertainties=list(sec.key_uncertainties),
        )

        manuscript.decision_analysis.paragraphs = prose.paragraphs
        manuscript.decision_analysis.bullet_groups = prose.bullet_groups
        manuscript.decision_analysis.subtitle = self._subtitle(sec, ranked_options)
        manuscript.decision_analysis.tables = self._build_tables(brief, sec, ranked_options)
        return manuscript

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recommended_title(self, brief: EditorialBrief, recommended_id: str) -> str:
        for opt in brief.strategic_options.options:
            if opt.option_id == recommended_id:
                return opt.title
        # Fallback: first recommended
        for opt in brief.strategic_options.options:
            if opt.recommended:
                return opt.title
        return recommended_id or "Recommended Option"

    def _ranked_options(self, brief: EditorialBrief) -> list[dict]:
        """Return option dicts in ranking order from brief.decision_analysis.option_rankings."""
        ranking = list(brief.decision_analysis.option_rankings)
        by_id = {opt.option_id: opt for opt in brief.strategic_options.options}
        ordered = []
        for oid in ranking:
            opt = by_id.get(oid)
            if opt:
                ordered.append({
                    "option_id": opt.option_id,
                    "title": opt.title,
                    "description": opt.description,
                    "recommended": opt.recommended,
                })
        # Append any options not mentioned in rankings
        ranked_ids = set(ranking)
        for opt in brief.strategic_options.options:
            if opt.option_id not in ranked_ids:
                ordered.append({
                    "option_id": opt.option_id,
                    "title": opt.title,
                    "description": opt.description,
                    "recommended": opt.recommended,
                })
        return ordered

    def _subtitle(self, sec, ranked_options: list[dict]) -> str:
        n = len(ranked_options)
        return f"{n} option{'s' if n != 1 else ''} evaluated" if n else ""

    def _build_tables(self, brief: EditorialBrief, sec, ranked_options: list[dict]) -> list[dict[str, Any]]:
        """Build structured decision matrix table. No markdown — structured dict only."""
        matrix = list(sec.decision_matrix) if sec.decision_matrix else []
        dims = list(sec.comparison_dimensions)

        if not matrix and not dims:
            return []

        if matrix:
            return [self._matrix_from_decision_matrix(matrix, dims)]

        # Fallback: plain option listing when no matrix data
        if ranked_options and dims:
            return [self._matrix_placeholder(ranked_options, dims)]

        return []

    @staticmethod
    def _resolve_dim(entry: "dict | object", label: str) -> str:
        """Look up a dimension value from a flat or nested matrix entry.

        Tries in order:
        1. nested ``entry["dimensions"][label]`` (original nested schema)
        2. flat ``entry[snake_case_label]`` (current flat schema from decision analysis)
        3. Pydantic attribute access with the same snake_case fallback
        """
        snake = label.lower().replace(" ", "_")
        if isinstance(entry, dict):
            nested = (entry.get("dimensions") or {}).get(label)
            if nested is not None:
                return str(nested)
            flat = entry.get(snake)
            if flat is not None:
                return str(flat)
        else:
            nested = (getattr(entry, "dimensions", None) or {}).get(label)
            if nested is not None:
                return str(nested)
            flat = getattr(entry, snake, None)
            if flat is not None:
                return str(flat)
        return "—"

    def _matrix_from_decision_matrix(self, matrix: list, dims: list[str]) -> dict[str, Any]:
        """Convert decision_matrix list → structured table dict.

        Rows are joined by option_id (not list position) so that reordering
        the source list does not change which values appear in which row.
        """
        headers = ["Option"] + dims + ["Overall"]

        # Index by option_id for id-based lookup, preserving list order for display.
        matrix_by_id: dict[str, object] = {}
        ordered_entries: list[object] = []
        for entry in matrix:
            oid = entry.get("option_id", "") if isinstance(entry, dict) else getattr(entry, "option_id", "")
            if oid and oid not in matrix_by_id:
                matrix_by_id[oid] = entry
            ordered_entries.append(entry)

        rows = []
        for entry in ordered_entries:
            if isinstance(entry, dict):
                title = entry.get("option_title") or entry.get("title") or entry.get("option_id", "")
                overall = str(entry.get("overall_score", "—"))
            else:
                title = getattr(entry, "option_title", "") or getattr(entry, "title", "")
                overall = str(getattr(entry, "overall_score", "—"))
            row = [title] + [self._resolve_dim(entry, d) for d in dims] + [overall]
            rows.append(row)

        return {
            "title": "Strategic Option Comparison",
            "headers": headers,
            "rows": rows,
            "notes": "",
        }

    def _matrix_placeholder(self, ranked_options: list[dict], dims: list[str]) -> dict[str, Any]:
        headers = ["Option", "Rank"] + dims[:4]
        rows = []
        for i, opt in enumerate(ranked_options):
            row = [opt["title"], str(i + 1)] + ["—"] * min(len(dims), 4)
            rows.append(row)
        return {
            "title": "Strategic Option Comparison",
            "headers": headers,
            "rows": rows,
            "notes": "Detailed dimension scores pending.",
        }

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    def _generate(self, **kwargs):
        if self._client is None or getattr(self._client, "is_mock", False):
            LOGGER.debug("[DecisionAnalysisWriter] using mock client")
            return self._mock_generate(**kwargs)

        if not hasattr(self._client, "generate_decision_analysis_prose"):
            LOGGER.warning(
                "[DecisionAnalysisWriter] client lacks generate_decision_analysis_prose — using mock"
            )
            return self._mock_generate(**kwargs)

        try:
            return self._client.generate_decision_analysis_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning(
                "[DecisionAnalysisWriter] LLM call failed (%s: %s) — using mock",
                type(exc).__name__, exc,
            )
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(
        *,
        question: str,
        recommended_option_title: str,
        ranked_option_titles: list[str],
        comparison_dimensions: list[str],
        key_tradeoffs: list[str],
        key_uncertainties: list[str],
    ):
        from research_agent.claude_client import DecisionAnalysisProsePayload

        tradeoff_bullets = [t[:120] for t in key_tradeoffs[:6]] or [
            "No comparative tradeoffs recorded."
        ]
        uncertainty_bullets = [u[:120] for u in key_uncertainties[:6]] or [
            "Key uncertainties unspecified."
        ]
        alt_titles = [t for t in ranked_option_titles if t != recommended_option_title]
        alt_text = ", ".join(alt_titles[:2]) if alt_titles else "the alternatives considered"
        dims_text = ", ".join(comparison_dimensions[:4]) if comparison_dimensions else "the evaluation dimensions"

        return DecisionAnalysisProsePayload(
            paragraphs=[
                (
                    f"{recommended_option_title} ranks first among the {len(ranked_option_titles)} "
                    f"option(s) evaluated for: {question}."
                ),
                (
                    f"Compared to {alt_text}, this option achieves a stronger risk-adjusted position "
                    f"across {dims_text}. "
                    "The comparison reflects the full set of strategic, financial, and execution dimensions."
                ),
                (
                    f"The analysis identifies {len(key_tradeoffs)} decisive tradeoff(s). "
                    "These tradeoffs define where the recommended option accepts short-term costs "
                    "to secure long-term strategic advantage."
                ),
                (
                    f"{len(key_uncertainties)} key uncertainty(ies) affect the robustness of this ranking. "
                    "Should the primary uncertainties resolve unfavourably, the sensitivity of the "
                    "recommendation should be reassessed before commitment."
                ),
                (
                    "Confidence in this ranking is based on structured comparison across the evaluation dimensions "
                    "and is conditional on the assumptions underpinning the recommended path."
                ),
            ],
            bullet_groups=[tradeoff_bullets, uncertainty_bullets],
        )
