"""ExecutiveSummaryWriter — PH6.4 Editorial Writer.

Consumes EditorialBrief.executive_summary (plus linked sections via provenance)
and populates EditorialManuscript.executive_summary with authored prose.

Design constraints:
- Improves communication, never reasoning.
- Does not invent facts, risks, or conclusions.
- Does not read ReportAgent, Markdown, or DOCX/PPTX output.
- All other manuscript sections remain untouched.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)


class ExecutiveSummaryWriter(EditorialWriter):
    """Writer for EditorialManuscript.executive_summary.

    Call write(brief, manuscript) → populated EditorialManuscript.
    The client must implement generate_executive_summary_prose().
    When client is None or mock, a deterministic fallback is used.
    """

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        brief: EditorialBrief,
        manuscript: EditorialManuscript,
    ) -> EditorialManuscript:
        """Populate manuscript.executive_summary from brief. Returns manuscript."""
        sec = brief.executive_summary
        opt = self._find_recommended_option(brief)
        tradeoffs = self._get_tradeoffs(brief)

        prose = self._generate(
            question=brief.metadata.question,
            recommended_option_title=sec.recommended_option_title or opt.get("title", ""),
            board_recommendation=sec.board_recommendation,
            decision_readiness=sec.decision_readiness,
            overall_confidence=sec.overall_confidence,
            option_description=opt.get("description", ""),
            strategic_objective=opt.get("strategic_objective", ""),
            key_tradeoffs=tradeoffs,
            key_conditions=list(sec.key_conditions),
            critical_unknowns=list(sec.critical_unknowns),
        )

        manuscript.executive_summary.paragraphs = prose.paragraphs
        manuscript.executive_summary.bullet_groups = prose.bullet_groups
        manuscript.executive_summary.subtitle = self._subtitle(sec)
        return manuscript

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_recommended_option(self, brief: EditorialBrief) -> dict:
        """Return the recommended option entry dict from brief.strategic_options."""
        target_id = brief.executive_summary.recommended_option_id
        for entry in brief.strategic_options.options:
            if entry.option_id == target_id:
                return {
                    "title": entry.title,
                    "description": entry.description,
                    "strategic_objective": entry.strategic_objective,
                }
        # Fallback: first option that is marked recommended
        for entry in brief.strategic_options.options:
            if entry.recommended:
                return {
                    "title": entry.title,
                    "description": entry.description,
                    "strategic_objective": entry.strategic_objective,
                }
        return {}

    def _get_tradeoffs(self, brief: EditorialBrief) -> list[str]:
        """Extract key_tradeoffs from brief.decision_analysis (linked via analysis_id)."""
        return list(brief.decision_analysis.key_tradeoffs)

    def _subtitle(self, sec) -> str:
        parts = [p for p in [sec.board_recommendation, sec.decision_readiness] if p]
        return " | ".join(parts)

    def _generate(self, **kwargs):
        """Call LLM or fall back to mock."""
        if self._client is None or getattr(self._client, "is_mock", False):
            LOGGER.debug("[ExecutiveSummaryWriter] using mock client")
            return self._mock_generate(**kwargs)

        if not hasattr(self._client, "generate_executive_summary_prose"):
            LOGGER.warning(
                "[ExecutiveSummaryWriter] client lacks generate_executive_summary_prose — using mock"
            )
            return self._mock_generate(**kwargs)

        try:
            return self._client.generate_executive_summary_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning(
                "[ExecutiveSummaryWriter] LLM call failed (%s: %s) — using mock",
                type(exc).__name__, exc,
            )
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(
        *,
        question: str,
        recommended_option_title: str,
        board_recommendation: str,
        decision_readiness: str,
        overall_confidence: str,
        option_description: str,
        strategic_objective: str,
        key_tradeoffs: list[str],
        key_conditions: list[str],
        critical_unknowns: list[str],
    ):
        """Deterministic fallback when no live client is available."""
        from research_agent.claude_client import ExecutiveSummaryProsePayload

        tradeoffs_text = (
            "; ".join(key_tradeoffs[:3]) if key_tradeoffs else "no comparative tradeoffs specified"
        )
        desc_excerpt = option_description[:200].rstrip(".") if option_description else recommended_option_title

        cond_bullets = [c[:120] for c in key_conditions[:6]] or [
            "Validate core market assumptions before commitment."
        ]
        unknown_bullets = [u[:120] for u in critical_unknowns[:6]] or [
            "Independent evidence verification remains outstanding."
        ]

        return ExecutiveSummaryProsePayload(
            paragraphs=[
                (
                    f"{recommended_option_title} is the recommended strategic direction. "
                    f"The board-level assessment is: {board_recommendation}."
                ),
                (
                    f"This direction advances the stated objective of {strategic_objective}. "
                    f"{desc_excerpt}. "
                    f"Against alternatives, the decisive tradeoffs are: {tradeoffs_text}."
                ),
                (
                    f"The recommendation holds under {len(key_conditions)} principal condition(s). "
                    "Each condition defines a boundary within which the preferred direction "
                    "maintains its risk-adjusted advantage over competing options."
                ),
                (
                    f"Decision readiness is assessed as {decision_readiness}, with {overall_confidence} overall confidence. "
                    f"{len(critical_unknowns)} critical unknown(s) remain unresolved and must be addressed "
                    "through the validation priorities before a final commitment is made."
                ),
            ],
            bullet_groups=[cond_bullets, unknown_bullets],
        )
