"""ConfidenceWriter — PH6.9 Editorial Writer.

Consumes EditorialBrief.executive_confidence + validation_priorities and populates
EditorialManuscript.executive_confidence with decision readiness prose.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)


class ConfidenceWriter(EditorialWriter):
    """Writer for EditorialManuscript.executive_confidence."""

    section_name = "executive_confidence"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def write(self, brief: EditorialBrief, manuscript: EditorialManuscript) -> EditorialManuscript:
        conf = brief.executive_confidence
        vp = brief.validation_priorities

        prose = self._generate(
            question=brief.metadata.question,
            overall_confidence=conf.overall_confidence,
            decision_readiness=conf.decision_readiness,
            board_recommendation=conf.board_recommendation,
            confidence_drivers=conf.confidence_drivers,
            confidence_limiters=conf.confidence_limiters,
            critical_unknowns=conf.critical_unknowns,
            validation_priorities=vp.priorities,
            confidence_if_assumptions_hold=conf.confidence_if_assumptions_hold,
            confidence_if_assumptions_fail=conf.confidence_if_assumptions_fail,
        )

        manuscript.executive_confidence.paragraphs = prose.paragraphs
        manuscript.executive_confidence.bullet_groups = prose.bullet_groups
        manuscript.executive_confidence.subtitle = self._subtitle(conf)
        return manuscript

    def _subtitle(self, conf) -> str:
        parts = []
        if conf.board_recommendation:
            parts.append(conf.board_recommendation)
        if conf.decision_readiness:
            parts.append(conf.decision_readiness)
        return " | ".join(parts) if parts else conf.overall_confidence or ""

    def _generate(self, **kwargs):
        if self._client is None or getattr(self._client, "is_mock", False):
            return self._mock_generate(**kwargs)
        if not hasattr(self._client, "generate_confidence_prose"):
            LOGGER.warning("[ConfidenceWriter] client lacks generate_confidence_prose — using mock")
            return self._mock_generate(**kwargs)
        try:
            return self._client.generate_confidence_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning("[ConfidenceWriter] LLM call failed (%s: %s) — using mock", type(exc).__name__, exc)
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(
        *,
        question: str,
        overall_confidence: str,
        decision_readiness: str,
        board_recommendation: str,
        confidence_drivers: list[str],
        confidence_limiters: list[str],
        critical_unknowns: list[str],
        validation_priorities: list[str],
        confidence_if_assumptions_hold: str,
        confidence_if_assumptions_fail: str,
    ):
        from research_agent.claude_client import ConfidenceProsePayload

        priority_bullets = [p[:100] for p in validation_priorities[:6]] or [
            "Complete outstanding due diligence items before commitment."
        ]
        unknown_bullets = [u[:100] for u in critical_unknowns[:5]] or [
            "Critical unknowns remain unresolved."
        ]
        hold_clause = (
            f" If all key assumptions hold, confidence is expected to rise to {confidence_if_assumptions_hold}."
            if confidence_if_assumptions_hold else ""
        )
        fail_clause = (
            f" Confidence would fall to {confidence_if_assumptions_fail} if critical assumptions fail."
            if confidence_if_assumptions_fail else ""
        )
        return ConfidenceProsePayload(
            paragraphs=[
                (
                    f"Decision readiness is assessed as {decision_readiness or 'indeterminate'}. "
                    f"The board recommendation is: {board_recommendation or 'pending'}. "
                    f"Overall analytical confidence is {overall_confidence or 'not rated'}."
                ),
                (
                    f"{len(confidence_drivers)} factor(s) support this confidence assessment for: {question}. "
                    "These drivers reflect the quality and consistency of the underlying evidence base "
                    "and the robustness of the structured decision analysis."
                ),
                (
                    f"{len(confidence_limiters)} factor(s) constrain confidence below its potential ceiling.{hold_clause}{fail_clause} "
                    "These limitations define the boundaries within which the recommendation holds."
                ),
                (
                    f"{len(validation_priorities)} validation priority(ies) have been identified. "
                    "Completing these priorities is the primary mechanism for improving decision readiness "
                    f"and resolving the {len(critical_unknowns)} critical unknown(s) that remain outstanding."
                ),
            ],
            bullet_groups=[priority_bullets, unknown_bullets],
        )
