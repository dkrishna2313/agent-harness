"""RecommendationWriter — PH6.6 Editorial Writer.

Consumes EditorialBrief.recommendations and populates
EditorialManuscript.recommendations with authored prose and a priority table.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)

_HORIZON_ORDER = {"near_term": 0, "near-term": 0, "immediate": 0, "short_term": 0,
                  "short-term": 0, "medium_term": 1, "medium-term": 1,
                  "long_term": 2, "long-term": 2}
_PRIORITY_ORDER = {"high": 0, "critical": 0, "medium": 1, "low": 2}


class RecommendationWriter(EditorialWriter):
    """Writer for EditorialManuscript.recommendations."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def write(self, brief: EditorialBrief, manuscript: EditorialManuscript) -> EditorialManuscript:
        recs = brief.recommendations.recommendations
        recs_dicts = [
            {
                "title": r.title,
                "summary": r.summary,
                "priority": r.priority,
                "time_horizon": r.time_horizon,
                "recommendation_id": r.recommendation_id,
            }
            for r in recs
        ]

        prose = self._generate(
            question=brief.metadata.question,
            recommendations=recs_dicts,
        )

        manuscript.recommendations.paragraphs = prose.paragraphs
        manuscript.recommendations.bullet_groups = prose.bullet_groups
        manuscript.recommendations.subtitle = self._subtitle(recs_dicts)
        manuscript.recommendations.tables = self._build_table(recs_dicts)
        return manuscript

    def _subtitle(self, recs: list[dict]) -> str:
        n = len(recs)
        high = sum(1 for r in recs if r.get("priority", "").lower() in ("high", "critical"))
        return f"{n} recommendation{'s' if n != 1 else ''}, {high} high-priority" if n else ""

    def _build_table(self, recs: list[dict]) -> list[dict[str, Any]]:
        if not recs:
            return []
        headers = ["Recommendation", "Priority", "Time Horizon", "Summary"]
        rows = [
            [r.get("title", ""), r.get("priority", "").capitalize(),
             r.get("time_horizon", "").replace("_", " ").title(),
             r.get("summary", "")[:80]]
            for r in recs
        ]
        return [{"title": "Recommended Actions", "headers": headers, "rows": rows, "notes": ""}]

    def _generate(self, **kwargs):
        if self._client is None or getattr(self._client, "is_mock", False):
            return self._mock_generate(**kwargs)
        if not hasattr(self._client, "generate_recommendation_prose"):
            LOGGER.warning("[RecommendationWriter] client lacks generate_recommendation_prose — using mock")
            return self._mock_generate(**kwargs)
        try:
            return self._client.generate_recommendation_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning("[RecommendationWriter] LLM call failed (%s: %s) — using mock", type(exc).__name__, exc)
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(*, question: str, recommendations: list[dict]):
        from research_agent.claude_client import RecommendationProsePayload
        high = [r for r in recommendations if r.get("priority", "").lower() in ("high", "critical")]
        other = [r for r in recommendations if r not in high]
        near = [r for r in recommendations if any(w in r.get("time_horizon", "").lower() for w in ("near", "immediate", "short"))]
        high_bullets = [f"{r.get('title', '')}: {r.get('summary', '')[:80]}" for r in high[:5]] or ["Initiate recommended course of action."]
        other_bullets = [f"{r.get('title', '')}: {r.get('summary', '')[:80]}" for r in other[:5]] or ["Supporting actions to follow."]
        near_titles = ", ".join(r.get("title", "") for r in near[:3]) if near else "near-term actions"
        return RecommendationProsePayload(
            paragraphs=[
                (
                    f"{len(recommendations)} recommendation(s) address: {question}. "
                    f"Implementation centres on {len(high)} high-priority action(s) and {len(other)} supporting measure(s)."
                ),
                (
                    f"Near-term priorities focus on {near_titles}. "
                    "These actions establish the conditions for longer-horizon success and should begin immediately."
                ),
                (
                    "Supporting recommendations provide the operational and strategic foundation for sustained progress. "
                    "Sequencing these actions correctly reduces execution risk and preserves optionality."
                ),
                (
                    "Together, these recommendations represent a coherent implementation plan. "
                    "Outcomes are conditional on the critical assumptions identified in the decision analysis."
                ),
            ],
            bullet_groups=[high_bullets, other_bullets],
        )
