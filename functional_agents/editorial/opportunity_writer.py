"""OpportunityWriter — PH6.8 Editorial Writer.

Consumes EditorialBrief.strategic_opportunities and populates
EditorialManuscript.strategic_opportunities with authored prose and an impact table.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)

_IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}
_LIKE_ORDER = {"high": 0, "medium": 1, "low": 2}


class OpportunityWriter(EditorialWriter):
    """Writer for EditorialManuscript.strategic_opportunities."""

    section_name = "strategic_opportunities"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def write(self, brief: EditorialBrief, manuscript: EditorialManuscript) -> EditorialManuscript:
        opps = brief.strategic_opportunities.opportunities
        opps_dicts = [
            {
                "opportunity_id": o.opportunity_id,
                "statement": o.statement,
                "category": o.category,
                "likelihood": o.likelihood,
                "impact": o.impact,
            }
            for o in sorted(
                opps,
                key=lambda o: (_IMPACT_ORDER.get(o.impact.lower(), 99), _LIKE_ORDER.get(o.likelihood.lower(), 99))
            )
        ]

        prose = self._generate(
            question=brief.metadata.question,
            opportunities=opps_dicts,
        )

        manuscript.strategic_opportunities.paragraphs = prose.paragraphs
        manuscript.strategic_opportunities.bullet_groups = prose.bullet_groups
        manuscript.strategic_opportunities.subtitle = self._subtitle(opps_dicts)
        manuscript.strategic_opportunities.tables = self._build_table(opps_dicts)
        return manuscript

    def _subtitle(self, opps: list[dict]) -> str:
        n = len(opps)
        high = sum(1 for o in opps if o.get("impact", "").lower() == "high")
        cats = len({o.get("category", "") for o in opps if o.get("category")})
        return f"{n} opportunit{'ies' if n != 1 else 'y'} across {cats} categor{'ies' if cats != 1 else 'y'}" if n else ""

    def _build_table(self, opps: list[dict]) -> list[dict[str, Any]]:
        if not opps:
            return []
        headers = ["Opportunity", "Category", "Likelihood", "Impact"]
        rows = [
            [o.get("statement", "")[:80], o.get("category", ""),
             o.get("likelihood", ""), o.get("impact", "")]
            for o in opps
        ]
        return [{"title": "Strategic Opportunities", "headers": headers, "rows": rows, "notes": ""}]

    def _generate(self, **kwargs):
        if self._client is None or getattr(self._client, "is_mock", False):
            return self._mock_generate(**kwargs)
        if not hasattr(self._client, "generate_opportunity_prose"):
            LOGGER.warning("[OpportunityWriter] client lacks generate_opportunity_prose — using mock")
            return self._mock_generate(**kwargs)
        try:
            return self._client.generate_opportunity_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning("[OpportunityWriter] LLM call failed (%s: %s) — using mock", type(exc).__name__, exc)
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(*, question: str, opportunities: list[dict]):
        from research_agent.claude_client import OpportunityProsePayload
        high_impact = [o for o in opportunities if o.get("impact", "").lower() == "high"]
        categories = list({o.get("category", "") for o in opportunities if o.get("category")})
        top = high_impact[0] if high_impact else (opportunities[0] if opportunities else {})
        opp_bullets = [
            f"{o.get('statement', '')[:90]} (Likelihood: {o.get('likelihood')}, Impact: {o.get('impact')})"
            for o in high_impact[:5]
        ] or ["Strategic opportunity identified."]
        enabling = [f"Category: {c}" for c in categories[:5]] or ["Enabling conditions to be established."]
        return OpportunityProsePayload(
            paragraphs=[
                (
                    f"{len(opportunities)} strategic opportunit{'ies' if len(opportunities) != 1 else 'y'} identified "
                    f"across {len(categories)} categor{'ies' if len(categories) != 1 else 'y'} for: {question}."
                ),
                (
                    f"The highest-impact opportunity is: {top.get('statement', '')[:140]}. "
                    f"This carries {top.get('likelihood', 'uncertain').lower()} likelihood "
                    f"and {top.get('impact', 'uncertain').lower()} expected impact."
                ),
                (
                    "Realising these opportunities requires establishing the conditions identified in the recommendation register. "
                    f"Timing is material — {len(high_impact)} opportunit{'ies' if len(high_impact) != 1 else 'y'} "
                    "are time-sensitive given market dynamics."
                ),
                (
                    "Captured together, these opportunities represent meaningful strategic upside. "
                    "The recommended course of action is designed to position the organisation to capture this value."
                ),
            ],
            bullet_groups=[opp_bullets, enabling],
        )
