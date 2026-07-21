"""RiskWriter — PH6.7 Editorial Writer.

Consumes EditorialBrief.strategic_risks and populates
EditorialManuscript.strategic_risks with authored prose and a severity table.
"""

from __future__ import annotations

import logging
from typing import Any

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript
from .editorial_writer import EditorialWriter

LOGGER = logging.getLogger(__name__)

_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


class RiskWriter(EditorialWriter):
    """Writer for EditorialManuscript.strategic_risks."""

    section_name = "strategic_risks"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def write(self, brief: EditorialBrief, manuscript: EditorialManuscript) -> EditorialManuscript:
        risk_sec = brief.strategic_risks
        risks = risk_sec.risks
        top_risk = next(
            (r for r in risks if r.risk_id == risk_sec.top_risk_id),
            risks[0] if risks else None,
        )
        top_stmt = top_risk.statement if top_risk else "No principal risk identified."

        risks_dicts = [
            {
                "risk_id": r.risk_id,
                "statement": r.statement,
                "severity": r.severity,
                "likelihood": r.likelihood,
                "mitigation_notes": r.mitigation_notes,
            }
            for r in sorted(risks, key=lambda r: _SEV_ORDER.get(r.severity.lower(), 99))
        ]

        prose = self._generate(
            question=brief.metadata.question,
            risks=risks_dicts,
            top_risk_statement=top_stmt,
        )

        manuscript.strategic_risks.paragraphs = prose.paragraphs
        manuscript.strategic_risks.bullet_groups = prose.bullet_groups
        manuscript.strategic_risks.subtitle = self._subtitle(risks_dicts)
        manuscript.strategic_risks.tables = self._build_table(risks_dicts)
        return manuscript

    def _subtitle(self, risks: list[dict]) -> str:
        n = len(risks)
        high = sum(1 for r in risks if r.get("severity", "").lower() == "high")
        return f"{n} risk{'s' if n != 1 else ''} identified, {high} High-severity" if n else ""

    def _build_table(self, risks: list[dict]) -> list[dict[str, Any]]:
        if not risks:
            return []
        headers = ["Risk", "Severity", "Likelihood", "Mitigation"]
        rows = [
            [r.get("statement", "")[:80], r.get("severity", ""), r.get("likelihood", ""),
             r.get("mitigation_notes", "")[:60] or "—"]
            for r in risks
        ]
        return [{"title": "Risk Register", "headers": headers, "rows": rows, "notes": ""}]

    def _generate(self, **kwargs):
        if self._client is None or getattr(self._client, "is_mock", False):
            return self._mock_generate(**kwargs)
        if not hasattr(self._client, "generate_risk_prose"):
            LOGGER.warning("[RiskWriter] client lacks generate_risk_prose — using mock")
            return self._mock_generate(**kwargs)
        try:
            return self._client.generate_risk_prose(**kwargs)
        except Exception as exc:
            LOGGER.warning("[RiskWriter] LLM call failed (%s: %s) — using mock", type(exc).__name__, exc)
            return self._mock_generate(**kwargs)

    @staticmethod
    def _mock_generate(*, question: str, risks: list[dict], top_risk_statement: str):
        from research_agent.claude_client import RiskProsePayload
        high = [r for r in risks if r.get("severity", "").lower() == "high"]
        mitigations = [r.get("mitigation_notes", "") for r in risks if r.get("mitigation_notes")]
        high_bullets = [
            f"{r.get('statement', '')[:90]} (Severity: {r.get('severity')}, Likelihood: {r.get('likelihood')})"
            for r in high[:6]
        ] or ["High-severity risks require mitigation before commitment."]
        mit_bullets = [m[:100] for m in mitigations[:5]] or ["Standard risk mitigation protocols apply."]
        return RiskProsePayload(
            paragraphs=[
                (
                    f"{len(risks)} risk(s) identified for: {question}. "
                    f"{len(high)} classified as High-severity requiring attention before commitment."
                ),
                (
                    f"The principal risk is: {top_risk_statement[:140]}. "
                    "This risk carries the greatest potential to affect the viability of the recommended direction."
                ),
                (
                    f"Mitigation strategies exist for {len(mitigations)} of {len(risks)} identified risk(s). "
                    "Executing these mitigations reduces residual exposure and strengthens the strategic case."
                ),
                (
                    "Residual risk remains and should be monitored against the critical assumptions identified in the decision analysis. "
                    "Escalation criteria should be defined before commitment."
                ),
            ],
            bullet_groups=[high_bullets, mit_bullets],
        )
