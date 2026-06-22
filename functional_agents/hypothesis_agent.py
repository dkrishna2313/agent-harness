"""HypothesisAgent – generates competing hypotheses from evidence and context (J6.3).

Runs between EvidenceAgent and QAAgent. Reads evidence, decision model,
research strategy, and profile coverage, then generates 3-5 competing
hypotheses each with evidence mappings, confidence, decision implications,
and disconfirming evidence needs.

Writes to:
  - context.hypotheses         (list of hypothesis dicts)
  - context.research_object["hypotheses"]
  - context.trace["_hypotheses"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class HypothesisAgent(FunctionalAgent):
    """Generates competing hypotheses from evidence and decision context.

    Each HypothesisItem contains:
      - id, title, summary, type
      - supporting_evidence        : evidence IDs that support the hypothesis
      - contradicting_evidence     : evidence IDs that weaken it
      - evidence_gaps              : missing evidence needed to test it
      - confidence                 : "high" | "medium" | "low"
      - confidence_rationale       : explanation of confidence level
      - decision_implications      : concrete strategic actions
      - disconfirming_evidence_needed : evidence that would invalidate the hypothesis
    """

    def __init__(
        self,
        *,
        client: Any = None,
        domain_profiles: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._domain_profiles = domain_profiles or []

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        # Derive profile_coverage in the flat {name: level} format expected by the prompt
        raw_coverage = evidence_note.get("profile_coverage_by_profile", {})
        profile_coverage: dict[str, str] = {
            pname: entry.get("coverage_level", "NONE").lower()
            for pname, entry in raw_coverage.items()
        }
        if not profile_coverage and context.profiles:
            profile_coverage = {p: "unknown" for p in context.profiles}

        contradictions: list[dict] = context.research_object.get("contradictions", [])

        hypothesis_payload = self._generate_hypotheses(
            context.decision_model,
            context.research_strategy,
            evidence_items,
            profile_coverage,
            contradictions,
        )

        hypotheses_as_dicts = [h.model_dump() for h in hypothesis_payload.hypotheses]

        context.hypotheses = hypotheses_as_dicts

        if context.research_object:
            context.research_object["hypotheses"] = hypotheses_as_dicts

        context.trace["_hypotheses"] = {
            "hypotheses": hypotheses_as_dicts,
            "synthesis_note": hypothesis_payload.synthesis_note,
        }

        LOGGER.log(
            PROGRESS,
            "[HypothesisAgent] generated=%d  synthesis=%r",
            len(hypothesis_payload.hypotheses),
            hypothesis_payload.synthesis_note[:80],
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(hypothesis_payload.hypotheses)} competing hypotheses generated. "
                + hypothesis_payload.synthesis_note[:100]
            ),
            hypothesis_count=len(hypothesis_payload.hypotheses),
            high_confidence=sum(1 for h in hypothesis_payload.hypotheses if h.confidence == "high"),
            medium_confidence=sum(1 for h in hypothesis_payload.hypotheses if h.confidence == "medium"),
            low_confidence=sum(1 for h in hypothesis_payload.hypotheses if h.confidence == "low"),
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_hypotheses(
        self,
        decision_model: dict,
        research_strategy: dict,
        evidence_items: list[dict],
        profile_coverage: dict,
        contradictions: list[dict],
    ):
        """Call the LLM client to generate competing hypotheses."""
        from research_agent.claude_client import HypothesisPayload, HypothesisItem

        if self._client is None:
            LOGGER.warning("[HypothesisAgent] no client provided — using mock hypotheses")
            return self._mock_hypotheses(decision_model, evidence_items)

        if hasattr(self._client, "generate_hypotheses"):
            return self._client.generate_hypotheses(
                decision_model, research_strategy,
                evidence_items, profile_coverage, contradictions,
            )

        LOGGER.warning("[HypothesisAgent] client does not support generate_hypotheses — using mock")
        return self._mock_hypotheses(decision_model, evidence_items)

    def _mock_hypotheses(self, decision_model: dict, evidence_items: list[dict]):
        """Deterministic fallback hypotheses."""
        from research_agent.claude_client import HypothesisPayload, HypothesisItem, MockClaudeClient
        return MockClaudeClient().generate_hypotheses(
            decision_model=decision_model,
            research_strategy={},
            evidence_items=evidence_items,
            profile_coverage={},
            contradictions=[],
        )
