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
        """J10.6 — generate hypotheses per Decision Domain; primary flows downstream.

        The PRIMARY domain (domain_evidence[0], whose evidence == the primary
        evidence in context.evidence_notes) runs on the real context, leaving
        context.hypotheses byte-identical to J10.5. Secondary domains run on
        isolated scratch contexts; their hypotheses are captured into
        context.domain_hypotheses. Goal/question mode has a single evidence set →
        single hypothesis set, unchanged.
        """
        domain_evidence = list(context.domain_evidence) if context.domain_evidence else []

        # Primary run on the real context (byte-identical to prior behaviour).
        self._execute_single(context)
        primary_meta = domain_evidence[0] if domain_evidence else {}
        domain_hypotheses = [self._capture_domain_hypotheses(context, primary_meta)]

        # Secondary domains on isolated scratch contexts (organizational only).
        for entry in domain_evidence[1:]:
            scratch = self._scratch_context(context, entry)
            try:
                self._execute_single(scratch)
                domain_hypotheses.append(self._capture_domain_hypotheses(scratch, entry))
            except Exception as exc:  # a secondary domain must never fail the run
                LOGGER.warning(
                    "[HypothesisAgent] secondary domain hypotheses failed (%s: %s) — skipping.",
                    type(exc).__name__, exc,
                )

        context.domain_hypotheses = domain_hypotheses

        primary_domain = (
            primary_meta.get("decision_domain_title") or context.question
        ) if isinstance(primary_meta, dict) else context.question
        context.trace["_hypothesis_reasoning"] = {
            "evidence_sets_received": len(domain_evidence),
            "hypothesis_sets_generated": len(domain_hypotheses),
            "hypothesis_sets_executed": 1 if domain_hypotheses else 0,
            "primary_domain": primary_domain,
        }
        return context

    def _execute_single(self, context: AgentContext) -> AgentContext:
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
    # J10.6 — multi-domain helpers
    # ------------------------------------------------------------------

    def _scratch_context(self, context: AgentContext, entry: dict) -> AgentContext:
        """Build an isolated context for a secondary domain's hypothesis pass.

        Reconstructs a minimal evidence_note (only evidence_items +
        profile_coverage are read by _execute_single) and isolates everything
        _execute_single mutates: hypotheses, research_object, trace, history.
        """
        import copy

        scratch = copy.copy(context)
        scratch.evidence_notes = [{
            "evidence_items": entry.get("evidence", []),
            "profile_coverage_by_profile": {},
        }]
        scratch.research_object = copy.deepcopy(context.research_object) if context.research_object else {}
        scratch.trace = {"_client": context.trace.get("_client")}
        scratch.hypotheses = []
        scratch.agent_history = []
        return scratch

    @staticmethod
    def _capture_domain_hypotheses(context: AgentContext, meta: dict) -> dict:
        """Extract one Decision Domain's hypothesis set + diagnostics (J10.6)."""
        meta = meta if isinstance(meta, dict) else {}
        synthesis_note = context.trace.get("_hypotheses", {}).get("synthesis_note", "")
        return {
            "decision_domain_id": meta.get("decision_domain_id"),
            "decision_domain_title": meta.get("decision_domain_title"),
            "hypotheses": context.hypotheses,
            "synthesis_note": synthesis_note,
            "diagnostics": {"hypothesis_count": len(context.hypotheses)},
        }

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
