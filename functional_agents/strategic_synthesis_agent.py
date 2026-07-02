"""StrategicSynthesisAgent — cross-domain strategic synthesis (J10.7).

Runs after HypothesisAgent and before ChallengeAgent. It is the first true
executive-reasoning layer: it integrates the independent per-Decision-Domain
reasoning streams (plans, evidence, hypotheses) into ONE executive perspective —
making implicit cross-domain relationships (dependencies, conflicts, leverage
points, dominant constraints) explicit.

This is NOT recommendation generation. It produces executive reasoning only:
no recommendations, no implementation plans. The existing primary execution path
(primary hypotheses → Challenge → Recommendation) is unchanged; this agent is
purely additive.

Writes to:
  - context.strategic_synthesis
  - context.research_object["strategic_synthesis"]
  - context.trace["_strategic_synthesis"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class StrategicSynthesisAgent(FunctionalAgent):
    """Integrates per-domain reasoning into one executive strategic perspective."""

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

        domain_plans = context.domain_plans or []
        domain_evidence = context.domain_evidence or []
        domain_hypotheses = context.domain_hypotheses or []
        decision_architecture = context.decision_architecture or {}

        domains_received = max(
            len(domain_plans), len(domain_evidence), len(domain_hypotheses)
        )

        # No-op guard: nothing to synthesize (e.g. pre-framing or empty run).
        if domains_received == 0:
            LOGGER.warning("[StrategicSynthesisAgent] no domains — skipping synthesis")
            context.trace["_strategic_synthesis"] = {"skipped": True, "reason": "no_domains"}
            self._record(context, status="skipped", summary="No decision domains — synthesis skipped.")
            return context

        payload = self._generate_synthesis(
            domain_plans, domain_evidence, domain_hypotheses, decision_architecture
        )
        synthesis = payload.model_dump()

        context.strategic_synthesis = synthesis
        if context.research_object is not None:
            context.research_object["strategic_synthesis"] = synthesis

        diagnostics = {
            "domains_received": domains_received,
            "dependencies_identified": len(synthesis.get("cross_domain_dependencies", [])),
            "conflicts_identified": len(synthesis.get("cross_domain_conflicts", [])),
            "strategic_themes": len(synthesis.get("emerging_themes", [])),
        }
        context.trace["_strategic_synthesis"] = {**synthesis, "diagnostics": diagnostics}

        LOGGER.log(
            PROGRESS,
            "[StrategicSynthesisAgent] domains=%d  dependencies=%d  conflicts=%d  themes=%d",
            domains_received,
            diagnostics["dependencies_identified"],
            diagnostics["conflicts_identified"],
            diagnostics["strategic_themes"],
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Cross-domain synthesis over {domains_received} domains: "
                f"{diagnostics['dependencies_identified']} dependencies, "
                f"{diagnostics['conflicts_identified']} conflicts, "
                f"{diagnostics['strategic_themes']} themes."
            ),
            domains_received=domains_received,
            dependencies_identified=diagnostics["dependencies_identified"],
            conflicts_identified=diagnostics["conflicts_identified"],
            strategic_themes=diagnostics["strategic_themes"],
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_synthesis(
        self,
        domain_plans: list[dict],
        domain_evidence: list[dict],
        domain_hypotheses: list[dict],
        decision_architecture: dict,
    ):
        """One integration call; deterministic mock fallback on no-client/error."""
        from research_agent.claude_client import MockClaudeClient

        if self._client is not None and hasattr(self._client, "generate_strategic_synthesis"):
            try:
                return self._client.generate_strategic_synthesis(
                    domain_plans, domain_evidence, domain_hypotheses, decision_architecture,
                )
            except Exception as exc:
                LOGGER.warning(
                    "[StrategicSynthesisAgent] synthesis failed (%s: %s) — deterministic fallback.",
                    type(exc).__name__, exc,
                )

        return MockClaudeClient().generate_strategic_synthesis(
            domain_plans, domain_evidence, domain_hypotheses, decision_architecture,
        )
