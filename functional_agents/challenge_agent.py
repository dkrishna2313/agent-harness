"""ChallengeAgent – stress-tests hypotheses to surface weaknesses (J6.4).

Runs between HypothesisAgent and QAAgent. For each hypothesis it identifies:
  - hidden assumptions
  - weak or insufficient evidence
  - contradicting evidence references
  - missing evidence
  - falsification tests
  - robustness score (low | medium | high)

Also produces a surviving_hypotheses list (strong | moderate | weak).

Writes to:
  - context.hypothesis_challenges     (list of challenge dicts)
  - context.surviving_hypotheses      (list of survival dicts)
  - context.research_object["hypothesis_challenges"]
  - context.research_object["surviving_hypotheses"]
  - context.trace["_challenges"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class ChallengeAgent(FunctionalAgent):
    """Adversarially challenges each hypothesis from HypothesisAgent.

    Each ChallengeItem contains:
      - hypothesis_id           : which hypothesis is being challenged
      - challenge_summary       : main weakness in 1-3 sentences
      - hidden_assumptions      : implicit unverified assumptions
      - weak_evidence           : evidence quality problems
      - contradicting_evidence  : evidence IDs that weaken the hypothesis
      - missing_evidence        : absent evidence needed to validate
      - falsification_tests     : conditions that would invalidate the hypothesis
      - robustness              : "low" | "medium" | "high"

    Each SurvivingHypothesis contains:
      - hypothesis_id    : which hypothesis
      - survival_status  : "strong" | "moderate" | "weak"
      - reason           : rationale for survival status
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

        hypotheses: list[dict] = context.hypotheses
        if not hypotheses:
            LOGGER.warning("[ChallengeAgent] no hypotheses to challenge — skipping")
            self._record(context, status="warning", summary="No hypotheses available to challenge.")
            return context

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        # Derive profile_coverage flat dict
        raw_coverage = evidence_note.get("profile_coverage_by_profile", {})
        profile_coverage: dict[str, str] = {
            pname: entry.get("coverage_level", "NONE").lower()
            for pname, entry in raw_coverage.items()
        }
        if not profile_coverage and context.profiles:
            profile_coverage = {p: "unknown" for p in context.profiles}

        contradictions: list[dict] = context.research_object.get("contradictions", [])
        research_gaps: list[dict] = context.research_object.get("gaps", [])

        challenge_payload = self._generate_challenges(
            hypotheses, evidence_items, contradictions, research_gaps, profile_coverage,
        )

        challenges_as_dicts = [c.model_dump() for c in challenge_payload.hypothesis_challenges]
        surviving_as_dicts = [s.model_dump() for s in challenge_payload.surviving_hypotheses]

        context.hypothesis_challenges = challenges_as_dicts
        context.surviving_hypotheses = surviving_as_dicts

        if context.research_object:
            context.research_object["hypothesis_challenges"] = challenges_as_dicts
            context.research_object["surviving_hypotheses"] = surviving_as_dicts

        context.trace["_challenges"] = {
            "hypothesis_challenges": challenges_as_dicts,
            "surviving_hypotheses": surviving_as_dicts,
            "challenge_synthesis": challenge_payload.challenge_synthesis,
        }

        robust_counts = _count_robustness(challenges_as_dicts)
        strong_count = sum(1 for s in surviving_as_dicts if s.get("survival_status") == "strong")
        weak_count = sum(1 for s in surviving_as_dicts if s.get("survival_status") == "weak")

        LOGGER.log(
            PROGRESS,
            "[ChallengeAgent] challenged=%d  surviving: strong=%d weak=%d  "
            "robustness: high=%d medium=%d low=%d",
            len(challenges_as_dicts),
            strong_count, weak_count,
            robust_counts["high"], robust_counts["medium"], robust_counts["low"],
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(challenges_as_dicts)} hypotheses challenged. "
                f"strong={strong_count}, weak={weak_count}. "
                + challenge_payload.challenge_synthesis[:100]
            ),
            challenge_count=len(challenges_as_dicts),
            strong_surviving=strong_count,
            weak_surviving=weak_count,
            robustness_high=robust_counts["high"],
            robustness_medium=robust_counts["medium"],
            robustness_low=robust_counts["low"],
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_challenges(
        self,
        hypotheses: list[dict],
        evidence_items: list[dict],
        contradictions: list[dict],
        research_gaps: list[dict],
        profile_coverage: dict,
    ):
        """Call the LLM client to challenge each hypothesis."""
        if self._client is None:
            LOGGER.warning("[ChallengeAgent] no client — using mock challenges")
            return self._mock_challenges(hypotheses, evidence_items, contradictions)

        if hasattr(self._client, "generate_challenges"):
            return self._client.generate_challenges(
                hypotheses, evidence_items, contradictions, research_gaps, profile_coverage,
            )

        LOGGER.warning("[ChallengeAgent] client does not support generate_challenges — using mock")
        return self._mock_challenges(hypotheses, evidence_items, contradictions)

    def _mock_challenges(
        self,
        hypotheses: list[dict],
        evidence_items: list[dict],
        contradictions: list[dict],
    ):
        from research_agent.claude_client import MockClaudeClient
        return MockClaudeClient().generate_challenges(
            hypotheses=hypotheses,
            evidence_items=evidence_items,
            contradictions=contradictions,
            research_gaps=[],
            profile_coverage={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_robustness(challenges: list[dict]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for c in challenges:
        r = c.get("robustness", "medium").lower()
        if r in counts:
            counts[r] += 1
    return counts
