"""AssumptionAgent – identifies strategic assumptions from evidence and hypotheses (J7.1).

Runs between ChallengeAgent and RecommendationAgent in the functional pipeline.

Reads:
  - context.surviving_hypotheses
  - context.hypothesis_challenges
  - context.evidence_notes
  - context.decision_model
  - context.research_strategy

Writes:
  - context.assumptions                               (list of assumption dicts)
  - context.research_object["strategic_assumptions"]
  - context.trace["_assumptions"]

Also persists the assumptions into the Decision Model artifact via
research_agent.decision_model.write_decision_model().
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class AssumptionAgent(FunctionalAgent):
    """Produces 5–10 strategic assumptions from challenged hypotheses and evidence.

    Each DecisionAssumption:
      - states what must be true for a recommendation to remain valid
      - is evidence-supported (evidence_ids)
      - carries its own confidence and evidence_support ratings
      - flags conflicts with other assumptions
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

        surviving_hypotheses: list[dict] = context.surviving_hypotheses
        hypothesis_challenges: list[dict] = context.hypothesis_challenges

        if not surviving_hypotheses and not hypothesis_challenges:
            LOGGER.warning("[AssumptionAgent] no hypotheses available — generating from evidence only")

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        payload = self._generate_assumptions(
            surviving_hypotheses=surviving_hypotheses,
            hypothesis_challenges=hypothesis_challenges,
            evidence_items=evidence_items,
            decision_model=context.decision_model,
            research_strategy=context.research_strategy,
        )

        assumptions_as_dicts = [a.model_dump() for a in payload.assumptions]

        # Resolve conflicts: ensure conflicts_with is populated in both directions
        assumptions_as_dicts = _resolve_conflicts(assumptions_as_dicts, payload.conflict_pairs)

        # Store in context
        context.assumptions = assumptions_as_dicts

        if context.research_object:
            context.research_object["strategic_assumptions"] = assumptions_as_dicts

        context.trace["_assumptions"] = {
            "count": len(assumptions_as_dicts),
            "conflict_pairs": payload.conflict_pairs,
            "assumptions": assumptions_as_dicts,
        }

        # Persist into Decision Model artifact when decision_model_id is available
        dm_id: str | None = (
            context.research_object.get("decision_model_id")
            if context.research_object
            else None
        )
        if dm_id:
            _persist_assumptions_to_dm(dm_id, assumptions_as_dicts)

        LOGGER.log(
            PROGRESS,
            "[AssumptionAgent] %d assumptions generated; %d conflict pairs detected",
            len(assumptions_as_dicts),
            len(payload.conflict_pairs),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(assumptions_as_dicts)} strategic assumptions generated; "
                f"{len(payload.conflict_pairs)} conflict pair(s) detected."
            ),
            assumption_count=len(assumptions_as_dicts),
            conflict_pairs=len(payload.conflict_pairs),
            critical_count=sum(1 for a in assumptions_as_dicts if a.get("importance") == "Critical"),
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_assumptions(
        self,
        *,
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
        research_strategy: dict,
    ):
        from research_agent.claude_client import AssumptionPayload

        if self._client is None:
            LOGGER.warning("[AssumptionAgent] no client — using mock assumptions")
            return _mock_assumptions(surviving_hypotheses, evidence_items, decision_model)

        if hasattr(self._client, "generate_assumptions"):
            return self._client.generate_assumptions(
                surviving_hypotheses=surviving_hypotheses,
                hypothesis_challenges=hypothesis_challenges,
                evidence_items=evidence_items,
                decision_model=decision_model,
                research_strategy=research_strategy,
            )

        LOGGER.warning("[AssumptionAgent] client lacks generate_assumptions — using mock")
        return _mock_assumptions(surviving_hypotheses, evidence_items, decision_model)


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

def _resolve_conflicts(
    assumptions: list[dict],
    conflict_pairs: list[list[str]],
) -> list[dict]:
    """Ensure conflicts_with is symmetrically populated from conflict_pairs."""
    id_map = {a["assumption_id"]: a for a in assumptions}
    for pair in conflict_pairs:
        if len(pair) == 2:
            a_id, b_id = pair[0], pair[1]
            if a_id in id_map and b_id not in id_map[a_id].get("conflicts_with", []):
                id_map[a_id].setdefault("conflicts_with", []).append(b_id)
            if b_id in id_map and a_id not in id_map[b_id].get("conflicts_with", []):
                id_map[b_id].setdefault("conflicts_with", []).append(a_id)
    return assumptions


# ---------------------------------------------------------------------------
# Decision Model persistence
# ---------------------------------------------------------------------------

def _persist_assumptions_to_dm(decision_model_id: str, assumptions: list[dict]) -> None:
    """Load the persisted DecisionModel, inject assumptions, and re-write it."""
    try:
        from research_agent.decision_model import (
            DecisionAssumption, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = [DecisionAssumption.model_validate(a) for a in assumptions]
        updated = dm.model_copy(update={"strategic_assumptions": parsed})
        write_decision_model(updated)
    except Exception as exc:
        LOGGER.warning("[AssumptionAgent] could not persist assumptions to DM: %s", exc)


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_assumptions(
    surviving_hypotheses: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
):
    from research_agent.claude_client import AssumptionItem, AssumptionPayload

    question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))
    ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]

    templates = [
        ("The underlying technology is sufficiently mature for commercial deployment", "Technology", "Critical", "Moderate", "Medium"),
        ("Market demand will remain at projected levels over the investment horizon", "Market", "Critical", "Strong", "High"),
        ("Capital costs will not materially exceed current estimates", "Economics", "Important", "Moderate", "Medium"),
        ("The regulatory environment will remain stable and permissive", "Regulation", "Critical", "Weak", "Low"),
        ("Supply chain constraints will be resolved within the planning timeframe", "Supply Chain", "Important", "Moderate", "Medium"),
    ]

    assumptions = []
    for i, (stmt, cat, imp, ev_sup, conf) in enumerate(templates):
        sup_ev = ev_ids[i*2 : i*2+2] if len(ev_ids) > i*2 else ev_ids[:1]
        assumptions.append(AssumptionItem(
            assumption_id=f"A-{i+1:03d}",
            statement=stmt,
            category=cat,
            importance=imp,
            evidence_support=ev_sup,
            confidence=conf,
            rationale=f"Strategic assumption relevant to: {question[:80]}",
            evidence_ids=sup_ev,
            conflicts_with=[],
            status="Active",
        ))

    # Mock conflict: market demand (A-002) vs regulatory environment (A-004)
    conflict_pairs: list[list[str]] = []
    if len(assumptions) >= 4:
        assumptions[1].conflicts_with.append("A-004")
        assumptions[3].conflicts_with.append("A-002")
        conflict_pairs.append(["A-002", "A-004"])

    return AssumptionPayload(assumptions=assumptions, conflict_pairs=conflict_pairs)
