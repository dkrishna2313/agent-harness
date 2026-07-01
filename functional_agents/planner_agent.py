"""PlannerAgent – classifies the question and generates a research plan (J5.1)."""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class PlannerAgent(FunctionalAgent):
    """Calls Claude (or mock) to classify the question and generate:
    - research_type (FACT_LOOKUP / COMPARISON / EXPLANATION / RESEARCH)
    - subquestions (3-7 focused decompositions)
    - investigation_areas (4-8 topic labels)

    Results are written into context.plan, the Research Object, and agent_history.
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

        profiles_context = self._build_profiles_context(context)

        # J10.4 — multi-domain planning. Generate one plan per Reasoning Target
        # (one per Decision Domain in engagement mode; a single target in goal/
        # question mode). Only the PRIMARY plan (targets[0]) executes downstream —
        # context.plan is pinned to it and is byte-identical to J10.3, so no
        # downstream agent sees any change. domain_plans is organizational only.
        targets = context.get_reasoning_targets()
        primary_target = targets[0] if targets else None

        # Fall back to context.question when no targets exist yet (e.g. a run
        # before ProblemFramingAgent populates the question).
        planning_targets = targets if targets else ([None] if context.question else [])

        domain_plans: list[dict] = []
        for i, target in enumerate(planning_targets):
            planning_question = target.question if target is not None else context.question
            plan = self._generate_plan(
                planning_question,
                profiles_context,
                decision_model=context.decision_model or None,
                research_strategy=context.research_strategy or None,
            )
            # Existing planning schema (unchanged) …
            plan_obj = {
                "question": planning_question,
                "research_type": plan.research_type,
                "subquestions": plan.subquestions,
                "investigation_areas": plan.investigation_areas,
                "profiles_used": plan.profiles_used,
                "reasoning": plan.reasoning,
            }
            # … wrapped with organizational metadata for domain_plans only.
            domain_plans.append({
                **plan_obj,
                "decision_domain_id": target.decision_domain_id if target else None,
                "decision_domain_title": target.decision_domain_title if target else None,
                "target_kind": target.kind if target else None,
                "is_primary": i == 0,
            })

        context.domain_plans = domain_plans

        # Primary plan drives the pipeline. Keep context.plan to the EXISTING
        # 6-key schema (strip the organizational metadata) so it is identical to
        # prior milestones and no downstream consumer changes.
        _primary = domain_plans[0] if domain_plans else {}
        context.plan = {
            "question": _primary.get("question", context.question),
            "research_type": _primary.get("research_type", ""),
            "subquestions": _primary.get("subquestions", []),
            "investigation_areas": _primary.get("investigation_areas", []),
            "profiles_used": _primary.get("profiles_used", []),
            "reasoning": _primary.get("reasoning", ""),
        }
        # J10.2/J10.4 — planner diagnostics (existing fields retained; additive).
        context.trace["_planner_reasoning"] = {
            "targets_received": len(targets),
            "targets_planned": 1 if domain_plans else 0,   # retained (J10.2)
            "plans_generated": len(domain_plans),           # J10.4
            "plans_executed": 1 if domain_plans else 0,     # J10.4
            "primary_target_kind": primary_target.kind if primary_target else None,
        }

        # Write PRIMARY plan fields into the Research Object (J5.1.6) — unchanged.
        _p = context.plan
        if context.research_object:
            context.research_object["research_type"] = _p["research_type"]
            context.research_object["subquestions"] = _p["subquestions"]
            context.research_object["investigation_areas"] = _p["investigation_areas"]

        LOGGER.log(
            PROGRESS,
            "[PlannerAgent] type=%s  subquestions=%d  areas=%d  domain_plans=%d",
            _p["research_type"],
            len(_p["subquestions"]),
            len(_p["investigation_areas"]),
            len(domain_plans),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Classified as {_p['research_type']}; "
                f"generated {len(_p['subquestions'])} subquestions and "
                f"{len(_p['investigation_areas'])} investigation areas "
                f"({len(domain_plans)} domain plan(s), 1 executed)."
            ),
            research_type=_p["research_type"],
            subquestions_generated=len(_p["subquestions"]),
            investigation_areas_generated=len(_p["investigation_areas"]),
            domain_plans_generated=len(domain_plans),
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_profiles_context(self, context: AgentContext) -> list[dict]:
        """Build a lightweight profile summary list for the planning prompt."""
        result: list[dict] = []
        # Prefer loaded DomainProfile objects when available
        profile_map: dict[str, Any] = {
            p.name: p for p in self._domain_profiles if hasattr(p, "name")
        }
        for name in context.profiles:
            if name in profile_map:
                p = profile_map[name]
                result.append({
                    "name": name,
                    "description": getattr(p, "description", ""),
                    "key_topics": list(getattr(p, "evaluator_topic_terms", {}).keys())[:8],
                })
            else:
                result.append({"name": name, "description": "", "key_topics": []})
        return result

    def _generate_plan(
        self,
        question: str,
        profiles_context: list[dict],
        decision_model: dict | None = None,
        research_strategy: dict | None = None,
    ):
        """Call the LLM client to generate the research plan.

        When a Decision Model is provided (goal-driven runs), it is passed to
        the planning prompt so the plan is grounded in the pre-derived research
        questions and decision areas rather than re-deriving them from scratch.
        """
        from research_agent.claude_client import ResearchPlanningPayload

        if self._client is None:
            LOGGER.warning("[PlannerAgent] no client provided — using mock plan")
            # In goal-driven mode, seed the plan from the decision model
            subquestions = (
                list(decision_model.get("research_questions", []))
                if decision_model else []
            ) or [
                f"What are the key facts about: {question}?",
                "What evidence exists in the available sources?",
                "What are the main constraints or limitations?",
                "What are the practical implications?",
                "What gaps remain in the available evidence?",
            ]
            investigation_areas = (
                list(decision_model.get("decision_areas", []))
                if decision_model else []
            ) or ["Overview", "Key Facts", "Evidence Quality", "Implications", "Open Questions"]
            return ResearchPlanningPayload(
                research_type="RESEARCH",
                subquestions=subquestions,
                investigation_areas=investigation_areas,
                profiles_used=[p.get("name", "") for p in profiles_context],
                reasoning="No client available; plan seeded from decision model." if decision_model
                    else "No client available; using default plan structure.",
            )

        if hasattr(self._client, "plan_research_question"):
            return self._client.plan_research_question(
                question, profiles_context,
                decision_model=decision_model,
                research_strategy=research_strategy,
            )

        # Fallback for clients that predate this method
        LOGGER.warning("[PlannerAgent] client does not support plan_research_question — using mock plan")
        from research_agent.claude_client import MockClaudeClient
        return MockClaudeClient().plan_research_question(question, profiles_context)
