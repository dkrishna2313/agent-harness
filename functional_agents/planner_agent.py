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

        # PH2.1 — every plan passes the explicit LLM boundary
        # (raw → normalize → validate → typed PlannerOutput) before it becomes
        # business logic. A boundary failure is deterministic and its stage is
        # recorded in context.trace["_planner_boundary"].
        from .planner_boundary import PlannerBoundaryError

        # PH3.3 — slice decision_model/research_strategy down to the exact
        # sub-keys _planning_prompt (research_agent/claude_client.py) actually
        # reads. Verified-unread fields cannot affect the prompt text or the
        # mock's output (see context_slices.py module docstring). The same
        # slice is reused for every reasoning-target iteration below since
        # decision_model/research_strategy do not vary per target.
        from .context_slices import planner_input_slice, record_slice_diagnostics
        _slice = planner_input_slice(context)
        sliced_decision_model = _slice["decision_model"] or None
        sliced_research_strategy = _slice["research_strategy"] or None
        record_slice_diagnostics(
            context, "planner",
            original={
                "decision_model": context.decision_model or {},
                "research_strategy": context.research_strategy or {},
            },
            sliced={
                "decision_model": _slice["decision_model"],
                "research_strategy": _slice["research_strategy"],
            },
        )

        domain_plans: list[dict] = []
        primary_boundary: dict | None = None
        try:
            for i, target in enumerate(planning_targets):
                planning_question = target.question if target is not None else context.question
                plan, boundary = self._generate_plan(
                    planning_question,
                    profiles_context,
                    decision_model=sliced_decision_model,
                    research_strategy=sliced_research_strategy,
                )
                if i == 0:
                    primary_boundary = boundary
                # Existing planning schema (unchanged), now sourced from the typed
                # PlannerOutput the boundary produced …
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
        except PlannerBoundaryError as exc:
            # Deterministic failure: record the failing stage; do not run business
            # logic on unvalidated output.
            context.trace["_planner_boundary"] = exc.diagnostics
            self._record(
                context, status="error",
                summary=(f"Planner boundary failed at stage "
                         f"'{exc.diagnostics.get('failed_stage', 'boundary')}': {exc}"),
                boundary_failed_stage=exc.diagnostics.get("failed_stage", "boundary"),
            )
            raise

        context.trace["_planner_boundary"] = primary_boundary or {
            "stages": {}, "failed_stage": None
        }
        # PH3.1 — record boundary normalization/validation timings (no-op w/o tracker)
        from .performance import record_boundary_stages
        record_boundary_stages(context, primary_boundary)
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
        """PH2.1 boundary: raw LLM payload → normalize → validate → PlannerOutput.

        Returns (PlannerOutput, boundary_diagnostics). Raises a distinct
        PlannerBoundaryError subclass on generation / normalization / validation
        failure so business logic never consumes unvalidated output.
        """
        from .planner_boundary import plan_from_raw

        raw = self._raw_planner_payload(
            question, profiles_context, decision_model, research_strategy
        )
        return plan_from_raw(raw)

    def _raw_planner_payload(
        self,
        question: str,
        profiles_context: list[dict],
        decision_model: dict | None,
        research_strategy: dict | None,
    ) -> dict:
        """Obtain the RAW (unvalidated) planner payload dict (LLM-generation stage)."""
        from .planner_boundary import PlannerGenerationError

        if self._client is None:
            LOGGER.warning("[PlannerAgent] no client provided — using deterministic default plan")
            return self._default_plan_dict(question, profiles_context, decision_model)

        try:
            if hasattr(self._client, "plan_research_question_raw"):
                return self._client.plan_research_question_raw(
                    question, profiles_context,
                    decision_model=decision_model, research_strategy=research_strategy,
                )
            if hasattr(self._client, "plan_research_question"):
                payload = self._client.plan_research_question(
                    question, profiles_context,
                    decision_model=decision_model, research_strategy=research_strategy,
                )
                return payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
        except Exception as exc:
            raise PlannerGenerationError(
                f"planner LLM generation failed: {exc}",
                {"failed_stage": "generation"},
            ) from exc

        # Client predates the planner method — deterministic mock payload.
        LOGGER.warning("[PlannerAgent] client lacks plan_research_question — using mock payload")
        from research_agent.claude_client import MockClaudeClient
        return MockClaudeClient().plan_research_question(question, profiles_context).model_dump()

    def _default_plan_dict(
        self, question: str, profiles_context: list[dict], decision_model: dict | None
    ) -> dict:
        """Deterministic default planner payload (no client), seeded from the DM."""
        subquestions = (
            list(decision_model.get("research_questions", [])) if decision_model else []
        ) or [
            f"What are the key facts about: {question}?",
            "What evidence exists in the available sources?",
            "What are the main constraints or limitations?",
            "What are the practical implications?",
            "What gaps remain in the available evidence?",
        ]
        investigation_areas = (
            list(decision_model.get("decision_areas", [])) if decision_model else []
        ) or ["Overview", "Key Facts", "Evidence Quality", "Implications", "Open Questions"]
        return {
            "research_type": "RESEARCH",
            "subquestions": subquestions,
            "investigation_areas": investigation_areas,
            "profiles_used": [p.get("name", "") for p in profiles_context],
            "reasoning": ("No client available; plan seeded from decision model."
                          if decision_model else "No client available; using default plan structure."),
        }
