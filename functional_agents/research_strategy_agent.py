"""ResearchStrategyAgent – transforms a Decision Model into an executable research plan (J6.2).

Runs between ProblemFramingAgent and PlannerAgent in goal-driven workflows.
Reads context.decision_model, calls Claude to produce a ResearchStrategyPayload,
and writes it to:
  - context.research_strategy
  - context.research_object["research_strategy"]
  - context.trace["_research_strategy"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# J9.1b — hard caps on the strategy object. These fields are routing metadata
# (consumed only by the report trace), so bounding them cannot degrade reasoning
# quality but does prevent the LLM from producing output that overflows the
# generate_research_strategy max_tokens ceiling and truncating the tool call.
MAX_RESEARCH_QUESTIONS = 6
MAX_REQUIRED_EVIDENCE = 6
MAX_SOURCE_PRIORITIES = 5
MAX_COVERAGE_TARGETS = 8
# Output ceiling for generate_research_strategy. The bounded object is small
# (~600-800 tokens incl. tool-call JSON overhead); 2000 leaves ample headroom.
RESEARCH_STRATEGY_MAX_TOKENS = 2000
_CHARS_PER_TOKEN = 4  # rough token estimate for instrumentation only


class ResearchStrategyAgent(FunctionalAgent):
    """Converts a Decision Model into a prioritised research strategy.

    The ResearchStrategyPayload contains:
      - profile_priorities         : {profile_name: rank} — 1 = highest priority
      - research_question_priorities: [{question, priority}] ordered by decision impact
      - required_evidence          : specific evidence items needed
      - source_priorities          : source types in priority order
      - coverage_targets           : {area: "strong"|"moderate"|"light"}
      - strategy_rationale         : 2-3 sentence explanation
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

        if not context.decision_model:
            LOGGER.warning("[ResearchStrategyAgent] called with empty decision_model — skipping")
            self._record(context, status="warning", summary="No decision model; strategy skipped.")
            return context

        profiles_context = self._build_profiles_context(context)

        # J9.1b — estimate prompt size for instrumentation (best-effort).
        prompt_token_estimate = self._estimate_prompt_tokens(
            context.decision_model, profiles_context
        )

        strategy, diagnostics = self._generate_strategy(
            context.decision_model, profiles_context
        )
        # J9.1b — enforce caps post-hoc so the stored object is bounded even if the
        # model over-produces (defence in depth; the prompt/schema prevent it upstream).
        strategy = self._bound_strategy(strategy)

        rs_dict = strategy.model_dump()
        context.research_strategy = rs_dict

        if context.research_object:
            context.research_object["research_strategy"] = rs_dict

        context.trace["_research_strategy"] = rs_dict

        # J9.1b — record strategy generation diagnostics for prompt-growth audits.
        context.trace["_research_strategy_diagnostics"] = {
            "prompt_token_estimate": prompt_token_estimate,
            "max_tokens": RESEARCH_STRATEGY_MAX_TOKENS,
            "stop_reason": diagnostics.get("stop_reason", "end_turn"),
            "truncated": diagnostics.get("truncated", False),
            "used_fallback": diagnostics.get("used_fallback", False),
            "output_shape": {
                "research_questions": len(strategy.research_question_priorities),
                "required_evidence": len(strategy.required_evidence),
                "source_priorities": len(strategy.source_priorities),
                "coverage_targets": len(strategy.coverage_targets),
                "rationale_chars": len(strategy.strategy_rationale or ""),
            },
        }
        if diagnostics.get("truncated"):
            LOGGER.warning(
                "[ResearchStrategyAgent] live response truncated (stop_reason=max_tokens, "
                "limit=%d) — used bounded deterministic fallback.",
                RESEARCH_STRATEGY_MAX_TOKENS,
            )

        LOGGER.log(
            PROGRESS,
            "[ResearchStrategyAgent] profiles=%d  questions=%d  coverage_targets=%d",
            len(strategy.profile_priorities),
            len(strategy.research_question_priorities),
            len(strategy.coverage_targets),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Research strategy generated: {len(strategy.profile_priorities)} profile priorities, "
                f"{len(strategy.research_question_priorities)} question priorities, "
                f"{len(strategy.coverage_targets)} coverage targets."
            ),
            profile_priorities_count=len(strategy.profile_priorities),
            research_question_priorities_count=len(strategy.research_question_priorities),
            required_evidence_count=len(strategy.required_evidence),
            coverage_targets_count=len(strategy.coverage_targets),
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_profiles_context(self, context: AgentContext) -> list[dict]:
        """Build a lightweight profile summary list for the strategy prompt."""
        result: list[dict] = []
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

    def _estimate_prompt_tokens(self, decision_model: dict, profiles_context: list[dict]) -> int:
        """Best-effort prompt token estimate for instrumentation (J9.1b)."""
        try:
            from research_agent.claude_client import _strategy_prompt
            return len(_strategy_prompt(decision_model, profiles_context)) // _CHARS_PER_TOKEN
        except Exception:
            return 0

    def _generate_strategy(self, decision_model: dict, profiles_context: list[dict]):
        """Generate the Research Strategy.

        Returns (ResearchStrategyPayload, diagnostics). On a live max_tokens
        truncation the run does NOT fail: we fall back to the deterministic
        bounded strategy and flag it in diagnostics (J9.1b).
        """
        diagnostics: dict[str, Any] = {"stop_reason": "end_turn", "truncated": False, "used_fallback": False}

        if self._client is None:
            LOGGER.warning("[ResearchStrategyAgent] no client provided — using deterministic strategy")
            diagnostics["used_fallback"] = True
            return self._deterministic_strategy(decision_model, profiles_context), diagnostics

        if not hasattr(self._client, "generate_research_strategy"):
            LOGGER.warning("[ResearchStrategyAgent] client lacks generate_research_strategy — deterministic")
            diagnostics["used_fallback"] = True
            return self._deterministic_strategy(decision_model, profiles_context), diagnostics

        try:
            strategy = self._client.generate_research_strategy(decision_model, profiles_context)
            return strategy, diagnostics
        except RuntimeError as exc:
            # _call_json raises RuntimeError("...stop_reason=max_tokens...") on truncation.
            if "max_tokens" in str(exc):
                diagnostics.update(stop_reason="max_tokens", truncated=True, used_fallback=True)
                return self._deterministic_strategy(decision_model, profiles_context), diagnostics
            raise

    def _deterministic_strategy(self, decision_model: dict, profiles_context: list[dict]):
        """Build a bounded strategy from the decision model without an LLM call."""
        from research_agent.claude_client import ResearchStrategyPayload

        profiles = [p.get("name", "") for p in profiles_context if p.get("name")]
        rqs = decision_model.get("research_questions", [])[:MAX_RESEARCH_QUESTIONS]
        areas = decision_model.get("decision_areas", [])
        uncertainties = decision_model.get("critical_uncertainties", [])
        evidence_reqs = decision_model.get("evidence_requirements", [])
        return ResearchStrategyPayload(
            profile_priorities={p: i + 1 for i, p in enumerate(profiles)},
            research_question_priorities=[
                {"question": q, "priority": i + 1} for i, q in enumerate(rqs)
            ],
            required_evidence=(evidence_reqs or [
                "Primary data sources",
                "Expert assessments",
                "Quantitative benchmarks",
            ])[:MAX_REQUIRED_EVIDENCE],
            source_priorities=["primary research", "industry reports", "expert analysis", "case studies"][:MAX_SOURCE_PRIORITIES],
            coverage_targets={
                **{area: "strong" for area in areas[:2]},
                **{area: "moderate" for area in areas[2:]},
                **{u: "strong" for u in uncertainties[:1]},
            },
            strategy_rationale="Deterministic strategy: profiles ranked by order, questions ranked by position.",
        )

    def _bound_strategy(self, strategy):
        """Truncate strategy lists/maps to the J9.1b caps (defence in depth)."""
        try:
            updates = {
                "research_question_priorities": list(strategy.research_question_priorities)[:MAX_RESEARCH_QUESTIONS],
                "required_evidence": list(strategy.required_evidence)[:MAX_REQUIRED_EVIDENCE],
                "source_priorities": list(strategy.source_priorities)[:MAX_SOURCE_PRIORITIES],
                "coverage_targets": dict(list(strategy.coverage_targets.items())[:MAX_COVERAGE_TARGETS]),
            }
            return strategy.model_copy(update=updates)
        except Exception:
            return strategy
