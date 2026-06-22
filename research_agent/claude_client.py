"""Anthropic Claude integration for the research workflow."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .evidence_filter import sanitize_evidence_items
from .prompts import SYSTEM_PROMPT
from .schemas import (
    Chunk,
    ClaudeCallTrace,
    EvidenceItem,
    ResearchMemo,
    ResearchPlan,
    SourceDocument,
    assign_evidence_ids,
)

DEFAULT_MODEL = "claude-sonnet-4-6"
# Haiku is available as an opt-in via ANTHROPIC_EXTRACTION_MODEL env var, but
# defaults to Sonnet — Haiku extracts significantly fewer items per question
# which degrades citation coverage in the synthesized memo.
DEFAULT_EXTRACTION_MODEL = DEFAULT_MODEL


class EvidenceExtractionPayload(BaseModel):
    """Strict payload — used only to generate the JSON schema for the tool definition."""

    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class _RawEvidencePayload(BaseModel):
    """Lenient payload — used to validate Claude's response.

    Items are kept as raw dicts so that ``extract_evidence`` can validate them
    one-by-one and discard individual failures without losing the entire batch.
    """

    evidence_items: list = Field(default_factory=list)


class MemoSynthesisPayload(BaseModel):
    executive_summary: str = ""
    confirmed_facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    power_implications: list[str] = Field(default_factory=list)
    cooling_implications: list[str] = Field(default_factory=list)
    networking_implications: list[str] = Field(default_factory=list)
    rack_architecture_implications: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ResearchPlanningPayload(BaseModel):
    """Structured output for PlannerAgent (J5.1)."""

    research_type: str = Field(
        description="Question classification: FACT_LOOKUP, COMPARISON, EXPLANATION, or RESEARCH"
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description="3-7 focused subquestions that decompose the main question",
    )
    investigation_areas: list[str] = Field(
        default_factory=list,
        description="4-8 topic areas to investigate (e.g. Power, Cooling, Economics)",
    )
    profiles_used: list[str] = Field(
        default_factory=list,
        description="Profile names whose domain knowledge informed this plan",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the classification and planning choices",
    )


class DecisionModelPayload(BaseModel):
    """Structured output for ProblemFramingAgent (J6.1).

    Transforms a business goal into a structured Decision Model that guides
    the rest of the research pipeline.
    """

    objective: str = Field(
        description="The core decision objective, restated as a precise research goal"
    )
    decision_areas: list[str] = Field(
        default_factory=list,
        description="3-6 key decision areas or dimensions the research must address",
    )
    critical_uncertainties: list[str] = Field(
        default_factory=list,
        description="2-5 critical unknowns that most affect the decision outcome",
    )
    research_questions: list[str] = Field(
        default_factory=list,
        description="3-6 specific research questions derived from the goal",
    )
    evidence_requirements: list[str] = Field(
        default_factory=list,
        description="Types of evidence needed (e.g. market data, technical specs, case studies)",
    )


class ResearchStrategyPayload(BaseModel):
    """Structured output for ResearchStrategyAgent (J6.2).

    Translates the Decision Model into an executable research plan that guides
    profile selection, evidence gathering, and coverage targeting.
    """

    profile_priorities: dict[str, int] = Field(
        default_factory=dict,
        description="Profile name → integer priority rank (1 = highest). Lists all profiles in order of relevance to the decision model.",
    )
    research_question_priorities: list[dict] = Field(
        default_factory=list,
        description='Ordered list of {question: str, priority: int} dicts ranked by decision impact.',
    )
    required_evidence: list[str] = Field(
        default_factory=list,
        description="Specific evidence items needed (e.g. 'AI power demand forecasts', 'SMR deployment schedules')",
    )
    source_priorities: list[str] = Field(
        default_factory=list,
        description="Source types in priority order (e.g. 'regulatory filings', 'grid operator reports')",
    )
    coverage_targets: dict[str, str] = Field(
        default_factory=dict,
        description="Topic/area → required coverage level: 'strong', 'moderate', or 'light'",
    )
    strategy_rationale: str = Field(
        default="",
        description="2-3 sentence explanation of the strategy choices",
    )


_SCHEMA_ADAPTERS = {
    "research_plan": TypeAdapter(ResearchPlan),
    "research_planning": TypeAdapter(ResearchPlanningPayload),
    "problem_framing": TypeAdapter(DecisionModelPayload),
    "research_strategy": TypeAdapter(ResearchStrategyPayload),
    # Used for the tool-definition schema sent to Claude (strict EvidenceItem types).
    "evidence_extraction": TypeAdapter(EvidenceExtractionPayload),
    # Used for response validation (lenient — items validated per-item in extract_evidence).
    "evidence_extraction_raw": TypeAdapter(_RawEvidencePayload),
    "memo_synthesis": TypeAdapter(MemoSynthesisPayload),
}


class LLMClient(Protocol):
    """Minimal interface required by the agent."""

    is_mock: bool
    model: str
    call_traces: list[ClaudeCallTrace]


class MockClaudeClient:
    """Deterministic client used when Claude is unavailable."""

    is_mock = True
    model = "mock-claude"

    def __init__(self) -> None:
        self.call_traces: list[ClaudeCallTrace] = []

    def plan_research_question(
        self,
        question: str,
        profiles_context: list[dict],
        decision_model: dict | None = None,
        research_strategy: dict | None = None,
    ) -> ResearchPlanningPayload:
        q = question.lower()
        if any(w in q for w in ("compare", "vs", "versus", "difference between")):
            research_type = "COMPARISON"
        elif any(w in q for w in ("why", "how does", "explain", "what causes")):
            research_type = "EXPLANATION"
        elif any(w in q for w in ("what is", "what are", "how many", "how much", "list")):
            research_type = "FACT_LOOKUP"
        else:
            research_type = "RESEARCH"

        profiles_used = [p.get("name", "") for p in profiles_context if p.get("name")]
        # Seed from decision model when available (goal-driven runs)
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
            research_type=research_type,
            subquestions=subquestions,
            investigation_areas=investigation_areas,
            profiles_used=profiles_used,
            reasoning="Mock plan seeded from decision model." if decision_model else "Mock deterministic plan.",
        )

    def frame_problem(
        self,
        goal: str,
        profiles_context: list[dict],
    ) -> "DecisionModelPayload":
        """Return a deterministic decision model for the given business goal."""
        return DecisionModelPayload(
            objective=f"Research and analyse: {goal}",
            decision_areas=["Market Landscape", "Technical Feasibility", "Risk Assessment", "Investment Criteria"],
            critical_uncertainties=["Market timing", "Competitive dynamics", "Regulatory environment"],
            research_questions=[
                f"What is the current state of: {goal}?",
                "What are the key technical and market constraints?",
                "What evidence exists on investment returns and risk factors?",
                "What are the strategic options and their trade-offs?",
            ],
            evidence_requirements=["Market data", "Technical specifications", "Case studies", "Analyst reports"],
        )

    def generate_research_strategy(
        self,
        decision_model: dict,
        profiles_context: list[dict],
    ) -> "ResearchStrategyPayload":
        """Return a deterministic research strategy from a decision model."""
        profiles = [p.get("name", "") for p in profiles_context if p.get("name")]
        rqs = decision_model.get("research_questions", [])
        areas = decision_model.get("decision_areas", [])
        uncertainties = decision_model.get("critical_uncertainties", [])
        evidence_reqs = decision_model.get("evidence_requirements", [])
        return ResearchStrategyPayload(
            profile_priorities={p: i + 1 for i, p in enumerate(profiles)},
            research_question_priorities=[
                {"question": q, "priority": i + 1} for i, q in enumerate(rqs)
            ],
            required_evidence=evidence_reqs or [
                "Primary data sources",
                "Expert assessments",
                "Quantitative benchmarks",
            ],
            source_priorities=["primary research", "industry reports", "expert analysis", "case studies"],
            coverage_targets={
                **{area: "strong" for area in areas[:2]},
                **{area: "moderate" for area in areas[2:]},
                **{u: "strong" for u in uncertainties[:1]},
            },
            strategy_rationale="Mock strategy: profiles ranked by order, questions ranked by position, coverage targets set to strong/moderate.",
        )


class ClaudeClient:
    """Thin Anthropic SDK wrapper for structured research calls."""

    is_mock = False

    def __init__(
        self,
        *,
        model: str | None = None,
        extraction_model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4000,
        anthropic_client: Any | None = None,
        use_extraction_cache: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL
        # Extraction uses a fast cheap model by default; override with env var or arg.
        self.extraction_model = (
            extraction_model
            or os.getenv("ANTHROPIC_EXTRACTION_MODEL")
            or DEFAULT_EXTRACTION_MODEL
        )
        self.max_tokens = max_tokens
        self.call_traces: list[ClaudeCallTrace] = []

        from .extraction_cache import ExtractionCache
        self._extraction_cache: ExtractionCache | None = (
            ExtractionCache() if use_extraction_cache else None
        )
        LOGGER.debug(
            "ClaudeClient: synthesis_model=%s  extraction_model=%s  cache=%s",
            self.model,
            self.extraction_model,
            "enabled" if self._extraction_cache else "disabled",
        )

        if anthropic_client is not None:
            self._client = anthropic_client
            return

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Claude runs.")

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Install anthropic to use Claude.") from exc

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def frame_problem(
        self,
        goal: str,
        profiles_context: list[dict],
    ) -> DecisionModelPayload:
        """Transform a business goal into a structured Decision Model (J6.1)."""
        payload = self._call_json(
            operation="problem_framing",
            schema_name="problem_framing",
            prompt=_problem_framing_prompt(goal, profiles_context),
            max_tokens=2000,
        )
        return DecisionModelPayload.model_validate(payload)

    def generate_research_strategy(
        self,
        decision_model: dict,
        profiles_context: list[dict],
    ) -> ResearchStrategyPayload:
        """Transform a Decision Model into an executable research strategy (J6.2)."""
        payload = self._call_json(
            operation="generate_research_strategy",
            schema_name="research_strategy",
            prompt=_strategy_prompt(decision_model, profiles_context),
            max_tokens=2000,
        )
        return ResearchStrategyPayload.model_validate(payload)

    def plan_research_question(
        self,
        question: str,
        profiles_context: list[dict],
        decision_model: dict | None = None,
        research_strategy: dict | None = None,
    ) -> ResearchPlanningPayload:
        """Classify the question and generate a structured research plan (J5.1 / J6.1a)."""
        payload = self._call_json(
            operation="plan_research_question",
            schema_name="research_planning",
            prompt=_planning_prompt(
                question, profiles_context,
                decision_model=decision_model,
                research_strategy=research_strategy,
            ),
            max_tokens=2000,
        )
        return ResearchPlanningPayload.model_validate(payload)

    def create_research_plan(
        self,
        question: str,
        source_texts: Sequence[SourceDocument],
    ) -> ResearchPlan:
        payload = self._call_json(
            operation="create_research_plan",
            schema_name="research_plan",
            prompt=_research_plan_prompt(question, source_texts),
        )
        return ResearchPlan.model_validate(payload)

    def extract_evidence(
        self,
        question: str,
        source_texts: Sequence[SourceDocument],
    ) -> list[EvidenceItem]:
        # Each evidence item serialises to roughly 150 tokens; use a generous
        # ceiling so the response is never truncated mid-JSON.
        # response_schema_name uses the lenient schema so per-item validation
        # can discard bad items without failing the whole batch.
        payload = self._call_json(
            operation="extract_evidence",
            schema_name="evidence_extraction",
            prompt=_evidence_prompt(question, source_texts),
            max_tokens=max(self.max_tokens, 16_000),
            response_schema_name="evidence_extraction_raw",
        )
        raw_items = payload.get("evidence_items", [])
        LOGGER.debug("extract_evidence: raw item count from payload=%d", len(raw_items))

        validated: list[EvidenceItem] = []
        discarded = 0
        for item in raw_items:
            try:
                validated.append(EvidenceItem.model_validate(item))
            except Exception as exc:
                LOGGER.debug(
                    "extract_evidence: discarding item due to validation error: %s", exc
                )
                discarded += 1

        if discarded:
            LOGGER.warning(
                "extract_evidence: discarded %d of %d items due to validation errors",
                discarded,
                len(raw_items),
            )

        clean = sanitize_evidence_items(validated, stage="claude_extract_evidence")
        result = assign_evidence_ids(clean)
        LOGGER.debug("extract_evidence: final EvidenceItem count=%d", len(result))
        return result

    def extract_evidence_from_chunks(
        self,
        question: str,
        chunks: Sequence[Chunk],
    ) -> list[EvidenceItem]:
        chunk_list = list(chunks)

        # Cache read — skip LLM call on a hit
        if self._extraction_cache is not None:
            cached = self._extraction_cache.get(question, chunk_list)
            if cached is not None:
                from research_agent.log import PROGRESS
                LOGGER.log(PROGRESS, "[extraction_cache] hit  chunks=%d  items=%d", len(chunk_list), len(cached))
                return cached

        payload = self._call_json(
            operation="extract_evidence",
            schema_name="evidence_extraction",
            prompt=_evidence_chunk_prompt(question, chunk_list),
            max_tokens=max(self.max_tokens, 16_000),
            response_schema_name="evidence_extraction_raw",
            model_override=self.extraction_model,
        )
        raw_items = payload.get("evidence_items", [])
        LOGGER.debug("extract_evidence_from_chunks: raw item count from payload=%d", len(raw_items))

        validated: list[EvidenceItem] = []
        discarded = 0
        for item in raw_items:
            try:
                validated.append(EvidenceItem.model_validate(item))
            except Exception as exc:
                LOGGER.debug(
                    "extract_evidence_from_chunks: discarding item due to validation error: %s", exc
                )
                discarded += 1

        if discarded:
            LOGGER.warning(
                "extract_evidence_from_chunks: discarded %d of %d items due to validation errors",
                discarded,
                len(raw_items),
            )

        clean = sanitize_evidence_items(validated, stage="claude_extract_from_chunks")
        result = assign_evidence_ids(clean)
        LOGGER.debug("extract_evidence_from_chunks: final EvidenceItem count=%d", len(result))

        # Cache write
        if self._extraction_cache is not None:
            self._extraction_cache.put(question, chunk_list, result)

        return result

    def synthesize_memo(
        self,
        question: str,
        evidence_items: Sequence[EvidenceItem],
    ) -> ResearchMemo:
        payload = self._call_json(
            operation="synthesize_memo",
            schema_name="memo_synthesis",
            prompt=_memo_prompt(question, evidence_items),
            max_tokens=max(self.max_tokens, 12_000),
        )
        return ResearchMemo(
            title=f"Research Memo: {question}",
            question=question,
            executive_summary=_string_value(payload.get("executive_summary")),
            confirmed_facts=_string_list(payload.get("confirmed_facts")),
            inferences=_string_list(payload.get("inferences")),
            power_implications=_string_list(payload.get("power_implications")),
            cooling_implications=_string_list(payload.get("cooling_implications")),
            networking_implications=_string_list(payload.get("networking_implications")),
            rack_architecture_implications=_string_list(
                payload.get("rack_architecture_implications")
            ),
            open_questions=_string_list(payload.get("open_questions")),
            source_notes=list(evidence_items),
            evidence=list(evidence_items),
        )

    def _call_json(
        self,
        *,
        operation: str,
        schema_name: str,
        prompt: str,
        max_tokens: int | None = None,
        response_schema_name: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any] | list[Any]:
        # response_schema_name lets callers use one schema for the tool definition
        # (what Claude sees) and a different, more lenient schema for parsing the
        # response (e.g. evidence_extraction_raw for per-item validation).
        _response_schema = response_schema_name or schema_name
        _model = model_override or self.model
        request_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = self._client.messages.create(
                model=_model,
                max_tokens=max_tokens or self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                tools=[_tool_definition(operation, schema_name)],
                tool_choice={"type": "tool", "name": operation},
            )
            stop_reason = getattr(response, "stop_reason", None)
            output_tokens = getattr(getattr(response, "usage", None), "output_tokens", None)
            LOGGER.debug(
                "%s: stop_reason=%s output_tokens=%s max_tokens=%s",
                operation,
                stop_reason,
                output_tokens,
                max_tokens or self.max_tokens,
            )
            if stop_reason == "max_tokens":
                raise RuntimeError(
                    f"{operation}: response truncated (stop_reason=max_tokens, "
                    f"limit={max_tokens or self.max_tokens}). "
                    "The tool input is incomplete and would silently validate as empty. "
                    "Increase max_tokens for this operation."
                )
            tool_input = _response_tool_input(response)
            LOGGER.debug(
                "%s: tool_input present=%s raw_length=%s",
                operation,
                tool_input is not None,
                len(str(tool_input)) if tool_input is not None else 0,
            )
            if tool_input is not None:
                payload = _validate_payload(tool_input, _response_schema)
            else:
                text = _response_text(response)
                payload = parse_or_repair_json(
                    text,
                    _response_schema,
                    {
                        "operation": operation,
                        "expected_shape": _schema_description(_response_schema),
                        "repair": lambda repair_prompt: self._repair_json_text(
                            operation=operation,
                            prompt=repair_prompt,
                            max_tokens=max_tokens or self.max_tokens,
                        ),
                    },
                )
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=operation,
                    model_name=_model,
                    request_timestamp=request_timestamp,
                    success=True,
                    token_usage=_token_usage(response),
                )
            )
            return payload
        except Exception as exc:
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=operation,
                    model_name=_model,
                    request_timestamp=request_timestamp,
                    success=False,
                    error=str(exc),
                )
            )
            raise

    def _repair_json_text(self, *, operation: str, prompt: str, max_tokens: int) -> str:
        request_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=f"{operation}_repair",
                    model_name=self.model,
                    request_timestamp=request_timestamp,
                    success=True,
                    token_usage=_token_usage(response),
                )
            )
            return _response_text(response)
        except Exception as exc:
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=f"{operation}_repair",
                    model_name=self.model,
                    request_timestamp=request_timestamp,
                    success=False,
                    error=str(exc),
                )
            )
            raise


def create_research_plan(
    question: str,
    source_texts: Sequence[SourceDocument],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ResearchPlan:
    return ClaudeClient(model=model, api_key=api_key).create_research_plan(question, source_texts)


def extract_evidence(
    question: str,
    source_texts: Sequence[SourceDocument],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> list[EvidenceItem]:
    return ClaudeClient(model=model, api_key=api_key).extract_evidence(question, source_texts)


def synthesize_memo(
    question: str,
    evidence_items: Sequence[EvidenceItem],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ResearchMemo:
    return ClaudeClient(model=model, api_key=api_key).synthesize_memo(question, evidence_items)


def extract_evidence_from_chunks(
    question: str,
    chunks: Sequence[Chunk],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> list[EvidenceItem]:
    return ClaudeClient(model=model, api_key=api_key).extract_evidence_from_chunks(question, chunks)


def aggregate_call_traces(call_traces: Sequence[ClaudeCallTrace]) -> dict[str, Any]:
    if not call_traces:
        return {
            "model_name": None,
            "request_timestamp": None,
            "response_success": None,
            "token_usage": None,
            "errors": [],
        }

    token_usage: dict[str, int] = {}
    errors: list[str] = []
    for trace in call_traces:
        for key, value in trace.token_usage.items():
            token_usage[key] = token_usage.get(key, 0) + value
        if trace.error:
            errors.append(trace.error)

    return {
        "model_name": call_traces[-1].model_name,
        "request_timestamp": call_traces[0].request_timestamp,
        "response_success": all(trace.success for trace in call_traces),
        "token_usage": token_usage or None,
        "errors": errors,
    }


def _planning_prompt(
    question: str,
    profiles_context: list[dict],
    decision_model: dict | None = None,
    research_strategy: dict | None = None,
) -> str:
    """Build the PlannerAgent prompt for question classification and decomposition (J5.1 / J6.1a / J6.2)."""
    profile_lines = ""
    for p in profiles_context:
        name = p.get("name", "unknown")
        desc = p.get("description", "")
        topics = ", ".join(p.get("key_topics", []))
        profile_lines += f"\n- {name}: {desc}"
        if topics:
            profile_lines += f" (key topics: {topics})"

    # Decision Model context — injected when available (goal-driven runs)
    dm_section = ""
    if decision_model:
        dm_section = f"""
Decision Model (pre-derived from business goal — use this to ground your plan):
  Objective: {decision_model.get('objective', '')}
  Decision areas: {', '.join(decision_model.get('decision_areas', []))}
  Critical uncertainties: {', '.join(decision_model.get('critical_uncertainties', []))}
  Research questions: {'; '.join(decision_model.get('research_questions', []))}
  Evidence requirements: {', '.join(decision_model.get('evidence_requirements', []))}

Your subquestions and investigation areas should be aligned with the Decision Model above.
"""

    # Research Strategy context — injected when available (J6.2)
    rs_section = ""
    if research_strategy:
        rq_prios = research_strategy.get("research_question_priorities", [])
        rq_ordered = "; ".join(
            rqp.get("question", "") for rqp in sorted(rq_prios, key=lambda x: x.get("priority", 99))
        )
        coverage = ", ".join(
            f"{k}={v}" for k, v in list(research_strategy.get("coverage_targets", {}).items())[:5]
        )
        rs_section = f"""
Research Strategy (use this to prioritise subquestions and structure your plan):
  Priority questions (most important first): {rq_ordered}
  Required evidence: {', '.join(research_strategy.get('required_evidence', [])[:4])}
  Source priorities: {', '.join(research_strategy.get('source_priorities', [])[:4])}
  Coverage targets: {coverage}

Align your subquestions with the priority question order above.
"""

    return f"""You are a research planning agent. Analyze the question below and produce a structured research plan.

Question:
{question}

Domain profiles loaded:{profile_lines if profile_lines else " (none)"}
{dm_section}{rs_section}
Instructions:
1. Classify the research_type as exactly one of:
   - FACT_LOOKUP: asking for a specific fact, number, or definition
   - COMPARISON: comparing two or more entities, technologies, or options
   - EXPLANATION: asking why or how something works
   - RESEARCH: broad investigation requiring synthesis across multiple topics

2. Generate 3-7 focused subquestions that decompose the main question into
   answerable parts. Draw on the domain profiles to make subquestions specific.
   If a Decision Model is provided, map subquestions to the research_questions above.

3. Generate 4-8 investigation areas (short topic labels like "Power Requirements",
   "Deployment Timeline", "Economics") that structure the research.
   If a Decision Model is provided, align areas with the decision_areas above.

4. List which profile names informed this plan in profiles_used.

5. Write a brief reasoning (2-3 sentences) explaining your classification.

Return structured JSON only.
"""


def _problem_framing_prompt(goal: str, profiles_context: list[dict]) -> str:
    """Build the ProblemFramingAgent prompt for decision model generation (J6.1)."""
    profile_lines = ""
    for p in profiles_context:
        name = p.get("name", "unknown")
        desc = p.get("description", "")
        topics = ", ".join(p.get("key_topics", []))
        profile_lines += f"\n- {name}: {desc}"
        if topics:
            profile_lines += f" (key topics: {topics})"

    return f"""You are a strategic research planning agent. Transform the business goal below into a structured Decision Model that will guide a research pipeline.

Business Goal:
{goal}

Domain profiles available:{profile_lines if profile_lines else " (none)"}

Instructions:
1. Restate the goal as a precise research objective (1-2 sentences).

2. Identify 3-6 key decision areas — the dimensions that must be understood to act on this goal (e.g. "Market readiness", "Technical feasibility", "Regulatory landscape").

3. Identify 2-5 critical uncertainties — the unknowns that most affect the decision outcome.

4. Generate 3-6 specific, answerable research questions derived directly from the goal and decision areas. Draw on the domain profiles to make questions specific and actionable.

5. List 2-5 evidence requirements — the types of evidence needed to answer the research questions (e.g. "Benchmark performance data", "Vendor cost sheets", "Industry analyst reports").

Return structured JSON only.
"""


def _strategy_prompt(decision_model: dict, profiles_context: list[dict]) -> str:
    """Build the ResearchStrategyAgent prompt (J6.2)."""
    profile_lines = ""
    for p in profiles_context:
        name = p.get("name", "unknown")
        desc = p.get("description", "")
        topics = ", ".join(p.get("key_topics", []))
        profile_lines += f"\n- {name}: {desc}"
        if topics:
            profile_lines += f" (key topics: {topics})"

    dm_areas = "\n".join(f"  - {a}" for a in decision_model.get("decision_areas", []))
    dm_questions = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(decision_model.get("research_questions", [])))
    dm_uncertainties = "\n".join(f"  - {u}" for u in decision_model.get("critical_uncertainties", []))
    dm_evidence = "\n".join(f"  - {e}" for e in decision_model.get("evidence_requirements", []))

    return f"""You are a research strategy agent. Given a Decision Model and available domain profiles, produce an executable research strategy.

Decision Model:
  Objective: {decision_model.get("objective", "")}
  Decision Areas:
{dm_areas}
  Research Questions:
{dm_questions}
  Critical Uncertainties:
{dm_uncertainties}
  Evidence Requirements:
{dm_evidence}

Domain Profiles:{profile_lines if profile_lines else " (none)"}

Instructions:
1. Rank each profile by its relevance to this decision model (1 = most relevant). Include all available profiles.

2. Order the research questions by decision impact — most important first. Return a list of {{question, priority}} objects.

3. List the specific evidence items needed to satisfy the decision model's evidence requirements. Be concrete (e.g. "AI power demand forecasts 2024–2030" not just "forecasts").

4. List source types in priority order (e.g. "grid operator reports", "peer-reviewed studies", "vendor datasheets").

5. For each decision area and critical uncertainty, assign a coverage target: "strong", "moderate", or "light".

6. Write 2-3 sentences explaining the strategic choices.

Return structured JSON only.
"""


def _research_plan_prompt(question: str, source_texts: Sequence[SourceDocument]) -> str:
    return f"""Create a concise research plan for this local-source research question.

Question:
{question}

Available sources:
{_source_inventory(source_texts)}

Return JSON only with this shape:
{{
  "research_questions": ["..."],
  "key_topics": ["..."],
  "source_priorities": ["..."]
}}
"""


def _evidence_prompt(question: str, source_texts: Sequence[SourceDocument]) -> str:
    return f"""Extract source-grounded evidence for this question.

Question:
{question}

Rules:
- Use only the source text below.
- Each evidence_snippet must be copied or tightly paraphrased from one source.
- Use categories only from: architecture, power, cooling, networking, rack architecture, operations, other.
- Do not invent evidence IDs; evidence_id is assigned by the harness after extraction.
- Prefer 3-8 evidence items per source when useful evidence is available.
- Return JSON only.

CRITICAL — claim field rules (violations cause the item to be discarded):
- Each claim must summarise ONLY what its own source document states.
- Do NOT compare against other sources or reference claims from other documents.
- Do NOT use the words: contradicts, contradicting, inconsistent, conflicting, in contrast to.
- Do NOT write phrases like "Unlike other sources…", "This contradicts…", "This is inconsistent with…".
- A valid claim: "HALEU fuel is not commercially available from OECD member suppliers."
- An INVALID claim: "This contradicts claims of global HALEU availability." ← will be discarded.

JSON shape:
{{
  "evidence_items": [
    {{
      "claim": "...",
      "source_document": "filename.ext",
      "evidence_snippet": "...",
      "category": "architecture",
      "relevance": "...",
      "confidence": "high"
    }}
  ]
}}

Sources:
{_source_blocks(source_texts)}
"""


_SYNTHESIS_FIELDS = ("evidence_id", "claim", "source_document", "evidence_snippet", "category")


def _slim_evidence(item: EvidenceItem) -> dict[str, Any]:
    """Return only the fields Claude needs for synthesis (drops scoring noise)."""
    d = item.model_dump()
    return {k: d[k] for k in _SYNTHESIS_FIELDS}


def _memo_prompt(question: str, evidence_items: Sequence[EvidenceItem]) -> str:
    evidence_json = json.dumps([_slim_evidence(i) for i in evidence_items], indent=2)
    return f"""Synthesize a Markdown memo payload from the source-grounded evidence.

Question:
{question}

Rules:
- Use only the provided evidence items.
- Evidence IDs and source document names are assigned by the harness.
- Every entry in confirmed_facts, power_implications, cooling_implications, networking_implications, and rack_architecture_implications must end with exactly one citation in this format: [Source: filename.pdf, Evidence: E001].
- Use only source_document and evidence_id values present in the provided evidence.
- Do not invent source names or evidence IDs.
- Distinguish confirmed facts from inferences.
- Keep entries concise.
- Return JSON only, not Markdown.

JSON shape:
{{
  "executive_summary": "...",
  "confirmed_facts": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "inferences": ["..."],
  "power_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "cooling_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "networking_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "rack_architecture_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "open_questions": ["..."]
}}

Evidence:
{evidence_json}
"""


def _source_inventory(source_texts: Sequence[SourceDocument]) -> str:
    if not source_texts:
        return "No sources loaded."
    return "\n".join(
        f"- {source.path.name}: {source.char_count} extracted characters" for source in source_texts
    )


def _source_blocks(source_texts: Sequence[SourceDocument], *, max_chars_per_source: int = 12_000) -> str:
    if not source_texts:
        return "No sources loaded."
    blocks: list[str] = []
    for source in source_texts:
        blocks.append(
            "\n".join(
                [
                    f"Source document: {source.path.name}",
                    f"Path: {source.path}",
                    "Text:",
                    source.text[:max_chars_per_source],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _evidence_chunk_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    return f"""Extract source-grounded evidence for this question.

Question:
{question}

Rules:
- Use only the source text below.
- Each evidence_snippet must be copied or tightly paraphrased from one chunk.
- Use categories only from: architecture, power, cooling, networking, rack architecture, operations, other.
- Do not invent evidence IDs; evidence_id is assigned by the harness after extraction.
- Set source_chunk_id to the Chunk ID shown in the header for the chunk you drew evidence from.
- Prefer 3-8 evidence items per source when useful evidence is available.
- Return JSON only.

CRITICAL — claim field rules (violations cause the item to be discarded):
- Each claim must summarise ONLY what its own source chunk states.
- Do NOT compare against other chunks or reference claims from other documents.
- Do NOT use the words: contradicts, contradicting, inconsistent, conflicting, in contrast to.
- Do NOT write phrases like "Unlike other sources…", "This contradicts…", "This is inconsistent with…".
- A valid claim: "The BWRX-300 is designed for construction in 24–36 months using modular techniques."
- An INVALID claim: "This contradicts estimates of longer construction timelines." ← will be discarded.

JSON shape:
{{
  "evidence_items": [
    {{
      "claim": "...",
      "source_document": "filename.ext",
      "source_chunk_id": "filename_ext_C001",
      "evidence_snippet": "...",
      "category": "architecture",
      "relevance": "...",
      "confidence": "high"
    }}
  ]
}}

Chunks:
{_chunk_blocks(chunks)}
"""


def _chunk_blocks(chunks: Sequence[Chunk]) -> str:
    """Format pre-selected chunks for the evidence extraction prompt.

    Selection and budget enforcement are handled upstream by
    ``select_relevant_chunks``; this function formats whatever it receives.
    """
    if not chunks:
        return "No chunks available."
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    f"Chunk ID: {chunk.chunk_id}",
                    f"Document: {chunk.document_name}",
                    f"Chunk: {chunk.chunk_number}",
                    "Text:",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _response_tool_input(response: Any) -> Any | None:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use" and hasattr(block, "input"):
            return block.input
        if hasattr(block, "input") and getattr(block, "name", None):
            return block.input
    return None


def parse_or_repair_json(
    raw_response: str,
    schema_name: str,
    repair_prompt_context: dict[str, Any],
) -> dict[str, Any]:
    """Parse Claude JSON, repair once if needed, and validate with Pydantic."""

    try:
        return _validate_payload(_parse_json_text(raw_response), schema_name)
    except Exception as first_error:
        repair = repair_prompt_context.get("repair")
        if repair is None:
            raise ValueError(f"{schema_name} JSON parse failed: {first_error}") from first_error

        repair_prompt = _repair_prompt(raw_response, schema_name, repair_prompt_context)
        repaired_response = repair(repair_prompt)
        try:
            return _validate_payload(_parse_json_text(repaired_response), schema_name)
        except Exception as second_error:
            raise ValueError(
                f"{schema_name} JSON parse failed after repair: {second_error}"
            ) from second_error


def _parse_json_text(text: str) -> Any:
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(text.strip())


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []

    fence_matches = re_findall_json_fences(stripped)
    candidates.extend(fence_matches)

    if stripped:
        candidates.append(stripped)

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start >= 0 and end >= start:
            candidates.append(stripped[start : end + 1])

    deduped: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def re_findall_json_fences(text: str) -> list[str]:
    import re

    return [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    ]


def _validate_payload(payload: Any, schema_name: str) -> dict[str, Any]:
    if schema_name not in _SCHEMA_ADAPTERS:
        raise ValueError(f"Unknown schema: {schema_name}")
    validated = _SCHEMA_ADAPTERS[schema_name].validate_python(payload)
    if isinstance(validated, BaseModel):
        return validated.model_dump()
    return validated


def _repair_prompt(
    raw_response: str,
    schema_name: str,
    repair_prompt_context: dict[str, Any],
) -> str:
    expected_shape = repair_prompt_context.get("expected_shape", _schema_description(schema_name))
    operation = repair_prompt_context.get("operation", schema_name)
    return f"""The previous Claude response for {operation} was not valid JSON for schema {schema_name}.

Return valid JSON only. Do not include markdown fences, comments, or prose.

Expected schema:
{expected_shape}

Invalid response:
{raw_response}
"""


def _tool_definition(operation: str, schema_name: str) -> dict[str, Any]:
    return {
        "name": operation,
        "description": f"Return structured JSON for {operation}.",
        "input_schema": _SCHEMA_ADAPTERS[schema_name].json_schema(),
    }


def _schema_description(schema_name: str) -> str:
    if schema_name not in _SCHEMA_ADAPTERS:
        return "{}"
    return json.dumps(_SCHEMA_ADAPTERS[schema_name].json_schema(), indent=2)


def _token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    values: dict[str, int] = {}
    for attr in ("input_tokens", "output_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            values[attr] = value
    return values


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validation_error_message(exc: ValidationError) -> str:
    return "; ".join(error["msg"] for error in exc.errors())
