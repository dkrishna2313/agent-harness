"""PH5.1 — Deterministic Agent Fingerprint Validation.

Validates the two remaining YES-deterministic pipeline agents that require no
LLM calls:

  ResearchGapAgent   — pure heuristic scoring over coverage and gap fields
  IterationPlanAgent — pure priority-scoring heuristics over structured inputs

Both agents were classified YES-deterministic in AGENT_ARCHITECTURAL_CONTRACTS.md.
This file provides repeatable engineering evidence that the classification holds.

Test methodology (identical to PH4):
  - Freeze inputs as canonical Python literals (no file I/O, no network)
  - Run the agent N times from identical inputs
  - Compute SHA-256 over canonical JSON of the output fields used by the CLI
    fingerprint command
  - Assert all N fingerprints are identical
  - Assert the fingerprint matches the registered canonical value

Canonical engagement: ENG-002 (Go / No-Go, SMR investment assessment).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from functional_agents.context import AgentContext
from functional_agents.iteration_plan_agent import IterationPlanAgent
from functional_agents.research_gap_agent import ResearchGapAgent

# ---------------------------------------------------------------------------
# Canonical registered fingerprints
# These are the SHA-256 values validated across 5 independent runs and recorded
# as part of the PH5.1 validation run on 2026-07-16.
# ---------------------------------------------------------------------------

RESEARCH_GAP_FINGERPRINT = (
    "3ccbd120829fbbf4bbd18d7998fff1e68fdd1cf078eb2eefaa912dc9b47d98c8"
)

ITERATION_PLAN_FINGERPRINT = (
    "a4e28499c2592c88145f6b1563a7cafd80c5a4305492c90443754525505eab67"
)

# ---------------------------------------------------------------------------
# Frozen ENG-002 planner artifact
# Derived from MockClaudeClient.plan_research_question() seeded from the
# ENG-002 Go/No-Go decision model.  Committed as a literal so the test is
# fully self-contained (no file I/O).
# ---------------------------------------------------------------------------

_ENG002_PLANNER: dict = {
    "question": "Small Modular Reactor Technology Go / No-Go Assessment",
    "research_type": "RESEARCH",
    "subquestions": [
        "What is the current state of: Small Modular Reactor Technology Go / No-Go Assessment?",
        "What are the key technical and market constraints?",
        "What evidence exists on investment returns and risk factors?",
        "What are the strategic options and their trade-offs?",
    ],
    "investigation_areas": [
        "Market Landscape",
        "Technical Feasibility",
        "Risk Assessment",
        "Investment Criteria",
    ],
    "profiles_used": ["smr"],
    "reasoning": "Mock plan seeded from decision model.",
}

# Synthetic evidence derived deterministically from the frozen planner.
# All subquestions have NONE coverage; all investigation areas have zero
# evidence — identical to what the 'debug research-gap' CLI command produces.
_ENG002_SYNTHETIC_EVIDENCE: list[dict] = [
    {
        "coverage_by_subquestion": {
            sq: {"coverage": "NONE", "evidence_count": 0}
            for sq in _ENG002_PLANNER["subquestions"]
        },
        "evidence_by_area": {
            area: [] for area in _ENG002_PLANNER["investigation_areas"]
        },
        "evidence_items": [],
    }
]

# ---------------------------------------------------------------------------
# Frozen ENG-002 iteration-plan context
# Represents the state after a first pipeline pass on the ENG-002 engagement.
# All inputs are typed literals — no LLM calls.
# ---------------------------------------------------------------------------

_ENG002_EXEC_CONF: dict = {
    "overall_confidence": "Low",
    "decision_readiness": False,
    "board_recommendation": "Defer",
    "validation_priorities": ["Regulatory timeline", "Construction cost"],
    "critical_unknowns": ["GDA milestone dates", "Cost estimates"],
}

_ENG002_ASSUMPTIONS: list[dict] = [
    {
        "assumption_id": "ASM-001",
        "statement": "Regulatory GDA completes on schedule",
        "importance": "Critical",
        "confidence": "Low",
    },
    {
        "assumption_id": "ASM-002",
        "statement": "Construction costs within 20% of estimate",
        "importance": "Critical",
        "confidence": "Low",
    },
]

_ENG002_RECOMMENDATIONS: list[dict] = [
    {
        "recommendation_id": "REC-001",
        "title": "Commission independent cost study",
        "priority": "High",
    },
]

_ENG002_RISKS: list[dict] = [
    {"risk_id": "RSK-001", "statement": "Regulatory delay beyond commitment window"},
]

_ENG002_GAP_ANALYSIS: dict = {
    "overall_research_health": "POOR",
    "weak_questions": ["Regulatory timelines", "Construction cost data"],
    "missing_investigation_areas": ["Market Landscape", "Technical Feasibility"],
    "decision_support_gaps": ["DSG-001"],
    "recommended_followups": ["Obtain GDA schedule from ONR"],
}


# ---------------------------------------------------------------------------
# Fingerprint helpers (mirror the CLI debug command implementations)
# ---------------------------------------------------------------------------

def _rga_fingerprint(ctx: AgentContext) -> str:
    """Compute the canonical SHA-256 fingerprint for ResearchGapAgent output."""
    analysis = ctx.research_gap_analysis or {}
    fp_data = {
        "overall_research_health": analysis.get("overall_research_health", ""),
        "weak_questions": analysis.get("weak_questions", []),
        "missing_investigation_areas": analysis.get("missing_investigation_areas", []),
        "decision_support_gaps": analysis.get("decision_support_gaps", []),
        "recommended_followups": analysis.get("recommended_followups", []),
        "assumption_heavy_topics": analysis.get("assumption_heavy_topics", []),
    }
    canonical = json.dumps(fp_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _iplan_fingerprint(ctx: AgentContext) -> str:
    """Compute the canonical SHA-256 fingerprint for IterationPlanAgent output."""
    plan = ctx.iteration_plan or {}
    fp_data = {
        "priority_research_tasks": plan.get("priority_research_tasks", []),
        "plan_summary": plan.get("plan_summary", ""),
    }
    canonical = json.dumps(fp_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen context factories
# ---------------------------------------------------------------------------

def _rga_context() -> AgentContext:
    return AgentContext(
        question=_ENG002_PLANNER["question"],
        goal="",
        profiles=[],
        execution_profile="",
        research_object={},
        run_id="ph51-rga",
        plan=_ENG002_PLANNER,
        evidence_notes=_ENG002_SYNTHETIC_EVIDENCE,
        hypotheses=[],
        validated_contradictions=[],
    )


def _iplan_context() -> AgentContext:
    return AgentContext(
        question=_ENG002_PLANNER["question"],
        goal="Go/No-Go investment decision",
        profiles=["smr"],
        execution_profile="default",
        research_object={},
        run_id="ph51-iplan",
        executive_confidence=_ENG002_EXEC_CONF,
        assumptions=_ENG002_ASSUMPTIONS,
        recommendations=_ENG002_RECOMMENDATIONS,
        risks=_ENG002_RISKS,
        research_gap_analysis=_ENG002_GAP_ANALYSIS,
    )


# ===========================================================================
# ResearchGapAgent — Fingerprint Validation
# ===========================================================================

class TestResearchGapAgentFingerprint:
    """PH5.1-RGA — ResearchGapAgent determinism verified via SHA-256 fingerprints."""

    def test_rga_single_run_produces_analysis(self):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        assert result.context.research_gap_analysis is not None

    def test_rga_single_run_fingerprint_matches_canonical(self):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        fp = _rga_fingerprint(result.context)
        assert fp == RESEARCH_GAP_FINGERPRINT, (
            f"ResearchGapAgent fingerprint drifted.\n"
            f"  Expected : {RESEARCH_GAP_FINGERPRINT}\n"
            f"  Got      : {fp}"
        )

    @pytest.mark.parametrize("run_index", range(1, 6))
    def test_rga_run_n_fingerprint_matches_canonical(self, run_index):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        fp = _rga_fingerprint(result.context)
        assert fp == RESEARCH_GAP_FINGERPRINT, (
            f"ResearchGapAgent fingerprint drifted on run {run_index}.\n"
            f"  Expected : {RESEARCH_GAP_FINGERPRINT}\n"
            f"  Got      : {fp}"
        )

    def test_rga_five_runs_all_identical(self):
        fingerprints = [
            _rga_fingerprint(ResearchGapAgent().run(_rga_context()).context)
            for _ in range(5)
        ]
        assert len(set(fingerprints)) == 1, (
            f"ResearchGapAgent produced non-identical fingerprints across 5 runs: "
            f"{fingerprints}"
        )

    def test_rga_overall_health_is_poor_for_zero_coverage(self):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        assert result.context.research_gap_analysis["overall_research_health"] == "POOR"

    def test_rga_all_subquestions_flagged_weak(self):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        weak = result.context.research_gap_analysis.get("weak_questions", [])
        # All 4 subquestions should be flagged (all NONE coverage)
        assert len(weak) == len(_ENG002_PLANNER["subquestions"])

    def test_rga_all_investigation_areas_missing(self):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        missing = result.context.research_gap_analysis.get("missing_investigation_areas", [])
        # All 4 areas should be missing (no evidence)
        assert len(missing) == len(_ENG002_PLANNER["investigation_areas"])

    def test_rga_recommended_followups_non_empty(self):
        ctx = _rga_context()
        result = ResearchGapAgent().run(ctx)
        followups = result.context.research_gap_analysis.get("recommended_followups", [])
        assert len(followups) > 0

    def test_rga_different_planner_produces_different_fingerprint(self):
        """Input-sensitivity: a different planner must produce a different fingerprint."""
        planner_b = {
            **_ENG002_PLANNER,
            "subquestions": ["Is the grid ready for AI data center load growth?"],
            "investigation_areas": ["Grid Capacity"],
        }
        synthetic_b = [
            {
                "coverage_by_subquestion": {
                    sq: {"coverage": "NONE", "evidence_count": 0}
                    for sq in planner_b["subquestions"]
                },
                "evidence_by_area": {area: [] for area in planner_b["investigation_areas"]},
                "evidence_items": [],
            }
        ]
        ctx_b = AgentContext(
            question=planner_b["question"],
            goal="",
            profiles=[],
            execution_profile="",
            research_object={},
            run_id="ph51-rga-b",
            plan=planner_b,
            evidence_notes=synthetic_b,
            hypotheses=[],
            validated_contradictions=[],
        )
        fp_b = _rga_fingerprint(ResearchGapAgent().run(ctx_b).context)
        assert fp_b != RESEARCH_GAP_FINGERPRINT, (
            "Input-sensitivity check failed: different planner produced the same fingerprint."
        )


# ===========================================================================
# IterationPlanAgent — Fingerprint Validation
# ===========================================================================

class TestIterationPlanAgentFingerprint:
    """PH5.1-IPLAN — IterationPlanAgent determinism verified via SHA-256 fingerprints."""

    def test_iplan_single_run_produces_plan(self):
        ctx = _iplan_context()
        result = IterationPlanAgent().run(ctx)
        assert result.context.iteration_plan is not None

    def test_iplan_single_run_fingerprint_matches_canonical(self):
        ctx = _iplan_context()
        result = IterationPlanAgent().run(ctx)
        fp = _iplan_fingerprint(result.context)
        assert fp == ITERATION_PLAN_FINGERPRINT, (
            f"IterationPlanAgent fingerprint drifted.\n"
            f"  Expected : {ITERATION_PLAN_FINGERPRINT}\n"
            f"  Got      : {fp}"
        )

    @pytest.mark.parametrize("run_index", range(1, 6))
    def test_iplan_run_n_fingerprint_matches_canonical(self, run_index):
        ctx = _iplan_context()
        result = IterationPlanAgent().run(ctx)
        fp = _iplan_fingerprint(result.context)
        assert fp == ITERATION_PLAN_FINGERPRINT, (
            f"IterationPlanAgent fingerprint drifted on run {run_index}.\n"
            f"  Expected : {ITERATION_PLAN_FINGERPRINT}\n"
            f"  Got      : {fp}"
        )

    def test_iplan_five_runs_all_identical(self):
        fingerprints = [
            _iplan_fingerprint(IterationPlanAgent().run(_iplan_context()).context)
            for _ in range(5)
        ]
        assert len(set(fingerprints)) == 1, (
            f"IterationPlanAgent produced non-identical fingerprints across 5 runs: "
            f"{fingerprints}"
        )

    def test_iplan_tasks_non_empty(self):
        ctx = _iplan_context()
        result = IterationPlanAgent().run(ctx)
        tasks = result.context.iteration_plan.get("priority_research_tasks", [])
        assert len(tasks) > 0

    def test_iplan_different_inputs_produce_different_fingerprint(self):
        """Input-sensitivity: changing assumptions must shift the fingerprint."""
        ctx_b = _iplan_context()
        ctx_b.assumptions = [
            {
                "assumption_id": "ASM-X",
                "statement": "Grid connection feasible within 36 months",
                "importance": "Critical",
                "confidence": "High",  # different from Low in ENG-002
            }
        ]
        fp_b = _iplan_fingerprint(IterationPlanAgent().run(ctx_b).context)
        assert fp_b != ITERATION_PLAN_FINGERPRINT, (
            "Input-sensitivity check failed: different assumptions produced the same fingerprint."
        )
