"""PH5.x — Executive Language Hotfix: identifier-leakage regression tests.

Asserts that no internal graph identifier (A-*, RSK-*, OPP-*, REC-*, OPT-*)
appears in any executive-facing prose field of the generated report.

Internal identifiers remain valid in the underlying structured model (linkage
arrays, traces, Pydantic objects); these tests only cover the text that reaches
the executive reader.

Covered sections
----------------
- Executive Summary
- Strategic Recommendation (recommended option prose)
- Recommendation Rationale / Why This Option Wins
- Decision Readiness / Confidence Narrative
- Critical Unknowns
- Validation Priorities
- Conditional Assessment (if assumptions hold/fail)
- Sensitivity Analysis
- Key Uncertainties
- Risk narrative (risk_story / _compose_risk_story)
- Decision story (_compose_decision_story)

Structured appendices with IDs are explicitly excluded from these assertions.
"""

from __future__ import annotations

import re

import pytest

from functional_agents.context import AgentContext
from functional_agents.narrative.composer import ExecutiveNarrativeComposer
from functional_agents.narrative.builder import ExecutiveNarrativeBuilder
from functional_agents.narrative.executive_narrative import ExecutiveNarrative
from functional_agents.report_agent import (
    _build_j7_executive_report,
    _clean_internal_language,
)

# ---------------------------------------------------------------------------
# Identifier pattern — the thing that must never appear in executive prose
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"\b(?:A|RSK|OPP|REC|OPT|EC|DA|SC)-[A-Z0-9]+\b")


def _assert_no_ids(text: str, label: str) -> None:
    """Fail with a descriptive message if an internal identifier is found."""
    found = _ID_RE.findall(text)
    assert not found, (
        f"{label} contains internal identifier(s) {found!r}.\n"
        f"Offending text: {text!r}"
    )


# ---------------------------------------------------------------------------
# Fixture — context with IDs embedded in every prose field the LLM can produce
# ---------------------------------------------------------------------------

def _make_id_heavy_context() -> AgentContext:
    """Context whose LLM-produced prose fields contain internal IDs everywhere.

    This is the worst-case mock: assumption/risk/opportunity IDs appear in
    sensitivity_analysis, rationale, critical_unknowns, and
    confidence_if_assumptions_fail exactly as J7.6/J7.7 mock code produces them.
    """
    return AgentContext(
        question="Should we invest in SMR technology?",
        profiles=["smr"],
        execution_profile="smr",
        assumptions=[
            {
                "assumption_id": "A-001",
                "statement": "Grid connection secured within 24 months",
                "importance": "Critical",
                "confidence": "Medium",
                "evidence_support": "Moderate",
            },
            {
                "assumption_id": "A-002",
                "statement": "HALEU fuel supply available by 2028",
                "importance": "Critical",
                "confidence": "Low",
                "evidence_support": "Weak",
            },
        ],
        risks=[
            {
                "risk_id": "RSK-001",
                "statement": "Grid interconnection delays exceed timeline",
                "severity": "High",
                "likelihood": "Medium",
                "mitigation_notes": "Secure queue position early",
                "related_assumption_ids": ["A-001"],
                "affected_recommendation_ids": ["REC-001"],
            },
            {
                "risk_id": "RSK-002",
                "statement": "Fuel supply chain disruption",
                "severity": "Medium",
                "likelihood": "Low",
                "mitigation_notes": "Diversify supplier base",
                "related_assumption_ids": ["A-002"],
                "affected_recommendation_ids": ["REC-002"],
            },
        ],
        opportunities=[
            {
                "opportunity_id": "OPP-001",
                "statement": "Early mover advantage in SMR market",
                "impact": "High",
                "related_assumption_ids": ["A-001"],
            },
        ],
        recommendations=[
            {
                "recommendation_id": "REC-001",
                "title": "File grid interconnection application immediately",
                "time_horizon": "near_term",
                "priority": "high",
                "summary": "Start the grid process now to maintain schedule.",
                "supported_assumption_ids": ["A-001"],
            },
            {
                "recommendation_id": "REC-002",
                "title": "Secure HALEU fuel supply agreements",
                "time_horizon": "medium_term",
                "priority": "high",
                "summary": "Engage fuel suppliers to lock in supply.",
                "supported_assumption_ids": ["A-002"],
            },
        ],
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Phased Deployment",
                "description": "Deploy in stages to preserve optionality.",
                "strategic_objective": "Optimise risk-adjusted returns",
                "estimated_time_horizon": "Medium-term",
                "capital_intensity": "Medium",
                "confidence": "High",
                "advantages": ["Lower downside risk", "Preserves flexibility"],
                "disadvantages": ["Slower to full scale"],
                "recommended": True,
                # Rationale as the mock produces it — with embedded option_id
                "rationale": (
                    "Preferred over OPT-B because it manages RSK-001 while still "
                    "capturing OPP-001. Unlike OPT-C, it commits meaningfully."
                ),
                "supporting_assumption_ids": ["A-001", "A-002"],
                "associated_risk_ids": ["RSK-001"],
                "associated_opportunity_ids": ["OPP-001"],
                "supporting_recommendation_ids": ["REC-001"],
            },
            {
                "option_id": "OPT-B",
                "title": "Aggressive Build",
                "description": "Full deployment immediately.",
                "strategic_objective": "First-mover market leadership",
                "estimated_time_horizon": "Near-term",
                "capital_intensity": "High",
                "confidence": "Medium",
                "advantages": ["Speed to market"],
                "disadvantages": ["High capital at risk"],
                "recommended": False,
                "rationale": (
                    "Higher reward but requires A-001 and A-002 to hold simultaneously."
                ),
                "supporting_assumption_ids": ["A-001"],
                "associated_risk_ids": ["RSK-001", "RSK-002"],
                "associated_opportunity_ids": ["OPP-001"],
                "supporting_recommendation_ids": ["REC-001", "REC-002"],
            },
        ],
        preferred_option={"option_id": "OPT-A", "title": "Phased Deployment"},
        decision_analysis={
            "recommended_option_id": "OPT-A",
            "executive_summary": (
                "The preferred option for SMR investment balances risk-adjusted returns "
                "with capital discipline. It outperforms alternatives on strategic fit."
            ),
            "comparison_dimensions": [
                "Strategic Fit", "Implementation Risk", "Capital Requirement",
            ],
            "option_rankings": ["OPT-A", "OPT-B"],
            "rationale": (
                "The recommended option (OPT-A) wins over alternatives because it achieves "
                "the strongest risk-adjusted outcome, particularly on strategic fit and "
                "opportunity capture, while managing RSK-001 and RSK-002."
            ),
            "key_tradeoffs": [
                "Higher capital commitment → lower execution risk",
                "Longer implementation horizon → greater strategic flexibility",
            ],
            "key_uncertainties": [
                "If A-001 fails the risk-return balance shifts toward conservative option",
                "If A-002 proves optimistic OPT-B becomes relatively more attractive",
            ],
            # Sensitivity as the mock produces it — with embedded assumption IDs
            "sensitivity_analysis": (
                "If A-001 fails, the recommended option would no longer dominate — "
                "the conservative option would become preferred. "
                "If A-002 proves optimistic, OPT-B's time-to-value advantage amplifies."
            ),
            "confidence": "Medium",
            "confidence_summary": (
                "Medium confidence. Analysis relies on A-001 and A-002 which have "
                "moderate evidence support."
            ),
        },
        strategic_synthesis={
            "executive_summary": (
                "Cross-domain synthesis favors phased deployment with capital discipline."
            ),
        },
        executive_confidence={
            "overall_confidence": "Medium",
            "decision_readiness": "Needs Additional Validation",
            "board_recommendation": "Proceed with Conditions",
            "confidence_rationale": (
                "Overall confidence is Medium. The decision analysis identified 2 "
                "strategic options with 2 Critical assumptions underpinning the "
                "recommended path."
            ),
            "confidence_drivers": [
                "Decision analysis confidence: Medium",
                "2 strategic options explicitly evaluated",
            ],
            "confidence_limiters": [
                "1 assumption(s) have Weak evidence support",
                "1 High-severity risk(s) require mitigation",
            ],
            # Critical unknowns as the mock produces them — raw IDs
            "critical_unknowns": [
                "Resolution of A-001",
                "Resolution of A-002",
            ],
            # Validation priorities as the mock produces them — with IDs
            "validation_priorities": [
                "Validate: Grid connection secured within 24 months",
                "Validate: HALEU fuel supply available by 2028",
                "Mitigate 1 High-severity risk(s) before commitment",
            ],
            # Conditional as the mock produces — with raw IDs
            "confidence_if_assumptions_hold": (
                "High confidence — if all 2 Critical assumption(s) (A-001, A-002) hold, "
                "the recommended option achieves its strategic objectives."
            ),
            "confidence_if_assumptions_fail": (
                "Low confidence — if Critical assumption(s) (A-001, A-002) fail, "
                "the strategy's risk-return profile shifts materially."
            ),
            "decision_horizon": "Q3 2026",
        },
        recommendation_portfolio={
            "near_term": ["REC-001"],
            "medium_term": ["REC-002"],
            "long_term": [],
        },
        research_object={
            "evidence_summary": {"total_evidence_items": 10, "citation_count": 6},
            "profiles": ["smr"],
        },
    )


# ---------------------------------------------------------------------------
# Helper: build the full report text
# ---------------------------------------------------------------------------

def _build_report(ctx: AgentContext) -> str:
    return _build_j7_executive_report(ctx)


# ---------------------------------------------------------------------------
# Unit tests — _clean_internal_language()
# ---------------------------------------------------------------------------

class TestCleanInternalLanguage:
    """Unit-level tests for the identifier-stripping regex in _clean_internal_language."""

    @pytest.mark.parametrize("raw, expected", [
        # Assumption IDs
        ("If A-001 fails, the option no longer dominates.", "If fails, the option no longer dominates."),
        ("Resolution of A-001", "Resolution of"),
        # Risk IDs
        ("The option manages RSK-003 through early procurement.", "The option manages through early procurement."),
        # Opportunity IDs
        ("It captures OPP-004 effectively.", "It captures effectively."),
        # Recommendation IDs
        ("Supported by REC-002.", "Supported by ."),
        # Option IDs — letter suffix; empty parens cleaned by artefact cleanup
        ("The recommended option (OPT-B) wins over alternatives.", "The recommended option wins over alternatives."),
        # Multiple IDs in one string
        ("A-001 and RSK-003 affect OPP-004.", "and affect ."),
        # Parenthetical ID list; (A-001, A-002) → (, ) → cleaned
        ("Critical assumptions (A-001, A-002) must hold.", "Critical assumptions must hold."),
        # No IDs — string unchanged (beyond existing phrase subs)
        ("The option delivers strong risk-adjusted returns.", "The option delivers strong risk-adjusted returns."),
        # Empty string — returned as-is
        ("", ""),
    ])
    def test_strips_identifiers(self, raw: str, expected: str) -> None:
        result = _clean_internal_language(raw)
        _assert_no_ids(result, f"_clean_internal_language({raw!r})")
        assert result == expected, f"Got: {result!r}"

    def test_no_ids_passthrough(self) -> None:
        text = "The preferred option delivers strong risk-adjusted returns with capital discipline."
        assert _clean_internal_language(text) == text

    def test_empty_parens_cleaned(self) -> None:
        raw = "Low confidence — if Critical assumption(s) (A-001, A-002) fail."
        result = _clean_internal_language(raw)
        _assert_no_ids(result, "empty_parens_test")
        assert "()" not in result


# ---------------------------------------------------------------------------
# Integration tests — full report text
# ---------------------------------------------------------------------------

class TestReportNoIdentifiers:
    """Integration tests: the full _build_j7_executive_report output contains no IDs."""

    @pytest.fixture
    def report(self) -> str:
        ctx = _make_id_heavy_context()
        return _build_report(ctx)

    # --- Broad sweep ---

    def _executive_sections(self, report: str) -> str:
        """Return only executive-narrative sections, excluding structured appendices.

        Appendices are allowed to carry structured IDs (option_id in headings,
        assumption_id in tables, etc.). The prose narrative must be ID-free.
        """
        # Split at first appendix heading and keep only the front matter.
        parts = re.split(r"#+\s+Appendix", report, maxsplit=1)
        return parts[0]

    def test_executive_sections_contain_no_ids(self, report: str) -> None:
        """Master guard: no internal ID appears in any executive-facing section."""
        executive_text = self._executive_sections(report)
        _assert_no_ids(executive_text, "executive sections (pre-appendix)")

    # --- Section-level assertions ---

    def test_executive_summary_no_ids(self, report: str) -> None:
        m = re.search(r"(?:Executive Summary|## 1\.)(.*?)(?=\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Executive Summary")

    def test_strategic_recommendation_no_ids(self, report: str) -> None:
        m = re.search(r"(?:Strategic Recommendation|## 2\.|## 3\.)(.*?)(?=\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Strategic Recommendation")

    def test_recommendation_rationale_no_ids(self, report: str) -> None:
        m = re.search(r"(?:Why This Option|Rationale|## 4\.)(.*?)(?=\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Recommendation Rationale / Why This Option")

    def test_confidence_narrative_no_ids(self, report: str) -> None:
        m = re.search(r"(?:Confidence|Decision Readiness|## 5\.)(.*?)(?=\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Confidence Narrative / Decision Readiness")

    def test_critical_unknowns_no_ids(self, report: str) -> None:
        m = re.search(r"\*\*Critical Unknowns:\*\*(.*?)(?=\n\*\*|\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Critical Unknowns")

    def test_validation_priorities_no_ids(self, report: str) -> None:
        m = re.search(r"\*\*Validation Priorities:\*\*(.*?)(?=\n\*\*|\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Validation Priorities")

    def test_conditional_assessment_no_ids(self, report: str) -> None:
        m = re.search(r"\*\*Conditional Assessment:\*\*(.*?)(?=\n\*\*|\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Conditional Assessment")

    def test_sensitivity_analysis_no_ids(self, report: str) -> None:
        m = re.search(r"\*\*Sensitivity Analysis:\*\*(.*?)(?=\n\*\*|\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Sensitivity Analysis")

    def test_key_uncertainties_no_ids(self, report: str) -> None:
        m = re.search(r"\*\*Key Uncertainties:\*\*(.*?)(?=\n\*\*|\n#|\Z)", report, re.S)
        if m:
            _assert_no_ids(m.group(1), "Key Uncertainties")


# ---------------------------------------------------------------------------
# Composer tests — story fields
# ---------------------------------------------------------------------------

class TestComposerNoIdentifiers:
    """decision_story and risk_story must not contain internal identifiers."""

    def _build_narrative(self) -> ExecutiveNarrative:
        ctx = _make_id_heavy_context()
        narrative = ExecutiveNarrativeBuilder().build(ctx)
        return ExecutiveNarrativeComposer().compose(narrative)

    def test_decision_story_no_ids(self) -> None:
        narrative = self._build_narrative()
        _assert_no_ids(narrative.decision_story, "decision_story")

    def test_risk_story_no_ids(self) -> None:
        narrative = self._build_narrative()
        _assert_no_ids(narrative.risk_story, "risk_story")

    def test_confidence_story_no_ids(self) -> None:
        narrative = self._build_narrative()
        _assert_no_ids(narrative.confidence_story, "confidence_story")

    def test_decision_story_contains_option_title_not_id(self) -> None:
        """decision_story must reference the option by title, not by option_id."""
        narrative = self._build_narrative()
        story = narrative.decision_story
        assert "Phased Deployment" in story, "Expected option title in decision_story"
        assert "OPT-A" not in story, "option_id must not appear in decision_story"

    def test_risk_story_uses_statement_not_id(self) -> None:
        """risk_story must lead with the risk statement, not the risk_id."""
        narrative = self._build_narrative()
        story = narrative.risk_story
        assert "RSK-001" not in story, "RSK-001 must not appear in risk_story"
        assert "RSK-002" not in story, "RSK-002 must not appear in risk_story"
        # The risk statement should still be present
        assert "Grid interconnection delays" in story or "interconnection" in story


# ---------------------------------------------------------------------------
# Structured model preservation — IDs still in graph
# ---------------------------------------------------------------------------

class TestStructuredModelPreservesIds:
    """Traceability: internal IDs must remain in the structured model."""

    def test_assumptions_retain_ids(self) -> None:
        ctx = _make_id_heavy_context()
        assert any(a["assumption_id"] == "A-001" for a in ctx.assumptions)
        assert any(a["assumption_id"] == "A-002" for a in ctx.assumptions)

    def test_risks_retain_ids(self) -> None:
        ctx = _make_id_heavy_context()
        assert any(r["risk_id"] == "RSK-001" for r in ctx.risks)
        assert any(r["risk_id"] == "RSK-002" for r in ctx.risks)

    def test_options_retain_ids(self) -> None:
        ctx = _make_id_heavy_context()
        assert any(o["option_id"] == "OPT-A" for o in ctx.strategic_options)
        assert any(o["option_id"] == "OPT-B" for o in ctx.strategic_options)

    def test_linkage_arrays_retain_ids(self) -> None:
        ctx = _make_id_heavy_context()
        opt_a = next(o for o in ctx.strategic_options if o["option_id"] == "OPT-A")
        assert "A-001" in opt_a["supporting_assumption_ids"]
        assert "RSK-001" in opt_a["associated_risk_ids"]
        assert "OPP-001" in opt_a["associated_opportunity_ids"]
        assert "REC-001" in opt_a["supporting_recommendation_ids"]

    def test_narrative_structured_fields_retain_option_id(self) -> None:
        """recommended_option dict retains option_id for downstream consumers."""
        ctx = _make_id_heavy_context()
        _build_j7_executive_report(ctx)
        narrative = ExecutiveNarrative.from_dict(ctx.executive_narrative)
        assert narrative.recommended_option.get("option_id") == "OPT-A"

    def test_narrative_structured_fields_retain_risk_ids(self) -> None:
        """key_risks dicts retain risk_id for downstream consumers."""
        ctx = _make_id_heavy_context()
        _build_j7_executive_report(ctx)
        narrative = ExecutiveNarrative.from_dict(ctx.executive_narrative)
        if narrative.key_risks:
            assert any(r.get("risk_id") == "RSK-001" for r in narrative.key_risks)
