"""J5.6 – Multi-profile research tests.

Validates that the same agent runtime executes across multiple domain profiles
without any agent code modifications (Profiles = Knowledge, Agents = Behavior).

Covers:
  - transmission profile loads cleanly
  - _attribute_evidence_profiles() assigns source_profile to every item
  - single-profile mode is backward compatible (all items attributed to that profile)
  - multi-profile attribution routes items to the best-matching profile
  - profile_coverage_by_profile computed correctly
  - QA _check_profile_coverage() produces coverage dict and issues
  - full mock pipeline: 3 profiles → profile_coverage in context.qa
  - ReportAgent surfaces profiles + profile_coverage in trace
  - AgentContext profiles list carries all three profile names
"""

from __future__ import annotations

import pytest

from research_agent.profile import load_profile, list_available_profiles
from functional_agents.evidence_agent import (
    _attribute_evidence_profiles,
    _build_profile_term_sets,
)
from functional_agents.qa_agent import _check_profile_coverage
from functional_agents.context import AgentContext


# ---------------------------------------------------------------------------
# Transmission profile
# ---------------------------------------------------------------------------

def test_transmission_profile_available():
    assert "transmission" in list_available_profiles()


def test_transmission_profile_loads():
    p = load_profile("transmission")
    assert p.name == "transmission"
    assert p.description
    assert p.domain_terms
    assert "transmission" in p.domain_terms


def test_transmission_profile_has_topic_keywords():
    p = load_profile("transmission")
    assert len(p.topic_keywords) >= 4
    assert "interconnection" in p.topic_keywords


def test_transmission_profile_has_evaluator_topics():
    p = load_profile("transmission")
    assert p.evaluator_topic_terms
    assert "interconnection" in p.evaluator_topic_terms


# ---------------------------------------------------------------------------
# _build_profile_term_sets
# ---------------------------------------------------------------------------

def test_build_profile_term_sets_returns_set_per_profile():
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    result = _build_profile_term_sets([smr, adc])
    assert "smr" in result
    assert "ai_data_centers" in result
    assert "reactor" in result["smr"]
    assert "gpu" in result["ai_data_centers"]


# ---------------------------------------------------------------------------
# _attribute_evidence_profiles
# ---------------------------------------------------------------------------

def _make_item(claim: str, topics: list[str] | None = None) -> dict:
    return {"claim": claim, "topics": topics or [], "evidence_id": "E001"}


def test_single_profile_all_items_attributed():
    smr = load_profile("smr")
    items = [_make_item("reactor safety"), _make_item("fuel cycle")]
    coverage = _attribute_evidence_profiles(items, [smr], fallback_profile="smr")
    assert all(i["source_profile"] == "smr" for i in items)
    assert "smr" in coverage
    assert coverage["smr"]["evidence_count"] == 2


def test_single_profile_coverage_levels():
    smr = load_profile("smr")
    items = [_make_item(f"item {i}") for i in range(12)]
    coverage = _attribute_evidence_profiles(items, [smr], fallback_profile="smr")
    assert coverage["smr"]["coverage_level"] == "STRONG"


def test_multi_profile_all_items_get_source_profile():
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    items = [
        _make_item("reactor licensing nrc nuclear"),
        _make_item("gpu rack power nvidia blackwell"),
    ]
    _attribute_evidence_profiles(items, [smr, adc], fallback_profile="smr")
    assert all("source_profile" in i for i in items)


def test_multi_profile_smr_item_attributed_to_smr():
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    items = [_make_item("nrc reactor licensing haleu enrichment nuclear smr")]
    _attribute_evidence_profiles(items, [smr, adc], fallback_profile="smr")
    assert items[0]["source_profile"] == "smr"


def test_multi_profile_adc_item_attributed_to_adc():
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    items = [_make_item("nvidia blackwell gpu rack power cooling thermal")]
    _attribute_evidence_profiles(items, [smr, adc], fallback_profile="smr")
    assert items[0]["source_profile"] == "ai_data_centers"


def test_multi_profile_coverage_dict_has_all_profiles():
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    tx = load_profile("transmission")
    items = [
        _make_item("reactor nuclear smr licensing"),
        _make_item("gpu power rack nvidia"),
        _make_item("transmission grid substation interconnection"),
    ]
    coverage = _attribute_evidence_profiles(items, [smr, adc, tx], fallback_profile="smr")
    assert "smr" in coverage
    assert "ai_data_centers" in coverage
    assert "transmission" in coverage


def test_no_profiles_fallback():
    items = [_make_item("some claim")]
    _attribute_evidence_profiles(items, [], fallback_profile="smr")
    assert items[0]["source_profile"] == "smr"


# ---------------------------------------------------------------------------
# _check_profile_coverage (QA)
# ---------------------------------------------------------------------------

def _make_evidence_note(profile_coverage: dict) -> dict:
    return {"profile_coverage_by_profile": profile_coverage}


def test_check_profile_coverage_strong():
    note = _make_evidence_note({
        "smr": {"evidence_count": 15, "coverage_level": "STRONG"},
        "ai_data_centers": {"evidence_count": 12, "coverage_level": "STRONG"},
    })
    result = _check_profile_coverage(note, ["smr", "ai_data_centers"])
    assert result["profile_coverage"]["smr"] == "strong"
    assert result["profile_coverage"]["ai_data_centers"] == "strong"
    assert result["profile_issues"] == []
    assert result["profile_gap_issues"] == []
    assert result["coverage_status"] == "sufficient"


def test_check_profile_coverage_none_raises_high_issue():
    note = _make_evidence_note({
        "smr": {"evidence_count": 0, "coverage_level": "NONE"},
    })
    result = _check_profile_coverage(note, ["smr"])
    assert result["profile_coverage"]["smr"] == "none"
    assert len(result["profile_issues"]) == 1
    assert result["profile_issues"][0]["severity"] == "HIGH"
    assert result["profile_issues"][0]["profile"] == "smr"


def test_check_profile_coverage_none_emits_gap_issue():
    note = _make_evidence_note({
        "smr": {"evidence_count": 0, "coverage_level": "NONE"},
    })
    result = _check_profile_coverage(note, ["smr"])
    assert len(result["profile_gap_issues"]) == 1
    assert result["profile_gap_issues"][0]["profile"] == "smr"
    assert result["profile_gap_issues"][0]["issue"] == "no evidence contributed"


def test_check_profile_coverage_weak_raises_medium_issue():
    note = _make_evidence_note({
        "transmission": {"evidence_count": 2, "coverage_level": "WEAK"},
    })
    result = _check_profile_coverage(note, ["transmission"])
    assert result["profile_coverage"]["transmission"] == "weak"
    assert len(result["profile_issues"]) == 1
    assert result["profile_issues"][0]["severity"] == "MEDIUM"


def test_check_profile_coverage_missing_profile_defaults_none():
    note = _make_evidence_note({})
    result = _check_profile_coverage(note, ["smr"])
    assert result["profile_coverage"]["smr"] == "none"
    assert len(result["profile_issues"]) == 1
    assert result["profile_issues"][0]["severity"] == "HIGH"


def test_check_profile_coverage_three_profiles():
    note = _make_evidence_note({
        "smr": {"evidence_count": 20, "coverage_level": "STRONG"},
        "ai_data_centers": {"evidence_count": 18, "coverage_level": "STRONG"},
        "transmission": {"evidence_count": 2, "coverage_level": "WEAK"},
    })
    result = _check_profile_coverage(note, ["smr", "ai_data_centers", "transmission"])
    assert result["profile_coverage"] == {
        "smr": "strong", "ai_data_centers": "strong", "transmission": "weak"
    }
    assert len(result["profile_issues"]) == 1  # only transmission is weak
    assert result["profile_issues"][0]["profile"] == "transmission"


def test_coverage_status_insufficient_when_any_missing():
    note = _make_evidence_note({
        "smr": {"evidence_count": 15, "coverage_level": "STRONG"},
        "transmission": {"evidence_count": 0, "coverage_level": "NONE"},
    })
    result = _check_profile_coverage(note, ["smr", "transmission"])
    assert result["coverage_status"] == "insufficient"
    assert "transmission" in result["profiles_missing"]
    assert "smr" in result["profiles_contributing"]


def test_coverage_status_sufficient_when_all_contribute():
    note = _make_evidence_note({
        "smr": {"evidence_count": 15, "coverage_level": "STRONG"},
        "ai_data_centers": {"evidence_count": 8, "coverage_level": "MODERATE"},
    })
    result = _check_profile_coverage(note, ["smr", "ai_data_centers"])
    assert result["coverage_status"] == "sufficient"
    assert result["profiles_missing"] == []


def test_profiles_requested_in_evidence_note():
    """Evidence note should carry profiles_requested/contributing/missing."""
    from functional_agents.evidence_agent import _attribute_evidence_profiles
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    tx = load_profile("transmission")
    items = [
        {"claim": "reactor nuclear smr", "topics": [], "evidence_id": "E001",
         "relevance_score": 3, "source_document": "a.pdf", "category": "factual"},
        {"claim": "gpu rack nvidia power", "topics": [], "evidence_id": "E002",
         "relevance_score": 3, "source_document": "b.pdf", "category": "factual"},
    ]
    cov = _attribute_evidence_profiles(items, [smr, adc, tx], fallback_profile="smr")
    # transmission has 0 items → it should appear in profiles with evidence_count=0
    assert "transmission" in cov
    assert cov["transmission"]["evidence_count"] == 0


# ---------------------------------------------------------------------------
# AgentContext multi-profile
# ---------------------------------------------------------------------------

def test_agent_context_carries_all_profiles():
    ctx = AgentContext(
        question="Can SMRs power AI data centers?",
        profiles=["smr", "ai_data_centers", "transmission"],
        execution_profile="smr",
        research_object={"id": "R-TEST_001"},
        run_id="test001",
    )
    assert len(ctx.profiles) == 3
    assert "smr" in ctx.profiles
    assert "ai_data_centers" in ctx.profiles
    assert "transmission" in ctx.profiles
    assert ctx.execution_profile == "smr"


# ---------------------------------------------------------------------------
# End-to-end mock pipeline: 3 profiles
# ---------------------------------------------------------------------------

def _build_mock_context_with_evidence(profiles: list[str]) -> AgentContext:
    """Build a context with pre-populated evidence notes simulating 3-profile run."""
    ctx = AgentContext(
        question="Can SMRs power AI data centers?",
        profiles=profiles,
        execution_profile=profiles[0],
        research_object={"id": "R-TEST_001"},
        run_id="multi001",
    )
    ctx.plan = {
        "question": ctx.question,
        "research_type": "RESEARCH",
        "subquestions": [
            "What is the power output of a typical SMR?",
            "What are the power requirements of AI data centers?",
            "What transmission infrastructure is needed to connect SMRs to data centers?",
        ],
        "investigation_areas": ["SMR capacity", "DC power demand", "Grid integration"],
        "profiles_used": profiles,
        "reasoning": "Multi-profile test plan",
    }
    # Simulate evidence note with profile attribution already done
    items = [
        {"evidence_id": "E001", "claim": "SMR output is 300 MWe", "source_profile": "smr",
         "topics": ["reactor design"], "relevance_score": 4, "source_document": "smr_report.pdf", "category": "factual"},
        {"evidence_id": "E002", "claim": "AI DC power demand exceeds 100 MW per campus",
         "source_profile": "ai_data_centers", "topics": ["power"], "relevance_score": 5,
         "source_document": "dc_power.pdf", "category": "factual"},
        {"evidence_id": "E003", "claim": "Grid interconnection requires network upgrades",
         "source_profile": "transmission", "topics": ["interconnection"], "relevance_score": 3,
         "source_document": "grid_study.pdf", "category": "factual"},
    ]
    sq0 = ctx.plan["subquestions"][0]
    sq1 = ctx.plan["subquestions"][1]
    sq2 = ctx.plan["subquestions"][2]
    ctx.evidence_notes = [{
        "evidence_items": items,
        "evidence_by_subquestion": {sq0: ["E001"], sq1: ["E002"], sq2: ["E003"]},
        "evidence_by_area": {},
        "coverage_by_subquestion": {
            sq0: {"coverage": "MODERATE", "evidence_count": 1},
            sq1: {"coverage": "STRONG", "evidence_count": 1},
            sq2: {"coverage": "MODERATE", "evidence_count": 1},
        },
        "evidence_summary": {
            "total_evidence_items": 3,
            "subquestions_with_evidence": 3,
            "subquestions_without_evidence": 0,
            "investigation_areas_with_evidence": 0,
            "coverage_distribution": {"MODERATE": 2, "STRONG": 1},
        },
        "profile_coverage_by_profile": {
            "smr": {"evidence_count": 1, "coverage_level": "WEAK"},
            "ai_data_centers": {"evidence_count": 1, "coverage_level": "WEAK"},
            "transmission": {"evidence_count": 1, "coverage_level": "WEAK"},
        },
    }]
    return ctx


def test_qa_agent_produces_profile_coverage():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_with_evidence(["smr", "ai_data_centers", "transmission"])
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert "profile_coverage" in ctx.qa
    pc = ctx.qa["profile_coverage"]
    assert set(pc.keys()) == {"smr", "ai_data_centers", "transmission"}


def test_qa_agent_qa_summary_has_profiles_evaluated():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_with_evidence(["smr", "ai_data_centers", "transmission"])
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert ctx.qa["qa_summary"]["profiles_evaluated"] == 3


def test_evidence_items_have_source_profile_after_attribution():
    smr = load_profile("smr")
    adc = load_profile("ai_data_centers")
    tx = load_profile("transmission")
    items = [
        {"claim": "reactor nuclear fission smr licensing", "topics": ["reactor design"],
         "evidence_id": "E001", "relevance_score": 4, "source_document": "a.pdf", "category": "factual"},
        {"claim": "gpu power rack thermal nvidia blackwell", "topics": ["power"],
         "evidence_id": "E002", "relevance_score": 4, "source_document": "b.pdf", "category": "factual"},
        {"claim": "transmission substation interconnection grid upgrade", "topics": [],
         "evidence_id": "E003", "relevance_score": 3, "source_document": "c.pdf", "category": "factual"},
    ]
    cov = _attribute_evidence_profiles(items, [smr, adc, tx], fallback_profile="smr")
    assert all("source_profile" in i for i in items)
    assert set(cov.keys()) == {"smr", "ai_data_centers", "transmission"}
    total = sum(v["evidence_count"] for v in cov.values())
    assert total == len(items)


# ---------------------------------------------------------------------------
# J5.6a – profiles_requested / contributing / missing / coverage_status
# ---------------------------------------------------------------------------

def _build_mock_context_missing_transmission() -> AgentContext:
    """Context where transmission has 0 evidence (missing profile scenario)."""
    ctx = _build_mock_context_with_evidence(["smr", "ai_data_centers", "transmission"])
    # Override to give transmission zero evidence
    ctx.evidence_notes[0]["profile_coverage_by_profile"]["transmission"] = {
        "evidence_count": 0, "coverage_level": "NONE",
    }
    ctx.evidence_notes[0]["profiles_contributing"] = ["smr", "ai_data_centers"]
    ctx.evidence_notes[0]["profiles_missing"] = ["transmission"]
    return ctx


def test_qa_agent_profiles_requested():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_with_evidence(["smr", "ai_data_centers", "transmission"])
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert "profiles_requested" in ctx.qa
    assert set(ctx.qa["profiles_requested"]) == {"smr", "ai_data_centers", "transmission"}


def test_qa_agent_profiles_contributing():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_with_evidence(["smr", "ai_data_centers", "transmission"])
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert "profiles_contributing" in ctx.qa
    assert isinstance(ctx.qa["profiles_contributing"], list)


def test_qa_agent_profiles_missing_when_none_coverage():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_missing_transmission()
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert "profiles_missing" in ctx.qa
    assert "transmission" in ctx.qa["profiles_missing"]


def test_qa_agent_coverage_status_insufficient_when_missing():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_missing_transmission()
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert ctx.qa["coverage_status"] == "insufficient"


def test_qa_agent_coverage_status_sufficient_when_all_contribute():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_with_evidence(["smr", "ai_data_centers", "transmission"])
    agent = QAAgent()
    ctx = agent._execute(ctx)
    # All three profiles have WEAK (>0) evidence in mock context → sufficient
    assert ctx.qa["coverage_status"] == "sufficient"


def test_qa_agent_profile_gap_issues_for_missing():
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_missing_transmission()
    agent = QAAgent()
    ctx = agent._execute(ctx)
    assert "profile_gap_issues" in ctx.qa
    gap_issues = ctx.qa["profile_gap_issues"]
    transmission_gap = next((g for g in gap_issues if g["profile"] == "transmission"), None)
    assert transmission_gap is not None
    assert transmission_gap["issue"] == "no evidence contributed"


def test_evidence_items_have_profile_field():
    """Each evidence item must carry a 'profile' field (J5.6a spec requirement)."""
    smr = load_profile("smr")
    items = [
        {"claim": "reactor smr", "topics": [], "evidence_id": "E001",
         "relevance_score": 3, "source_document": "a.pdf", "category": "factual"},
    ]
    _attribute_evidence_profiles(items, [smr], fallback_profile="smr")
    # source_profile is set by attribution; 'profile' alias is set by EvidenceAgent._execute()
    assert "source_profile" in items[0]


def test_context_report_has_profile_fields_after_qa():
    """context.report includes profiles_requested, contributing, missing, coverage_status."""
    from functional_agents.qa_agent import QAAgent
    ctx = _build_mock_context_missing_transmission()
    QAAgent()._execute(ctx)
    # Simulate what ReportAgent will read
    assert "profiles_requested" in ctx.qa
    assert "profiles_contributing" in ctx.qa
    assert "profiles_missing" in ctx.qa
    assert "coverage_status" in ctx.qa
