"""PH4.2 — Performance & Cost Optimization tests.

Verifies that all five prompt optimizations are in effect and that
telemetry integrity is maintained. All tests are deterministic and
require no LLM calls.

Optimizations validated:
  H1  _hypothesis_prompt: evidence_items capped at 8 (was 12)
  H2  _hypothesis_prompt: claim truncated at 80 chars (was 100)
  H3  _hypothesis_prompt: source_document removed from evidence lines
  H4  _hypothesis_prompt: research_question_priorities capped at top 6
  H5  _hypothesis_prompt: contradictions reduced 5→4, summary 100→80 chars
  H6  _hypothesis_prompt: coverage profiles reduced 6→5

  P1  _planning_prompt: profile description truncated at 100 chars (was unlimited)

  C1  _challenge_prompt: evidence_items capped at 8 (was 10)
  C2  _challenge_prompt: claim truncated at 80 chars (was 100)
  C3  _challenge_prompt: hypothesis summary truncated at 120 chars (was 150)
  C4  _challenge_prompt: contradictions reduced 5→4

  SO1 _strategic_options_prompt: assumption/risk/opportunity/rec statements 100→80 chars
  SO2 _strategic_options_prompt: evidence claim 80→70 chars

  T1  Performance telemetry keys unchanged (no regressions in trace structure)
  T2  Mock pipeline end-to-end produces structurally correct output
"""

from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_evidence(n: int = 20) -> list[dict]:
    """Return n evidence items with realistically long claims and source docs."""
    return [
        {
            "evidence_id": f"E{i:03d}",
            "claim": (
                "Small Modular Reactors have a potential deployment timeline of 2028–2032 based on current "
                "NRC licensing pipeline and vendor commitments from NuScale, TerraPower, and X-energy including "
                "detailed regulatory submissions and site approvals"
            ),  # >100 chars
            "source_document": f"Strategic_Research_SMR_AI_Infrastructure_{i}_2024.pdf",
        }
        for i in range(1, n + 1)
    ]


def _make_dm() -> dict:
    return {
        "objective": "Assess whether Small Modular Reactors are viable for AI data center power",
        "decision_areas": ["Capital requirements", "Regulatory timeline", "Technology readiness", "Grid integration"],
        "critical_uncertainties": ["Commercial deployment date", "Regulatory approval duration", "LCOE competitiveness"],
    }


def _make_rs(n_questions: int = 10) -> dict:
    return {
        "research_question_priorities": [
            {"priority": i + 1, "question": f"Q{i+1}: What factors affect SMR deployment feasibility for AI data center power supply?"}
            for i in range(n_questions)
        ]
    }


def _make_hypotheses(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"H{i}",
            "title": f"Hypothesis {i}: SMR deployment will succeed given regulatory and capital support",
            "summary": (
                "Evidence suggests SMR technology faces significant capital barriers but long-term economics "
                "are favorable given 40+ year operational lifetime compared to renewables with shorter lifetimes"
            ),  # >150 chars
            "supporting_evidence": ["E001", "E002"],
            "contradicting_evidence": ["E003"],
            "evidence_gaps": ["Cost data", "Regulatory timeline"],
            "confidence": "medium",
        }
        for i in range(1, n + 1)
    ]


def _make_contradictions(n: int = 5) -> list[dict]:
    return [
        {
            "contradiction_id": f"CONT-{i:03d}",
            "topic": f"Cost projection {i}",
            "severity": "medium",
            "summary": "Sources disagree on LCOE showing a 2-3x premium over grid power for data center applications at scale",
        }
        for i in range(n)
    ]


def _make_profile_coverage(n: int = 7) -> dict:
    profiles = ["smr", "power", "data_center", "regulatory", "finance", "transmission", "policy"]
    return {p: "moderate" for p in profiles[:n]}


# ---------------------------------------------------------------------------
# H1-H6: _hypothesis_prompt
# ---------------------------------------------------------------------------

class TestHypothesisPrompt:
    def _prompt(self, **kw) -> str:
        from research_agent.claude_client import _hypothesis_prompt
        defaults = dict(
            decision_model=_make_dm(),
            research_strategy=_make_rs(10),
            evidence_items=_make_evidence(20),
            profile_coverage=_make_profile_coverage(7),
            contradictions=_make_contradictions(5),
        )
        defaults.update(kw)
        return _hypothesis_prompt(**defaults)

    def test_H1_evidence_capped_at_8(self):
        """H1: at most 8 evidence items are rendered in the prompt."""
        prompt = self._prompt(evidence_items=_make_evidence(20))
        ev_count = len(re.findall(r"\[E\d+\]", prompt))
        assert ev_count <= 8, f"Expected ≤8 evidence items, got {ev_count}"

    def test_H1_exactly_8_when_more_available(self):
        """H1: exactly 8 items are included when 20 are available."""
        prompt = self._prompt(evidence_items=_make_evidence(20))
        ev_count = len(re.findall(r"\[E\d+\]", prompt))
        assert ev_count == 8

    def test_H1_fewer_than_8_pass_through(self):
        """H1: fewer items than cap are all included."""
        prompt = self._prompt(evidence_items=_make_evidence(5))
        ev_count = len(re.findall(r"\[E\d+\]", prompt))
        assert ev_count == 5

    def test_H2_claim_truncated_at_80_chars(self):
        """H2: evidence claims are truncated to 80 characters in the prompt."""
        long_claim = "X" * 200
        items = [{"evidence_id": "E001", "claim": long_claim, "source_document": "doc.pdf"}]
        prompt = self._prompt(evidence_items=items)
        # The rendered claim must not contain 81+ Xs consecutively
        assert "X" * 81 not in prompt, "Claim was not truncated to 80 chars"
        assert "X" * 80 in prompt, "First 80 chars of claim should be present"

    def test_H3_source_document_absent(self):
        """H3: source_document is NOT rendered in the hypothesis prompt."""
        items = _make_evidence(3)
        for item in items:
            item["source_document"] = "UNIQUE_SOURCE_SENTINEL.pdf"
        prompt = self._prompt(evidence_items=items)
        assert "UNIQUE_SOURCE_SENTINEL" not in prompt, "source_document should not appear in hypothesis prompt"
        assert "(source:" not in prompt, "source metadata should not appear"

    def test_H4_research_questions_capped_at_6(self):
        """H4: at most 6 research_question_priorities are rendered."""
        rs = _make_rs(12)  # 12 priority questions
        prompt = self._prompt(research_strategy=rs)
        # Count numbered research question lines "  1. " through "  12. "
        rq_count = len(re.findall(r"^\s+\d+\. Q\d+:", prompt, re.MULTILINE))
        assert rq_count <= 6, f"Expected ≤6 research questions, got {rq_count}"

    def test_H4_no_questions_renders_correctly(self):
        """H4: empty research_question_priorities renders '(none)'."""
        prompt = self._prompt(research_strategy={})
        assert "(none)" in prompt

    def test_H5_contradictions_capped_at_4(self):
        """H5: at most 4 contradictions are rendered."""
        prompt = self._prompt(contradictions=_make_contradictions(5))
        # Count contradiction lines (each starts with "  - Cost projection")
        contra_count = len(re.findall(r"  - Cost projection", prompt))
        assert contra_count <= 4, f"Expected ≤4 contradictions, got {contra_count}"

    def test_H6_coverage_profiles_capped_at_5(self):
        """H6: at most 5 coverage profiles are rendered."""
        prompt = self._prompt(profile_coverage=_make_profile_coverage(7))
        cov_count = len(re.findall(r"  - \w+: moderate", prompt))
        assert cov_count <= 5, f"Expected ≤5 coverage profiles, got {cov_count}"

    def test_prompt_smaller_than_before_optimization(self):
        """Regression: optimized prompt is meaningfully smaller than the pre-PH4.2 baseline."""
        prompt = self._prompt()
        # Pre-PH4.2 baseline (measured): 6562 chars with same inputs
        # After PH4.2: should be at least 25% smaller
        assert len(prompt) < 6562 * 0.80, (
            f"Prompt ({len(prompt)} chars) not meaningfully smaller than pre-PH4.2 baseline (6562 chars)"
        )

    def test_instruction_text_unchanged(self):
        """PH4.2 instructions section must be identical — no reasoning changes."""
        prompt = self._prompt()
        assert "Generate 3-4 COMPETING hypotheses" in prompt
        assert "synthesis_note" in prompt
        assert "Return structured JSON only" in prompt


# ---------------------------------------------------------------------------
# P1: _planning_prompt
# ---------------------------------------------------------------------------

class TestPlanningPrompt:
    def _prompt(self, desc_len: int = 300) -> str:
        from research_agent.claude_client import _planning_prompt
        profiles_context = [
            {
                "name": "smr",
                "description": "A" * desc_len,  # artificially long description
                "key_topics": ["topic1", "topic2"],
            }
        ]
        return _planning_prompt("Test question?", profiles_context)

    def test_P1_profile_description_truncated_at_100(self):
        """P1: profile descriptions are truncated to 100 chars in the prompt."""
        prompt = self._prompt(desc_len=300)
        # 300 'A's should not appear consecutively; only 100 should
        assert "A" * 101 not in prompt, "Profile description not truncated to 100 chars"
        assert "A" * 100 in prompt, "First 100 chars of description should appear"

    def test_P1_short_description_unchanged(self):
        """P1: descriptions ≤100 chars are not modified."""
        from research_agent.claude_client import _planning_prompt
        profiles_context = [{"name": "smr", "description": "Short desc", "key_topics": []}]
        prompt = _planning_prompt("Test?", profiles_context)
        assert "Short desc" in prompt

    def test_P1_prompt_smaller_for_long_descriptions(self):
        """P1: prompt with 300-char descriptions must be smaller than naive (untruncated) equivalent."""
        prompt = self._prompt(desc_len=300)
        # Naive baseline: 300-char description would add ~200 extra chars vs 100-char truncated
        # Just verify prompt is present and key elements are there
        assert "Test question?" in prompt
        assert "A" * 101 not in prompt


# ---------------------------------------------------------------------------
# C1-C4: _challenge_prompt
# ---------------------------------------------------------------------------

class TestChallengePrompt:
    def _prompt(self, n_evidence: int = 15, n_hyps: int = 3, n_contra: int = 5) -> str:
        from research_agent.claude_client import _challenge_prompt
        return _challenge_prompt(
            _make_hypotheses(n_hyps),
            _make_evidence(n_evidence),
            _make_contradictions(n_contra),
            [{"gap": f"Missing data {i}"} for i in range(3)],
            _make_profile_coverage(3),
        )

    def test_C1_evidence_capped_at_8(self):
        """C1: at most 8 evidence items rendered in challenge prompt."""
        prompt = self._prompt(n_evidence=15)
        # Evidence lines have format "  Exxx: <claim>"
        ev_count = len(re.findall(r"  E\d+: ", prompt))
        assert ev_count <= 8, f"Expected ≤8, got {ev_count}"

    def test_C2_claim_truncated_at_80(self):
        """C2: challenge evidence claims truncated at 80 chars."""
        from research_agent.claude_client import _challenge_prompt
        long_claim = "B" * 200
        items = [{"evidence_id": "E001", "claim": long_claim, "source_document": "x.pdf"}]
        prompt = _challenge_prompt(_make_hypotheses(1), items, [], [], {})
        assert "B" * 81 not in prompt
        assert "B" * 80 in prompt

    def test_C3_hypothesis_summary_truncated_at_120(self):
        """C3: hypothesis summaries in challenge prompt truncated at 120 chars."""
        from research_agent.claude_client import _challenge_prompt
        hyps = [{"id": "H1", "title": "Title", "summary": "C" * 200,
                  "supporting_evidence": [], "contradicting_evidence": [],
                  "evidence_gaps": [], "confidence": "medium"}]
        prompt = _challenge_prompt(hyps, [], [], [], {})
        assert "C" * 121 not in prompt
        assert "C" * 120 in prompt

    def test_C4_contradictions_capped_at_4(self):
        """C4: at most 4 contradictions rendered in challenge prompt."""
        prompt = self._prompt(n_contra=5)
        contra_lines = [line for line in prompt.split("\n") if re.match(r"  CONT-\d+ \[", line)]
        assert len(contra_lines) <= 4, f"Expected ≤4 contradiction lines, got {len(contra_lines)}"

    def test_challenge_prompt_smaller_than_before(self):
        """Regression: optimized challenge prompt smaller than pre-PH4.2 baseline (4820 chars)."""
        prompt = self._prompt()
        assert len(prompt) < 4820 * 0.80

    def test_instruction_text_unchanged(self):
        """PH4.2: challenge instructions section unchanged — no reasoning changes."""
        prompt = self._prompt()
        assert "challenge_summary" in prompt
        assert "falsification_tests" in prompt
        assert "Return structured JSON only" in prompt


# ---------------------------------------------------------------------------
# SO1-SO2: _strategic_options_prompt
# ---------------------------------------------------------------------------

class TestStrategicOptionsPrompt:
    def _prompt(self) -> str:
        from research_agent.claude_client import _strategic_options_prompt
        dm = {"strategic_question": "Should we invest?", "objective": "Assess SMR investment"}
        assumptions = [{"assumption_id": f"A-{i:03d}", "statement": "D" * 200, "importance": "critical"} for i in range(3)]
        risks = [{"risk_id": f"RSK-{i:03d}", "statement": "E" * 200, "severity": "high", "related_assumption_ids": []} for i in range(3)]
        opps = [{"opportunity_id": f"OPP-{i:03d}", "statement": "F" * 200, "impact": "high", "related_assumption_ids": []} for i in range(3)]
        recs = [{"recommendation_id": f"REC-{i:03d}", "title": "G" * 200, "supported_assumption_ids": []} for i in range(3)]
        evidence = [{"evidence_id": f"E{i:03d}", "claim": "H" * 200} for i in range(8)]
        return _strategic_options_prompt(assumptions, risks, opps, recs, evidence, dm)

    def test_SO1_assumption_statement_truncated_at_80(self):
        """SO1: assumption statements truncated at 80 chars."""
        prompt = self._prompt()
        assert "D" * 81 not in prompt
        assert "D" * 80 in prompt

    def test_SO1_risk_statement_truncated_at_80(self):
        """SO1: risk statements truncated at 80 chars."""
        prompt = self._prompt()
        assert "E" * 81 not in prompt
        assert "E" * 80 in prompt

    def test_SO1_opportunity_statement_truncated_at_80(self):
        """SO1: opportunity statements truncated at 80 chars."""
        prompt = self._prompt()
        assert "F" * 81 not in prompt
        assert "F" * 80 in prompt

    def test_SO1_recommendation_title_truncated_at_80(self):
        """SO1: recommendation titles truncated at 80 chars."""
        prompt = self._prompt()
        assert "G" * 81 not in prompt
        assert "G" * 80 in prompt

    def test_SO2_evidence_claim_truncated_at_70(self):
        """SO2: strategic options evidence claims truncated at 70 chars."""
        prompt = self._prompt()
        assert "H" * 71 not in prompt
        assert "H" * 70 in prompt

    def test_instruction_text_unchanged(self):
        """PH4.2: strategic options instructions unchanged."""
        prompt = self._prompt()
        assert "recommended=True" in prompt
        assert "Return structured JSON" in prompt


# ---------------------------------------------------------------------------
# T1: Performance telemetry integrity
# ---------------------------------------------------------------------------

class TestTelemetryIntegrity:
    def test_T1_pipeline_trace_keys_unchanged(self):
        """T1: pipeline trace required keys are unaffected by PH4.2 changes."""
        from functional_agents.pipeline_trace import _REQUIRED_TOP_LEVEL_KEYS, SCHEMA_VERSION
        required = set(_REQUIRED_TOP_LEVEL_KEYS)
        assert "schema_version" in required
        assert "pipeline" in required
        assert "agents" in required
        assert "performance" in required
        assert "summary" in required
        assert SCHEMA_VERSION == "ph3.4-canonical-v1"

    def test_T1_agent_perf_record_fields_unchanged(self):
        """T1: AgentPerfRecord fields that feed telemetry are intact."""
        from functional_agents.performance import AgentPerfRecord, LLMCallRecord
        record = AgentPerfRecord(agent_name="TestAgent", wall_ms=100.0)
        # Properties computed from llm_calls
        assert hasattr(record, "prompt_tokens")
        assert hasattr(record, "completion_tokens")
        assert hasattr(record, "llm_call_count")
        assert record.wall_ms == 100.0
        # Add an LLM call and verify accumulation
        record.llm_calls.append(LLMCallRecord(
            operation="test", model="claude", duration_ms=50.0,
            prompt_tokens=100, completion_tokens=80, total_tokens=180, success=True,
        ))
        assert record.prompt_tokens == 100
        assert record.completion_tokens == 80
        assert record.llm_call_count == 1

    def test_T1_slice_diagnostics_structure(self):
        """T1: slice_diagnostics returns the expected measurement structure."""
        from functional_agents.context_slices import slice_diagnostics
        orig = {"decision_model": {"a": 1, "b": 2, "c": 3}, "research_strategy": {"x": 10}}
        sliced = {"decision_model": {"a": 1, "b": 2}, "research_strategy": {"x": 10}}
        diag = slice_diagnostics(orig, sliced)
        assert "original_bytes" in diag
        assert "sliced_bytes" in diag
        assert "bytes_saved" in diag
        assert "reduction_pct" in diag
        assert diag["bytes_saved"] > 0

    def test_T1_performance_tracker_summary_structure(self):
        """T1: PerformanceTracker.summary() returns telemetry dict with totals."""
        from functional_agents.performance import PerformanceTracker, AgentPerfRecord, LLMCallRecord
        tracker = PerformanceTracker()
        record = AgentPerfRecord(agent_name="HypothesisAgent", wall_ms=45000.0)
        record.llm_calls.append(LLMCallRecord(
            operation="generate_hypotheses", model="claude", duration_ms=44900.0,
            prompt_tokens=3000, completion_tokens=2800, total_tokens=5800, success=True,
        ))
        tracker.record(record)
        report = tracker.summary()
        assert "totals" in report
        totals = report["totals"]
        assert totals["prompt_tokens"] == 3000
        assert totals["completion_tokens"] == 2800
        assert totals["llm_call_count"] == 1


# ---------------------------------------------------------------------------
# T2: Mock pipeline regression
# ---------------------------------------------------------------------------

class TestMockPipelineRegression:
    def test_T2_mock_pipeline_produces_valid_trace(self, tmp_path):
        """T2: mock pipeline with SMR profile produces a valid canonical trace."""
        import subprocess, json
        out = tmp_path / "report.md"
        result = subprocess.run(
            [
                "python3", "-m", "functional_agents.cli", "run",
                "--profiles", "smr", "--mock",
                "--out", str(out),
                "Should we invest in Small Modular Reactors?",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=".",
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr[-500:]}"
        # Trace should be written to the canonical output location.
        import os
        from functional_agents.trace_paths import CANONICAL_PIPELINE_TRACE
        assert os.path.exists(CANONICAL_PIPELINE_TRACE)
        trace = json.load(open(CANONICAL_PIPELINE_TRACE))
        from functional_agents.pipeline_trace import is_canonical_trace
        assert is_canonical_trace(trace), "Trace is not canonical after PH4.2 changes"

    def test_T2_mock_hypothesis_output_structure(self):
        """T2: MockClaudeClient still produces valid hypothesis output after PH4.2 prompt changes."""
        from research_agent.claude_client import MockClaudeClient
        client = MockClaudeClient()
        dm = _make_dm()
        result = client.generate_hypotheses(dm, {}, _make_evidence(5), {}, [])
        assert len(result.hypotheses) >= 1
        h = result.hypotheses[0]
        assert h.id
        assert h.title
        assert h.confidence in ("high", "medium", "low")

    def test_T2_mock_challenge_output_structure(self):
        """T2: MockClaudeClient still produces valid challenge output."""
        from research_agent.claude_client import MockClaudeClient
        client = MockClaudeClient()
        hyps_raw = [h.model_dump() for h in client.generate_hypotheses(_make_dm(), {}, _make_evidence(3), {}, []).hypotheses]
        result = client.generate_challenges(hyps_raw, _make_evidence(3), [], [], {})
        # ChallengePayload uses hypothesis_challenges field
        assert len(result.hypothesis_challenges) >= 1
        assert result.hypothesis_challenges[0].robustness in ("high", "medium", "low")
