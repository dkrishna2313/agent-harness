"""Tests for research_agent.evaluation.prohibited_term_checker (J3.1c).

Covers:
  - Context detection: negation before term (J3.1c.2)
  - Context detection: sentence-level exemption patterns (J3.1c.2)
  - Context detection: contrastive connectors (J3.1c.2)
  - Numeric context detection (J3.1c.3)
  - Classification: HARD_PROHIBITED / CONTEXT_ALLOWED / not_found
  - Audit trace fields (J3.1c.4)
  - Prohibition statistics (J3.1c.5)
  - Candidate cases: NVIDIA_011 (PCIe), SMR_010 (6 months) (J3.1c.6)
  - Scorer integration
"""

from __future__ import annotations

import pytest

from research_agent.evaluation.prohibited_term_checker import (
    ProhibitedTermResult,
    build_prohibition_stats,
    check_all_prohibited_terms,
    check_prohibited_term,
)


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_absent_term_returns_not_found(self):
        r = check_prohibited_term("PCIe connection", "Grace uses NVLink-C2C.")
        assert r.found is False
        assert r.classification == "not_found"
        assert r.penalty_applied is False
        assert r.reason == "term_absent"

    def test_not_found_empty_context(self):
        r = check_prohibited_term("PCIe connection", "Grace uses NVLink-C2C.")
        assert r.context_window == ""


# ---------------------------------------------------------------------------
# HARD_PROHIBITED: term present, no exempting context
# ---------------------------------------------------------------------------

class TestHardProhibited:
    def test_bare_term_is_hard_prohibited(self):
        r = check_prohibited_term(
            "PCIe connection",
            "The Grace CPU communicates via a PCIe connection to the GPU.",
        )
        assert r.found is True
        assert r.classification == "hard_prohibited"
        assert r.penalty_applied is True

    def test_numeric_term_hard_prohibited(self):
        r = check_prohibited_term(
            "6 months",
            "The BWRX-300 can be built in 6 months from first concrete.",
        )
        assert r.found is True
        assert r.classification == "hard_prohibited"
        assert r.penalty_applied is True

    def test_wrong_fact_hard_prohibited(self):
        r = check_prohibited_term(
            "air cooled only",
            "The rack is air cooled only, requiring no liquid infrastructure.",
        )
        assert r.found is True
        assert r.classification == "hard_prohibited"
        assert r.penalty_applied is True

    def test_context_window_populated(self):
        r = check_prohibited_term(
            "36 GPUs",
            "The NVL72 contains 36 GPUs per tray.",
        )
        assert r.context_window  # non-empty
        assert "36 GPUs" in r.context_window


# ---------------------------------------------------------------------------
# CONTEXT_ALLOWED: pre-term negation (J3.1c.2)
# ---------------------------------------------------------------------------

class TestPreTermNegation:
    # J3.1c.6 candidate: SMR_010
    def test_not_before_numeric_term(self):
        r = check_prohibited_term(
            "6 months",
            "BWRX-300 targets 3–4 years, not 6 months as vendor optimists claim.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False
        assert r.reason == "pre_term_negation"

    def test_no_before_term(self):
        r = check_prohibited_term(
            "PCIe connection",
            "The design requires no PCIe connection between Grace and Blackwell.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_without_before_term(self):
        r = check_prohibited_term(
            "PCIe connection",
            "NVLink-C2C allows chip-to-chip communication without a PCIe connection.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False


# ---------------------------------------------------------------------------
# CONTEXT_ALLOWED: sentence-level exemption (J3.1c.2)
# ---------------------------------------------------------------------------

class TestComparisonAndLegacyExemptions:
    """J3.1c.1 — patterns added to fix NVIDIA_011: exceed, surpass, legacy, etc."""

    def test_exceeding_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "Grace and Blackwell communicate at 900 GB/s, far exceeding what a PCIe connection could provide.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_legacy_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "The design abandons the legacy PCIe connection in favour of NVLink-C2C.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_conventional_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "NVLink-C2C is 7× faster than a conventional PCIe connection between CPU and GPU.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_traditional_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "Unlike the traditional PCIe connection, NVLink-C2C is co-packaged inside the module.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_faster_than_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "NVLink-C2C is significantly faster than a PCIe connection, reaching 900 GB/s.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_surpasses_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "The NVLink-C2C interconnect surpasses PCIe connection bandwidth by 7×.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_overhead_pcie_context_allowed(self):
        r = check_prohibited_term(
            "PCIe connection",
            "Co-packaging eliminates the power overhead of a PCIe connection.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_paragraph_exemption_adjacent_sentence(self):
        """Exemption word is only in the NEXT sentence — paragraph fallback triggers."""
        # Prohibited term's own sentence has no exemption signal.
        # The adjacent sentence 'NVLink-C2C replaces this...' provides the context.
        r = check_prohibited_term(
            "PCIe connection",
            (
                "The PCIe connection from prior designs adds significant latency. "
                "NVLink-C2C replaces this with a 900 GB/s chip-to-chip path."
            ),
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False
        assert r.reason == "paragraph_exemption"

    def test_paragraph_bandwidth_comparison(self):
        """'PCIe connection bandwidth is limited to X; NVLink provides Y' → context_allowed."""
        r = check_prohibited_term(
            "PCIe connection",
            "PCIe connection bandwidth is limited to ~128 GB/s; NVLink-C2C provides 900 GB/s bidirectional.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False


class TestSentenceExemption:
    # J3.1c.6 candidate: NVIDIA_011
    def test_eliminates_exemption(self):
        r = check_prohibited_term(
            "PCIe connection",
            "Grace Blackwell eliminates the need for a PCIe connection to the host.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False
        assert r.reason == "sentence_exemption"

    def test_eliminates_bottleneck_phrasing(self):
        # Note: "PCIe bottlenecks" doesn't contain the prohibited phrase "PCIe connection".
        # The real-world false-positive occurs when the answer explicitly mentions the phrase
        # "PCIe connection" while explaining it was eliminated.
        r = check_prohibited_term(
            "PCIe connection",
            "Grace CPU eliminates the PCIe connection bottleneck through NVLink-C2C integration.",
        )
        # "bottleneck" in same sentence as "PCIe connection" → context_allowed
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_replaces_exemption(self):
        r = check_prohibited_term(
            "PCIe connection",
            "NVLink-C2C replaces the PCIe connection, delivering 900 GB/s.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_avoids_exemption(self):
        r = check_prohibited_term(
            "PCIe connection",
            "Co-packaging avoids the PCIe connection overhead entirely.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_bottleneck_exemption(self):
        r = check_prohibited_term(
            "PCIe connection",
            "The PCIe connection bottleneck is what Grace was designed to remove.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_no_longer_need_exemption(self):
        r = check_prohibited_term(
            "PCIe connection",
            "Data centres no longer need a PCIe connection to link GPU and CPU.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_mis_stated_numeric(self):
        r = check_prohibited_term(
            "6 months",
            "Contrary to mis-stated claims of 6 months, construction takes 3–4 years.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False


# ---------------------------------------------------------------------------
# CONTEXT_ALLOWED: contrastive connectors (J3.1c.2)
# ---------------------------------------------------------------------------

class TestContrastiveConnectors:
    def test_unlike_connector(self):
        r = check_prohibited_term(
            "6 months",
            "Unlike large-reactor projects where 6 months of delays is routine, SMRs target 3 years.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_whereas_connector(self):
        r = check_prohibited_term(
            "PCIe connection",
            "The new architecture uses NVLink-C2C, whereas a PCIe connection was standard before.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_however_connector(self):
        r = check_prohibited_term(
            "air cooled only",
            "Some legacy designs were air cooled only; however, modern high-density racks require liquid.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False

    def test_versus_connector(self):
        r = check_prohibited_term(
            "6 months",
            "Vendor targets (6 months per module versus 3–4 years total) show the breakdown.",
        )
        assert r.classification == "context_allowed"
        assert r.penalty_applied is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_case_insensitive(self):
        r = check_prohibited_term("PCIe Connection", "Grace ELIMINATES the need for a PCIE CONNECTION.")
        assert r.classification == "context_allowed"

    def test_empty_answer(self):
        r = check_prohibited_term("6 months", "")
        assert r.found is False
        assert r.classification == "not_found"

    def test_term_at_start_of_answer(self):
        r = check_prohibited_term("6 months", "6 months is all it takes to build.")
        assert r.classification == "hard_prohibited"
        assert r.penalty_applied is True

    def test_context_window_is_sentence(self):
        answer = (
            "The first sentence is fine. "
            "Grace eliminates the need for a PCIe connection to the GPU. "
            "Third sentence unrelated."
        )
        r = check_prohibited_term("PCIe connection", answer)
        # Context window should be the middle sentence, not the full text
        assert len(r.context_window) < len(answer)


# ---------------------------------------------------------------------------
# Batch helpers and statistics (J3.1c.5)
# ---------------------------------------------------------------------------

class TestBatchAndStats:
    def test_check_all_returns_one_per_term(self):
        terms = ["PCIe connection", "6 months", "36 GPUs"]
        answer = "Grace eliminates PCIe connection overhead. Construction takes 3–4 years. The rack has 72 GPUs."
        results = check_all_prohibited_terms(terms, answer)
        assert len(results) == 3

    def test_build_prohibition_stats_structure(self):
        results = [
            ProhibitedTermResult("a", True, "...", "hard_prohibited", True, "no_exemption_found"),
            ProhibitedTermResult("b", True, "...", "context_allowed", False, "sentence_exemption"),
            ProhibitedTermResult("c", False, "", "not_found", False, "term_absent"),
        ]
        stats = build_prohibition_stats(results)
        assert stats["hard_prohibited"] == 1
        assert stats["context_allowed"] == 1
        assert stats["not_found"] == 1
        assert stats["total_checked"] == 3

    def test_all_absent_stats(self):
        results = [
            ProhibitedTermResult("a", False, "", "not_found", False, "term_absent"),
            ProhibitedTermResult("b", False, "", "not_found", False, "term_absent"),
        ]
        stats = build_prohibition_stats(results)
        assert stats["hard_prohibited"] == 0
        assert stats["context_allowed"] == 0
        assert stats["not_found"] == 2

    def test_to_dict_structure(self):
        r = ProhibitedTermResult(
            term="PCIe connection",
            found=True,
            context_window="Grace eliminates PCIe connection.",
            classification="context_allowed",
            penalty_applied=False,
            reason="sentence_exemption",
        )
        d = r.to_dict()
        assert d["prohibited_term"] == "PCIe connection"
        assert d["found"] is True
        assert d["classification"] == "context_allowed"
        assert d["penalty_applied"] is False
        assert d["reason"] == "sentence_exemption"
        assert "context" in d


# ---------------------------------------------------------------------------
# J3.1c.6: Scorer integration — NVIDIA_011 and SMR_010 candidate cases
# ---------------------------------------------------------------------------

class TestScorerIntegration:
    def test_nvidia_011_pcie_contextual_mention_no_penalty(self):
        """NVIDIA_011: 'PCIe connection' in answer that explains elimination → no penalty."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="NVIDIA_011",
            domain="nvidia",
            difficulty="medium",
            question="What is the Grace CPU's role in the GB200 system?",
            must_include=["CPU", "Grace"],
            must_not_include=["Grace is a separate server", "PCIe connection"],
            acceptable_alternatives=[],
        )
        # Answer that correctly explains Grace eliminates PCIe — mentions the term in negating context
        memo = ResearchMemo(
            title="Grace CPU",
            question=question.question,
            executive_summary=(
                "The Grace CPU is co-packaged with two B200 GPUs in the GB200 module. "
                "Grace connects to Blackwell via NVLink-C2C, eliminating the need for a "
                "PCIe connection that would otherwise create a bandwidth bottleneck."
            ),
            confirmed_facts=[
                "NVLink-C2C provides ~900 GB/s bandwidth, replacing the PCIe connection "
                "overhead that limited previous CPU-GPU designs.",
                "Grace CPU and Blackwell GPU share a unified memory address space.",
            ],
        )
        score = score_qa_response(question, memo)
        assert "PCIe connection" not in score.must_not_include_violations, (
            f"'PCIe connection' should not be a violation when used in negating context. "
            f"Audit: {score.prohibited_term_audit}"
        )
        assert score.passed is True

    def test_smr_010_6months_negated_no_penalty(self):
        """SMR_010: '6 months' appears but is explicitly negated → no penalty."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="SMR_010",
            domain="smr",
            difficulty="hard",
            question="What construction duration should be expected for BWRX-300?",
            must_include=["years", "construction"],
            must_not_include=["6 months", "20+ years"],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="SMR Construction",
            question=question.question,
            executive_summary=(
                "GE Hitachi targets 3–4 years for BWRX-300 construction from first "
                "concrete to commercial operation. This is far more realistic than "
                "not 6 months as sometimes cited by optimistic projections."
            ),
            confirmed_facts=[
                "Vendor target: 3–4 years construction (not 6 months).",
                "Vogtle 3 & 4 took 7–8 years; Hinkley Point C projects 7–9 years.",
                "SMR modular construction reduces the overall timeline.",
            ],
        )
        score = score_qa_response(question, memo)
        assert "6 months" not in score.must_not_include_violations, (
            f"'6 months' should not be a violation when negated. "
            f"Audit: {score.prohibited_term_audit}"
        )
        assert score.passed is True

    def test_hard_prohibited_still_fails(self):
        """A plain prohibited claim (no negating context) still triggers penalty."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="HARD_001",
            domain="nvidia",
            difficulty="easy",
            question="How does Grace connect to Blackwell?",
            must_include=["NVLink"],
            must_not_include=["PCIe connection"],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="t",
            question=question.question,
            executive_summary=(
                "The Grace CPU uses a PCIe connection to communicate with the Blackwell GPU."
            ),
        )
        score = score_qa_response(question, memo)
        assert "PCIe connection" in score.must_not_include_violations
        assert score.hallucination_penalty == 1.0

    def test_prohibited_term_audit_in_score(self):
        """QAScore.prohibited_term_audit contains per-term detail dicts (J3.1c.4)."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="AUDIT_011",
            domain="nvidia",
            difficulty="easy",
            question="Grace CPU role?",
            must_include=["Grace"],
            must_not_include=["PCIe connection"],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="t",
            question=question.question,
            executive_summary=(
                "Grace eliminates the need for a PCIe connection to the Blackwell GPU."
            ),
        )
        score = score_qa_response(question, memo)
        assert len(score.prohibited_term_audit) == 1
        audit = score.prohibited_term_audit[0]
        assert audit["prohibited_term"] == "PCIe connection"
        assert audit["found"] is True
        assert audit["classification"] == "context_allowed"
        assert audit["penalty_applied"] is False
        assert "context" in audit

    def test_context_allowed_count_in_score(self):
        """QAScore.context_allowed_count reflects number of exempted terms."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="COUNT_001",
            domain="smr",
            difficulty="hard",
            question="Construction timeline?",
            must_include=["years"],
            must_not_include=["6 months", "20+ years"],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="t",
            question=question.question,
            executive_summary=(
                "Target is 3–4 years, not 6 months. No evidence suggests 20+ years "
                "timelines for SMRs under normal conditions."
            ),
        )
        score = score_qa_response(question, memo)
        assert score.context_allowed_count >= 1
        # Neither term should appear as a violation
        assert score.must_not_include_violations == []


class TestContrastiveWhile:
    """J7.6c – 'while' and 'although' are contrastive connectors (NVIDIA_008 regression fix)."""

    def test_while_exempts_prohibited_term(self):
        """'while X uses GDDR7' is a comparison, not a hallucinated claim about the query GPU."""
        from research_agent.evaluation.prohibited_term_checker import check_prohibited_term
        answer = (
            "The GB200 Blackwell GPU uses HBM3e memory with 8 TB/s bandwidth. "
            "The GH100 uses HBM2e with 80 GB capacity, while the GB203 (Blackwell consumer GPU) uses GDDR7 with lower bandwidth."
        )
        result = check_prohibited_term("GDDR", answer)
        assert result.classification == "context_allowed", (
            f"Expected context_allowed but got {result.classification}. "
            f"Context: {result.context_window}"
        )
        assert result.penalty_applied is False

    def test_although_exempts_prohibited_term(self):
        from research_agent.evaluation.prohibited_term_checker import check_prohibited_term
        answer = "HBM3e is used in datacenter GPUs, although consumer cards rely on GDDR7."
        result = check_prohibited_term("GDDR", answer)
        assert result.classification == "context_allowed"
        assert result.penalty_applied is False

    def test_hard_prohibited_without_contrastive(self):
        """Without a contrastive connector, GDDR in datacenter context is still penalised."""
        from research_agent.evaluation.prohibited_term_checker import check_prohibited_term
        answer = "The GB200 GPU uses GDDR7 memory for its high bandwidth requirements."
        result = check_prohibited_term("GDDR", answer)
        assert result.classification == "hard_prohibited"
        assert result.penalty_applied is True
