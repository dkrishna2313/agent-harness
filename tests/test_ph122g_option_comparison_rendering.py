"""PH12.2g — Strategic Option Comparison table rendering regression tests.

Proves that decision_matrix flat fields are rendered correctly into the
Strategic Option Comparison table, joined by option_id (not list position).
"""
from __future__ import annotations

import copy
import pytest

from functional_agents.editorial.decision_analysis_writer import DecisionAnalysisWriter

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

DIMS = [
    "Strategic Fit",
    "Implementation Risk",
    "Execution Complexity",
    "Capital Requirement",
    "Expected Return",
    "Time to Value",
    "Dependency Strength",
    "Assumption Strength",
    "Risk Exposure",
    "Opportunity Capture",
]

_FLAT_ROW_A = {
    "option_id": "OPT-A",
    "strategic_fit": "High",
    "implementation_risk": "High",
    "execution_complexity": "High",
    "capital_requirement": "Medium",
    "expected_return": "Medium",
    "time_to_value": "Low",
    "dependency_strength": "Very High",
    "assumption_strength": "Medium",
    "risk_exposure": "High",
    "opportunity_capture": "Medium",
    "overall_score": "Medium",
    "strengths": [],
    "weaknesses": [],
}

_FLAT_ROW_B = {
    "option_id": "OPT-B",
    "strategic_fit": "High",
    "implementation_risk": "Medium",
    "execution_complexity": "Medium",
    "capital_requirement": "Medium",
    "expected_return": "High",
    "time_to_value": "Very High",
    "dependency_strength": "Medium",
    "assumption_strength": "High",
    "risk_exposure": "Medium",
    "opportunity_capture": "Medium",
    "overall_score": "High",
}

_FLAT_ROW_C = {
    "option_id": "OPT-C",
    "strategic_fit": "Very High",
    "implementation_risk": "Medium",
    "execution_complexity": "High",
    "capital_requirement": "High",
    "expected_return": "Very High",
    "time_to_value": "High",
    "dependency_strength": "Medium",
    "assumption_strength": "High",
    "risk_exposure": "Medium",
    "opportunity_capture": "Very High",
    "overall_score": "Very High",
}

_FLAT_ROW_D = {
    "option_id": "OPT-D",
    "strategic_fit": "Medium",
    "implementation_risk": "Very High",
    "execution_complexity": "Very High",
    "capital_requirement": "Very High",
    "expected_return": "Very High",
    "time_to_value": "Low",
    "dependency_strength": "Very High",
    "assumption_strength": "Low",
    "risk_exposure": "Very High",
    "opportunity_capture": "High",
    "overall_score": "Medium",
}

_FOUR_ROW_MATRIX = [_FLAT_ROW_A, _FLAT_ROW_B, _FLAT_ROW_C, _FLAT_ROW_D]

_FIELD_MAP = {
    "Strategic Fit": "strategic_fit",
    "Implementation Risk": "implementation_risk",
    "Execution Complexity": "execution_complexity",
    "Capital Requirement": "capital_requirement",
    "Expected Return": "expected_return",
    "Time to Value": "time_to_value",
    "Dependency Strength": "dependency_strength",
    "Assumption Strength": "assumption_strength",
    "Risk Exposure": "risk_exposure",
    "Opportunity Capture": "opportunity_capture",
}


def _writer() -> DecisionAnalysisWriter:
    return DecisionAnalysisWriter(client=None)


def _build_table(matrix: list[dict], dims: list[str] = DIMS) -> dict:
    return _writer()._matrix_from_decision_matrix(matrix, dims)


# ---------------------------------------------------------------------------
# TestStrategyOutputView  →  dimension resolution helper
# ---------------------------------------------------------------------------


class TestDimensionResolution:
    """Tests for _resolve_dim: nested vs flat vs missing."""

    def test_flat_field_resolved(self):
        entry = {"option_id": "OPT-X", "strategic_fit": "Very High"}
        assert DecisionAnalysisWriter._resolve_dim(entry, "Strategic Fit") == "Very High"

    def test_nested_field_takes_priority(self):
        entry = {
            "option_id": "OPT-X",
            "strategic_fit": "Low",
            "dimensions": {"Strategic Fit": "Very High"},
        }
        assert DecisionAnalysisWriter._resolve_dim(entry, "Strategic Fit") == "Very High"

    def test_missing_field_returns_dash(self):
        entry = {"option_id": "OPT-X"}
        assert DecisionAnalysisWriter._resolve_dim(entry, "Strategic Fit") == "—"

    def test_none_dimensions_falls_back_to_flat(self):
        entry = {"option_id": "OPT-X", "dimensions": None, "strategic_fit": "Medium"}
        assert DecisionAnalysisWriter._resolve_dim(entry, "Strategic Fit") == "Medium"


# ---------------------------------------------------------------------------
# TestFlatFieldsRender
# ---------------------------------------------------------------------------


class TestFlatFieldsRender:
    """All ten dimension columns populate from flat decision_matrix rows."""

    def test_all_ten_columns_populated(self):
        table = _build_table([_FLAT_ROW_C])
        headers = table["headers"]
        row = table["rows"][0]
        for dim in DIMS:
            col_idx = headers.index(dim)
            assert row[col_idx] != "—", f"Column '{dim}' rendered as em dash"

    def test_overall_score_present(self):
        table = _build_table([_FLAT_ROW_C])
        row = table["rows"][0]
        assert row[-1] == "Very High"

    def test_correct_values_for_opt_a(self):
        table = _build_table([_FLAT_ROW_A])
        headers = table["headers"]
        row = table["rows"][0]
        assert row[headers.index("Strategic Fit")] == "High"
        assert row[headers.index("Implementation Risk")] == "High"
        assert row[headers.index("Time to Value")] == "Low"
        assert row[headers.index("Opportunity Capture")] == "Medium"

    def test_correct_values_for_opt_c(self):
        table = _build_table([_FLAT_ROW_C])
        headers = table["headers"]
        row = table["rows"][0]
        assert row[headers.index("Strategic Fit")] == "Very High"
        assert row[headers.index("Expected Return")] == "Very High"
        assert row[headers.index("Opportunity Capture")] == "Very High"

    def test_headers_match_dims(self):
        table = _build_table([_FLAT_ROW_A])
        headers = table["headers"]
        assert headers[0] == "Option"
        assert headers[-1] == "Overall"
        assert headers[1:-1] == DIMS


# ---------------------------------------------------------------------------
# TestOptionIdJoin
# ---------------------------------------------------------------------------


class TestOptionIdJoin:
    """Rows are joined by option_id, not list position."""

    def test_reordered_matrix_same_values(self):
        original = _build_table(_FOUR_ROW_MATRIX)
        reordered = _build_table(list(reversed(_FOUR_ROW_MATRIX)))

        # Build a dict from option label → row values for position-independent comparison
        headers = original["headers"]

        def by_option(table):
            return {row[0]: row for row in table["rows"]}

        orig_map = by_option(original)
        rev_map = by_option(reordered)

        for opt_key in orig_map:
            assert orig_map[opt_key] == rev_map[opt_key], (
                f"Values for '{opt_key}' differ after reordering"
            )

    def test_all_four_sports_options_rendered(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        assert len(table["rows"]) == 4

    def test_all_four_options_have_populated_dimensions(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        headers = table["headers"]
        dim_indices = [headers.index(d) for d in DIMS]
        for row in table["rows"]:
            for idx in dim_indices:
                assert row[idx] != "—", f"Row {row[0]} has em dash in column {headers[idx]}"


# ---------------------------------------------------------------------------
# TestPartiallyMissingDimension
# ---------------------------------------------------------------------------


class TestPartiallyMissingDimension:
    """One missing dimension yields one em dash, not an empty row."""

    def test_one_missing_dim_produces_one_dash(self):
        row_data = dict(_FLAT_ROW_A)
        del row_data["risk_exposure"]  # remove one field
        table = _build_table([row_data])
        headers = table["headers"]
        row = table["rows"][0]

        # risk_exposure column should be —
        assert row[headers.index("Risk Exposure")] == "—"
        # all others should still be populated
        for dim in DIMS:
            if dim == "Risk Exposure":
                continue
            assert row[headers.index(dim)] != "—", f"Expected value for '{dim}'"

    def test_overall_present_even_with_missing_dim(self):
        row_data = dict(_FLAT_ROW_A)
        del row_data["strategic_fit"]
        table = _build_table([row_data])
        assert table["rows"][0][-1] == "Medium"


# ---------------------------------------------------------------------------
# TestMissingDecisionMatrix
# ---------------------------------------------------------------------------


class TestMissingDecisionMatrix:
    """When decision_matrix is absent, the method is not called; caller returns []."""

    def test_empty_matrix_returns_empty_table(self):
        # _matrix_from_decision_matrix with an empty list produces no rows
        table = _build_table([])
        assert table["rows"] == []

    def test_missing_dims_still_returns_minimal_table(self):
        # dims list empty but matrix present — just Option + Overall columns
        table = _build_table([_FLAT_ROW_A], dims=[])
        assert len(table["rows"]) == 1
        # headers: ["Option", "Overall"]
        assert table["headers"] == ["Option", "Overall"]
        assert table["rows"][0][-1] == "Medium"


# ---------------------------------------------------------------------------
# TestSportsOptionComparison
# ---------------------------------------------------------------------------


class TestSportsOptionComparison:
    """Sports engagement: all four options with all ten dimensions populated."""

    def test_all_four_options_rendered(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        assert len(table["rows"]) == 4

    def test_opt_a_values(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        headers = table["headers"]
        rows_by_label = {r[0]: r for r in table["rows"]}
        row = rows_by_label["OPT-A"]
        assert row[headers.index("Strategic Fit")] == "High"
        assert row[headers.index("Time to Value")] == "Low"
        assert row[headers.index("Overall")] == "Medium"

    def test_opt_b_values(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        headers = table["headers"]
        rows_by_label = {r[0]: r for r in table["rows"]}
        row = rows_by_label["OPT-B"]
        assert row[headers.index("Time to Value")] == "Very High"
        assert row[headers.index("Overall")] == "High"

    def test_opt_c_values(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        headers = table["headers"]
        rows_by_label = {r[0]: r for r in table["rows"]}
        row = rows_by_label["OPT-C"]
        assert row[headers.index("Strategic Fit")] == "Very High"
        assert row[headers.index("Expected Return")] == "Very High"
        assert row[headers.index("Opportunity Capture")] == "Very High"
        assert row[headers.index("Overall")] == "Very High"

    def test_opt_d_values(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        headers = table["headers"]
        rows_by_label = {r[0]: r for r in table["rows"]}
        row = rows_by_label["OPT-D"]
        assert row[headers.index("Implementation Risk")] == "Very High"
        assert row[headers.index("Assumption Strength")] == "Low"
        assert row[headers.index("Overall")] == "Medium"

    def test_title_is_strategic_option_comparison(self):
        table = _build_table(_FOUR_ROW_MATRIX)
        assert table["title"] == "Strategic Option Comparison"

    def test_no_outputs_dependency(self):
        """Confirm non-self-referential test methods don't read from outputs/."""
        # Verify _FOUR_ROW_MATRIX is a pure in-memory fixture
        for row in _FOUR_ROW_MATRIX:
            assert isinstance(row, dict)
            assert "option_id" in row
        # The fixture data itself has no filesystem path
        assert all(isinstance(v, (str, list)) for row in _FOUR_ROW_MATRIX for v in row.values())
