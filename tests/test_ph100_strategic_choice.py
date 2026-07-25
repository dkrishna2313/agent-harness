"""PH10.0 — StrategicChoice and StrategicChoiceSet unit tests.

Covers:
- StrategicChoice: fields, defaults, serialization, immutability, validator
- StrategicChoiceSet: fields, defaults, serialization, immutability, validator,
  convenience accessors
- Round-trip: StrategicChoiceSet containing StrategicChoice objects
- No behavioral change to existing pipeline objects
"""

import pytest

from functional_agents.strategy import (
    StrategicChoice,
    StrategicChoiceSet,
    StrategyCoordinator,
)


# ---------------------------------------------------------------------------
# StrategicChoice — defaults
# ---------------------------------------------------------------------------

class TestStrategicChoiceDefaults:
    def test_empty_construction(self):
        choice = StrategicChoice()
        assert choice.id == ""
        assert choice.dimension == ""
        assert choice.selected_value == ""
        assert choice.rationale == ""

    def test_list_field_defaults(self):
        choice = StrategicChoice()
        assert choice.supporting_evidence == []
        assert choice.supporting_assumptions == []
        assert choice.alternatives_considered == []

    def test_confidence_default(self):
        choice = StrategicChoice()
        assert choice.confidence == ""

    def test_requiredness_default(self):
        choice = StrategicChoice()
        assert choice.requiredness == "optional"

    def test_metadata_default(self):
        choice = StrategicChoice()
        assert choice.metadata == {}


# ---------------------------------------------------------------------------
# StrategicChoice — construction with values
# ---------------------------------------------------------------------------

class TestStrategicChoiceConstruction:
    def test_all_fields_set(self):
        choice = StrategicChoice(
            id="SC-001",
            dimension="market_entry",
            selected_value="phased",
            rationale="Reduces capital risk",
            supporting_evidence=["E001", "E002"],
            supporting_assumptions=["A001"],
            confidence="High",
            alternatives_considered=[
                {"value": "aggressive", "reason_not_selected": "too risky"}
            ],
            requiredness="required",
            metadata={"source": "decision_model"},
        )
        assert choice.id == "SC-001"
        assert choice.dimension == "market_entry"
        assert choice.selected_value == "phased"
        assert choice.rationale == "Reduces capital risk"
        assert choice.supporting_evidence == ["E001", "E002"]
        assert choice.supporting_assumptions == ["A001"]
        assert choice.confidence == "High"
        assert len(choice.alternatives_considered) == 1
        assert choice.alternatives_considered[0]["value"] == "aggressive"
        assert choice.requiredness == "required"
        assert choice.metadata["source"] == "decision_model"

    def test_requiredness_values(self):
        for val in ("required", "optional", "conditional", ""):
            choice = StrategicChoice(requiredness=val)
            assert choice.requiredness == val

    def test_invalid_requiredness_raises(self):
        with pytest.raises(Exception):
            StrategicChoice(requiredness="mandatory")


# ---------------------------------------------------------------------------
# StrategicChoice — immutability
# ---------------------------------------------------------------------------

class TestStrategicChoiceImmutability:
    def test_cannot_set_field_after_construction(self):
        choice = StrategicChoice(id="SC-001")
        with pytest.raises(Exception):
            choice.id = "SC-002"

    def test_cannot_set_confidence_after_construction(self):
        choice = StrategicChoice(confidence="High")
        with pytest.raises(Exception):
            choice.confidence = "Low"


# ---------------------------------------------------------------------------
# StrategicChoice — serialization
# ---------------------------------------------------------------------------

class TestStrategicChoiceSerialization:
    def test_to_dict_returns_dict(self):
        choice = StrategicChoice(id="SC-001", dimension="technology")
        d = choice.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "SC-001"
        assert d["dimension"] == "technology"

    def test_round_trip_empty(self):
        choice = StrategicChoice()
        restored = StrategicChoice.from_dict(choice.to_dict())
        assert restored.id == choice.id
        assert restored.dimension == choice.dimension

    def test_round_trip_full(self):
        choice = StrategicChoice(
            id="SC-002",
            dimension="capital_structure",
            selected_value="equity",
            rationale="Lower leverage risk",
            supporting_evidence=["E003"],
            confidence="Medium",
            requiredness="required",
        )
        d = choice.to_dict()
        restored = StrategicChoice.from_dict(d)
        assert restored.id == choice.id
        assert restored.dimension == choice.dimension
        assert restored.selected_value == choice.selected_value
        assert restored.supporting_evidence == choice.supporting_evidence
        assert restored.confidence == choice.confidence
        assert restored.requiredness == choice.requiredness

    def test_to_dict_contains_all_fields(self):
        choice = StrategicChoice()
        d = choice.to_dict()
        expected_keys = {
            "id", "dimension", "selected_value", "rationale",
            "supporting_evidence", "supporting_assumptions", "confidence",
            "alternatives_considered", "requiredness", "metadata",
        }
        assert expected_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# StrategicChoiceSet — defaults
# ---------------------------------------------------------------------------

class TestStrategicChoiceSetDefaults:
    def test_empty_construction(self):
        cs = StrategicChoiceSet()
        assert cs.id == ""
        assert cs.choices == []
        assert cs.overall_confidence == ""
        assert cs.internal_conflicts == []
        assert cs.completeness == 0.0
        assert cs.rationale == ""
        assert cs.metadata == {}


# ---------------------------------------------------------------------------
# StrategicChoiceSet — construction with values
# ---------------------------------------------------------------------------

class TestStrategicChoiceSetConstruction:
    def _make_choice(self, dimension: str, value: str) -> StrategicChoice:
        return StrategicChoice(
            id=f"SC-{dimension}",
            dimension=dimension,
            selected_value=value,
        )

    def test_all_fields_set(self):
        choices = [
            self._make_choice("market", "phased"),
            self._make_choice("technology", "proven"),
        ]
        cs = StrategicChoiceSet(
            id="SCS-001",
            choices=choices,
            overall_confidence="High",
            internal_conflicts=[{"choice_ids": ["SC-a", "SC-b"], "description": "conflict"}],
            completeness=0.8,
            rationale="Balanced approach",
            metadata={"run_id": "R-001"},
        )
        assert cs.id == "SCS-001"
        assert len(cs.choices) == 2
        assert cs.overall_confidence == "High"
        assert len(cs.internal_conflicts) == 1
        assert cs.completeness == 0.8
        assert cs.rationale == "Balanced approach"
        assert cs.metadata["run_id"] == "R-001"

    def test_completeness_boundary_zero(self):
        cs = StrategicChoiceSet(completeness=0.0)
        assert cs.completeness == 0.0

    def test_completeness_boundary_one(self):
        cs = StrategicChoiceSet(completeness=1.0)
        assert cs.completeness == 1.0

    def test_invalid_completeness_raises(self):
        with pytest.raises(Exception):
            StrategicChoiceSet(completeness=1.5)

    def test_invalid_completeness_negative_raises(self):
        with pytest.raises(Exception):
            StrategicChoiceSet(completeness=-0.1)


# ---------------------------------------------------------------------------
# StrategicChoiceSet — immutability
# ---------------------------------------------------------------------------

class TestStrategicChoiceSetImmutability:
    def test_cannot_set_id_after_construction(self):
        cs = StrategicChoiceSet(id="SCS-001")
        with pytest.raises(Exception):
            cs.id = "SCS-002"

    def test_cannot_set_completeness_after_construction(self):
        cs = StrategicChoiceSet(completeness=0.5)
        with pytest.raises(Exception):
            cs.completeness = 0.9


# ---------------------------------------------------------------------------
# StrategicChoiceSet — serialization
# ---------------------------------------------------------------------------

class TestStrategicChoiceSetSerialization:
    def test_to_dict_returns_dict(self):
        cs = StrategicChoiceSet(id="SCS-001")
        d = cs.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "SCS-001"

    def test_round_trip_empty(self):
        cs = StrategicChoiceSet()
        restored = StrategicChoiceSet.from_dict(cs.to_dict())
        assert restored.id == cs.id
        assert restored.completeness == cs.completeness

    def test_round_trip_with_choices(self):
        choice = StrategicChoice(
            id="SC-001",
            dimension="market_entry",
            selected_value="phased",
            confidence="High",
        )
        cs = StrategicChoiceSet(
            id="SCS-001",
            choices=[choice],
            overall_confidence="High",
            completeness=1.0,
        )
        d = cs.to_dict()
        restored = StrategicChoiceSet.from_dict(d)
        assert restored.id == cs.id
        assert len(restored.choices) == 1
        assert restored.choices[0].id == "SC-001"
        assert restored.overall_confidence == cs.overall_confidence
        assert restored.completeness == cs.completeness

    def test_to_dict_contains_all_fields(self):
        cs = StrategicChoiceSet()
        d = cs.to_dict()
        expected_keys = {
            "id", "choices", "overall_confidence", "internal_conflicts",
            "completeness", "rationale", "metadata",
        }
        assert expected_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# StrategicChoiceSet — convenience accessors
# ---------------------------------------------------------------------------

class TestStrategicChoiceSetAccessors:
    def _make_set(self) -> StrategicChoiceSet:
        return StrategicChoiceSet(
            id="SCS-001",
            choices=[
                StrategicChoice(id="SC-A", dimension="market", selected_value="phased"),
                StrategicChoice(id="SC-B", dimension="technology", selected_value="proven"),
            ],
            internal_conflicts=[{"choice_ids": ["SC-A", "SC-B"], "description": "minor"}],
        )

    def test_choice_by_dimension_found(self):
        cs = self._make_set()
        found = cs.choice_by_dimension("market")
        assert found is not None
        assert found.id == "SC-A"

    def test_choice_by_dimension_not_found(self):
        cs = self._make_set()
        assert cs.choice_by_dimension("nonexistent") is None

    def test_dimensions_covered(self):
        cs = self._make_set()
        dims = cs.dimensions_covered()
        assert "market" in dims
        assert "technology" in dims

    def test_has_conflicts_true(self):
        cs = self._make_set()
        assert cs.has_conflicts() is True

    def test_has_conflicts_false(self):
        cs = StrategicChoiceSet()
        assert cs.has_conflicts() is False


# ---------------------------------------------------------------------------
# Compatibility — existing pipeline objects unaffected
# ---------------------------------------------------------------------------

class TestCompatibility:
    def test_strategy_coordinator_still_builds(self):
        coord = StrategyCoordinator()
        assert coord._config is not None
        assert coord._plan is not None

    def test_strategic_choice_import_does_not_break_coordinator(self):
        from functional_agents.strategy import StrategicChoice, StrategicChoiceSet
        coord = StrategyCoordinator()
        assert coord._plan.framework == "executive"
