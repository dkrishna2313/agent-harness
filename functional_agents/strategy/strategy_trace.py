"""StrategyTrace — canonical trace artifact for the Strategy Layer (PH11.0).

Records the full execution chain of the Strategy sub-pipeline:

    StrategyPlan
        → list[StrategicChoiceSet]
        → list[TheoryOfWinning]
        → list[TheoryEvaluation]
        → StrategySelection
        → StrategicPosition

Produced by: StrategyCoordinator.build() (stored as _trace)
Consumed by: pipeline_trace.build_canonical_trace() (exposed as "strategy_trace")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .strategic_choice_set import StrategicChoiceSet
from .strategic_position import StrategicPosition, TheoryOfWinning
from .strategy_lineage import StrategyLineageLink  # PH11.2
from .strategy_plan import StrategyPlan
from .strategy_selector import StrategySelection
from .theory_evaluation import TheoryEvaluation


class StrategyTrace(BaseModel):
    """Canonical trace of one complete Strategy Layer execution.

    Captures the full derivation chain from StrategyPlan to StrategicPosition.
    Immutable after construction.

    Validation rules enforced on construction:
      1.  choice_sets is non-empty
      2.  theories is non-empty
      3.  evaluations is non-empty
      4.  theories and evaluations have the same count (one eval per theory)
      5.  theories and choice_sets have the same count (one theory per set)
      6.  no duplicate theory_ids in theories
      7.  no duplicate theory_ids in evaluations
      8.  every evaluation.theory_id resolves to a theory (full bijection)
      9.  selection.winner_theory_id references a known theory
      10. strategic_position.theory_of_winning.theory_id == winner_theory_id
      11. selection.runner_up_theory_id, when present, references a known theory
      12. selection.runner_up_theory_id must differ from winner_theory_id
      Rule 14 applies always (PH11.2a):
      14. every theory.source_choice_set_id resolves to a known choice_set
      Rules 15-18 apply only when lineage is non-empty:
      15. no duplicate lineage links (by full composite key)
      16. lineage targets of type theory_of_winning / strategic_choice_set /
          theory_evaluation resolve to known trace members
      17. no "unknown" sentinel in any lineage source_id or target_id
      18. metadata["research_id"] equals the research_object lineage link's source_id
    """

    trace_id: str
    created_at: str
    plan: StrategyPlan
    choice_sets: list[StrategicChoiceSet] = Field(default_factory=list)
    theories: list[TheoryOfWinning] = Field(default_factory=list)
    evaluations: list[TheoryEvaluation] = Field(default_factory=list)
    selection: StrategySelection
    strategic_position: StrategicPosition
    lineage: list[StrategyLineageLink] = Field(default_factory=list)  # PH11.2
    metadata: dict[str, Any] = Field(default_factory=dict)

    # PH12.1a — structured audit fields (backward-compatible optional)
    theory_option_mappings: list[dict[str, Any]] = Field(default_factory=list)
    constraint_results: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    alignment: dict[str, Any] = Field(default_factory=dict)
    saturation: dict[str, Any] = Field(default_factory=dict)

    # PH12.2 — theory content lineage (backward-compatible optional)
    theory_content: list[dict[str, Any]] = Field(default_factory=list)
    theory_content_lineage: dict[str, Any] = Field(default_factory=dict)
    theory_content_coverage: dict[str, Any] = Field(default_factory=dict)
    theory_content_confidence: dict[str, Any] = Field(default_factory=dict)
    theory_differentiation: dict[str, Any] = Field(default_factory=dict)
    content_homogenization: dict[str, Any] = Field(default_factory=dict)
    content_fallbacks: list[dict[str, Any]] = Field(default_factory=list)

    # PH12.2b — discrimination fields (backward-compatible optional)
    # theory_discrimination: {theory_id: {"distinctive_*_ids": [...], "shared_*_ids": [...], ...}}
    theory_discrimination: dict[str, Any] = Field(default_factory=dict)
    # content_differentiation_state: {"state": "none"|"partial"|"substantial"|"full", ...}
    content_differentiation_state: dict[str, Any] = Field(default_factory=dict)

    # PH12.2a — resolved strategy configuration snapshot
    strategy_configuration: dict[str, Any] = Field(default_factory=dict)
    strategy_config_fingerprint: str = ""

    model_config = {"frozen": True, "extra": "allow"}

    @model_validator(mode="after")
    def _validate_chain_consistency(self) -> "StrategyTrace":
        theories = self.theories
        evaluations = self.evaluations
        choice_sets = self.choice_sets
        selection = self.selection
        position = self.strategic_position

        # Rules 1–3: no empty collections
        if not choice_sets:
            raise ValueError("StrategyTrace: choice_sets must not be empty.")
        if not theories:
            raise ValueError("StrategyTrace: theories must not be empty.")
        if not evaluations:
            raise ValueError("StrategyTrace: evaluations must not be empty.")

        # Rule 4: one evaluation per theory
        if len(theories) != len(evaluations):
            raise ValueError(
                f"StrategyTrace: theories ({len(theories)}) and "
                f"evaluations ({len(evaluations)}) must have the same count."
            )

        # Rule 5: one theory per choice_set (traceability chain)
        if len(theories) != len(choice_sets):
            raise ValueError(
                f"StrategyTrace: theories ({len(theories)}) and "
                f"choice_sets ({len(choice_sets)}) must have the same count."
            )

        # Rule 6: no duplicate theory_ids in theories
        seen_t: set[str] = set()
        for t in theories:
            if t.theory_id in seen_t:
                raise ValueError(
                    f"StrategyTrace: duplicate theory_id={t.theory_id!r} in theories."
                )
            seen_t.add(t.theory_id)

        # Rule 7: no duplicate theory_ids in evaluations
        seen_ev: set[str] = set()
        for ev in evaluations:
            if ev.theory_id in seen_ev:
                raise ValueError(
                    f"StrategyTrace: duplicate theory_id={ev.theory_id!r} in evaluations."
                )
            seen_ev.add(ev.theory_id)

        # Rule 8: every evaluation theory_id resolves to a theory
        unresolved = seen_ev - seen_t
        if unresolved:
            raise ValueError(
                f"StrategyTrace: evaluation theory_id(s) {sorted(unresolved)!r} "
                f"have no matching theory."
            )

        # Rule 9: selection.winner_theory_id references a known theory
        if selection.winner_theory_id not in seen_t:
            raise ValueError(
                f"StrategyTrace: selection.winner_theory_id={selection.winner_theory_id!r} "
                f"not found in theories. Available: {sorted(seen_t)}"
            )

        # Rule 10: StrategicPosition.theory_of_winning.theory_id == winner
        tow = position.theory_of_winning
        if hasattr(tow, "theory_id") and tow.theory_id != selection.winner_theory_id:
            raise ValueError(
                f"StrategyTrace: strategic_position.theory_of_winning.theory_id="
                f"{tow.theory_id!r} does not match "
                f"selection.winner_theory_id={selection.winner_theory_id!r}."
            )

        # Rules 11–12: runner-up identity (only when runner_up_theory_id is present)
        runner_up = selection.runner_up_theory_id
        if runner_up:  # None or "" → no runner-up, skip
            if runner_up not in seen_t:
                raise ValueError(
                    f"StrategyTrace: selection.runner_up_theory_id={runner_up!r} "
                    f"not found in theories. Available: {sorted(seen_t)}"
                )
            if runner_up == selection.winner_theory_id:
                raise ValueError(
                    "StrategyTrace: runner-up theory ID must differ from winner theory ID."
                )

        # Rule 14 (always): source_choice_set_id must resolve to a known choice_set
        cs_ids: set[str] = {cs.id for cs in choice_sets}
        for t in theories:
            scid = t.source_choice_set_id  # required, non-empty — enforced by TheoryOfWinning
            if scid not in cs_ids:
                raise ValueError(
                    f"StrategyTrace: theory_id={t.theory_id!r} "
                    f"source_choice_set_id={scid!r} not found in choice_sets. "
                    f"Available: {sorted(cs_ids)}"
                )

        # Rules 15-18: lineage integrity (applied only when lineage is provided)
        if self.lineage:
            # Rule 15: no duplicate lineage links
            seen_links: set[tuple] = set()
            for link in self.lineage:
                key = (
                    link.source_type, link.source_id,
                    link.target_type, link.target_id,
                    link.relationship,
                )
                if key in seen_links:
                    raise ValueError(
                        f"StrategyTrace: duplicate lineage link {key!r}."
                    )
                seen_links.add(key)

            # Rule 16: artifact references in lineage resolve to known trace members
            for link in self.lineage:
                if link.target_type == "theory_of_winning" and link.target_id not in seen_t:
                    raise ValueError(
                        f"StrategyTrace: lineage target theory_of_winning "
                        f"id={link.target_id!r} not found in theories."
                    )
                if link.target_type == "strategic_choice_set" and link.target_id not in cs_ids:
                    raise ValueError(
                        f"StrategyTrace: lineage target strategic_choice_set "
                        f"id={link.target_id!r} not found in choice_sets."
                    )
                if link.target_type == "theory_evaluation" and link.target_id not in seen_ev:
                    raise ValueError(
                        f"StrategyTrace: lineage target theory_evaluation "
                        f"id={link.target_id!r} not found in evaluations."
                    )

            # Rule 17: no "unknown" sentinel identifiers in lineage
            for link in self.lineage:
                if link.source_id == "unknown" or link.target_id == "unknown":
                    raise ValueError(
                        f"StrategyTrace: lineage link contains forbidden sentinel "
                        f"'unknown' (source_id={link.source_id!r}, "
                        f"target_id={link.target_id!r})."
                    )

            # Rule 18: metadata["research_id"] must equal the research_object link's source_id
            ro_links = [lk for lk in self.lineage if lk.source_type == "research_object"]
            if ro_links:
                meta_rid = self.metadata.get("research_id", "")
                if meta_rid != ro_links[0].source_id:
                    raise ValueError(
                        f"StrategyTrace: metadata['research_id']={meta_rid!r} does not match "
                        f"lineage research_object source_id={ro_links[0].source_id!r}."
                    )

        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyTrace":
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Persistence helper (PH11.1)
# ---------------------------------------------------------------------------

def write_strategy_trace(trace: "StrategyTrace", out_dir: str | Path) -> Path:
    """Write the StrategyTrace as ``strategy.trace.json`` in *out_dir*.

    Uses the same formatting conventions as the canonical pipeline trace:
    ``json.dumps(data, indent=2, default=str)`` with UTF-8 encoding.

    The file is written only after ``model_dump(mode="json")`` succeeds,
    so a serialization failure raises before touching the filesystem.
    The directory is created when it does not already exist.

    Returns the path written.
    """
    out_path = Path(out_dir) / "strategy.trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = trace.model_dump(mode="json")           # serialise before touching disk
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out_path


def write_artifact_index(
    trace: "StrategyTrace",
    strategy_trace_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Write or update ``artifact.index.json`` in *out_dir* with the StrategyTrace entry.

    Merges with any pre-existing index: stale ``strategy_trace`` entries are
    removed and the fresh entry is appended. The file is always overwritten in
    full to avoid partial-update corruption.

    Returns the path written (``{out_dir}/artifact.index.json``).
    """
    idx_path = Path(out_dir) / "artifact.index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    existing_entries: list[dict] = []
    if idx_path.exists():
        try:
            raw = json.loads(idx_path.read_text(encoding="utf-8"))
            existing_entries = [
                e for e in raw.get("entries", [])
                if e.get("artifact_type") != "strategy_trace"
            ]
        except Exception:
            existing_entries = []

    entry: dict = {
        "artifact_type": "strategy_trace",
        "artifact_id": trace.trace_id,
        "trace_id": trace.trace_id,
        "path": str(strategy_trace_path),
        "mime_type": "application/json",
        "framework": trace.metadata.get("framework", ""),
        "plan_id": trace.metadata.get("plan_id", ""),
        "selected_theory_id": trace.metadata.get("selected_theory_id", ""),
        "research_id": trace.metadata.get("research_id", ""),
        "created_at": trace.created_at,
        "lineage_count": len(trace.lineage),
        "status": "generated",
    }

    index: dict = {
        "schema_version": "ph11.2-strategy-artifact-index-v1",
        "updated_at": trace.created_at,
        "entries": existing_entries + [entry],
    }
    idx_path.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    return idx_path
