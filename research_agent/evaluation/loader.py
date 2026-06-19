"""Load J2.1 evaluation YAML files into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class QAQuestion:
    """One Q&A benchmark question loaded from eval/nvidia/ or eval/smr/."""

    question_id: str
    domain: str                            # "nvidia" | "smr"
    difficulty: str                        # "easy" | "medium" | "hard"
    question: str
    must_include: list[str] = field(default_factory=list)
    acceptable_alternatives: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    expected_topics: list[str] = field(default_factory=list)
    evaluation_tags: list[str] = field(default_factory=list)
    notes: str = ""
    source_file: str = ""


@dataclass(frozen=True)
class ContradictionCase:
    """One contradiction test case loaded from eval/contradictions/."""

    contradiction_id: str
    domain: str
    expected_result: str          # "contradiction" | "no_contradiction"
    category: str
    claim_a: str
    claim_b: str
    severity: str | None = None
    entity: str = ""
    entity_a: str = ""
    entity_b: str = ""
    scope_a: str = ""
    scope_b: str = ""
    suppression_should_fire: bool = False
    suppression_should_not_fire: bool = False
    expected_suppression_reason: str | None = None
    known_limitation: bool = False     # engine limitation — failure is expected until J2.3+
    source_file: str = ""


def load_qa_questions(eval_dir: str | Path) -> list[QAQuestion]:
    """Load all Q&A YAML files from eval/nvidia/ and eval/smr/."""

    root = Path(eval_dir)
    questions: list[QAQuestion] = []

    for subdir in ("nvidia", "smr"):
        domain_dir = root / subdir
        if not domain_dir.is_dir():
            continue
        for path in sorted(domain_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Failed to parse {path}: {exc}") from exc

            questions.append(QAQuestion(
                question_id=str(raw.get("question_id", path.stem)),
                domain=str(raw.get("domain", subdir)),
                difficulty=str(raw.get("difficulty", "medium")),
                question=str(raw.get("question", "")).strip(),
                must_include=_str_list(raw.get("must_include")),
                acceptable_alternatives=_str_list(raw.get("acceptable_alternatives")),
                must_not_include=_str_list(raw.get("must_not_include")),
                expected_topics=_str_list(raw.get("expected_topics")),
                evaluation_tags=_str_list(raw.get("evaluation_tags")),
                notes=str(raw.get("notes", "")).strip(),
                source_file=str(path),
            ))

    return questions


def load_contradiction_cases(eval_dir: str | Path) -> list[ContradictionCase]:
    """Load all YAML files from eval/contradictions/."""

    root = Path(eval_dir)
    cases: list[ContradictionCase] = []

    contra_dir = root / "contradictions"
    if not contra_dir.is_dir():
        return cases

    for path in sorted(contra_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse {path}: {exc}") from exc

        cases.append(ContradictionCase(
            contradiction_id=str(raw.get("contradiction_id", path.stem)),
            domain=str(raw.get("domain", "unknown")),
            expected_result=str(raw.get("expected_result", "no_contradiction")),
            category=str(raw.get("category", "unknown")),
            claim_a=str(raw.get("claim_a", "")).strip(),
            claim_b=str(raw.get("claim_b", "")).strip(),
            severity=raw.get("severity"),
            entity=str(raw.get("entity", "")),
            entity_a=str(raw.get("entity_a", "")),
            entity_b=str(raw.get("entity_b", "")),
            scope_a=str(raw.get("scope_a", "")),
            scope_b=str(raw.get("scope_b", "")),
            suppression_should_fire=bool(raw.get("suppression_should_fire", False)),
            suppression_should_not_fire=bool(raw.get("suppression_should_not_fire", False)),
            expected_suppression_reason=raw.get("expected_suppression_reason"),
            known_limitation=bool(raw.get("known_limitation", False)),
            source_file=str(path),
        ))

    return cases


def _str_list(value: object) -> list[str]:
    """Coerce a YAML value to a flat list of strings, stripping inline comments."""

    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            s = str(item).strip()
            # Strip YAML inline comments (# ...) that survived safe_load
            if "#" in s:
                s = s[: s.index("#")].strip()
            if s:
                result.append(s)
        return result
    return [str(value).strip()]
