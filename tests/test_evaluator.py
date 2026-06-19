from pathlib import Path

from dc_power_agent.evaluator import classify_question_topics, evaluate_memo
from dc_power_agent.schemas import EvidenceItem, ResearchMemo, SourceDocument, assign_evidence_ids


def test_evaluator_warns_when_no_sources_or_content():
    memo = ResearchMemo(
        title="Empty",
        question="Question?",
        executive_summary="",
    )

    warnings = evaluate_memo(memo, [], mock_llm=True)
    codes = {warning.code for warning in warnings}

    assert "mock_llm" in codes
    assert "no_sources" in codes
    assert "zero_documents_loaded" in codes
    assert "few_documents_loaded" in codes
    assert "missing_executive_summary" in codes
    assert "missing_confirmed_facts" in codes


def test_topic_classifier_detects_power_terms():
    topics = classify_question_topics("How do UPS, BBU, PDU, and grid limits affect power?")

    assert "power" in topics
    assert "backup/resiliency" in topics


def test_topic_classifier_detects_cooling_terms():
    topics = classify_question_topics("What liquid cooling, CDU, thermal, and chilled water changes matter?")

    assert "cooling" in topics


def test_topic_classifier_detects_networking_terms():
    topics = classify_question_topics("Explain NVLink, InfiniBand, Ethernet, Spectrum, and ConnectX networking.")

    assert "networking" in topics


def test_mock_warning_only_appears_in_mock_mode():
    evidence = [
        _evidence("a.md", "Rubin compute architecture affects system planning.", "architecture"),
        _evidence("a.md", "Rack architecture affects power planning.", "rack architecture"),
        _evidence("a.md", "Power distribution affects rack design.", "power"),
        _evidence("b.md", "Cooling infrastructure affects deployment planning.", "cooling"),
        _evidence("b.md", "Networking architecture affects cluster scale.", "networking"),
        _evidence("b.md", "NVLink networking affects rack-scale communication.", "networking"),
        _evidence("c.md", "Architecture constraints affect AI data center design.", "architecture"),
        _evidence("c.md", "Power smoothing affects facility integration.", "power"),
        _evidence("c.md", "Liquid cooling affects thermal planning.", "cooling"),
        _evidence("c.md", "NVL72 rack architecture affects deployment shape.", "rack architecture"),
    ]
    memo = _complete_memo(evidence)
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    real_mode_codes = {warning.code for warning in evaluate_memo(memo, documents, mock_llm=False)}
    mock_mode_codes = {warning.code for warning in evaluate_memo(memo, documents, mock_llm=True)}

    assert "mock_llm" not in real_mode_codes
    assert "mock_llm" in mock_mode_codes


def test_evaluator_warns_on_missing_evidence():
    memo = _complete_memo([])
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "no_evidence" in codes


def test_evaluator_warns_on_empty_evidence_snippets():
    evidence = [
        _evidence("a.md", ""),
        _evidence("b.md", "Power systems require validation."),
        _evidence("c.md", "Cooling systems require validation."),
    ]
    memo = _complete_memo(evidence)
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "empty_evidence_snippet" in codes


def test_evaluator_passes_when_three_documents_have_evidence():
    evidence = [
        _evidence("a.md", "Rubin compute architecture affects system planning.", "architecture"),
        _evidence("a.md", "Rack architecture affects power planning.", "rack architecture"),
        _evidence("a.md", "Power distribution affects rack design.", "power"),
        _evidence("b.md", "Cooling infrastructure affects deployment planning.", "cooling"),
        _evidence("b.md", "Networking architecture affects cluster scale.", "networking"),
        _evidence("b.md", "NVLink networking affects rack-scale communication.", "networking"),
        _evidence("c.md", "Architecture constraints affect AI data center design.", "architecture"),
        _evidence("c.md", "Power smoothing affects facility integration.", "power"),
        _evidence("c.md", "Liquid cooling affects thermal planning.", "cooling"),
        _evidence("c.md", "NVL72 rack architecture affects deployment shape.", "rack architecture"),
    ]
    memo = _complete_memo(evidence)
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)

    assert warnings == []


def test_evaluator_warns_on_low_evidence_count():
    evidence = [
        _evidence("a.md", "Rubin compute architecture affects system planning.", "architecture"),
        _evidence("b.md", "Cooling infrastructure affects deployment planning.", "cooling"),
        _evidence("c.md", "Networking architecture affects cluster scale.", "networking"),
    ]
    memo = _complete_memo(evidence)
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "low_evidence_count" in codes


def test_evaluator_warns_on_low_high_quality_evidence_count():
    evidence = [
        _evidence("a.md", "Rubin compute architecture affects system planning.", "architecture"),
        _evidence("a.md", "Rack architecture affects power planning.", "rack architecture"),
        _evidence("a.md", "Power distribution affects rack design.", "power"),
        _evidence("b.md", "Cooling infrastructure affects deployment planning.", "cooling"),
        _evidence("b.md", "Networking architecture affects cluster scale.", "networking"),
        _evidence("b.md", "NVLink networking affects rack-scale communication.", "networking"),
        _evidence("c.md", "Architecture constraints affect AI data center design.", "architecture"),
        _evidence("c.md", "Power smoothing affects facility integration.", "power"),
        _evidence("c.md", "Liquid cooling affects thermal planning.", "cooling"),
        _evidence("c.md", "NVL72 rack architecture affects deployment shape.", "rack architecture"),
    ]
    low_quality = [
        item.model_copy(update={"overall_score": 3.0})
        for item in evidence
    ]
    memo = _complete_memo(low_quality)
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "low_high_quality_evidence_count" in codes


def test_evaluator_warns_when_required_topic_section_is_missing():
    evidence = _full_evidence()
    memo = _complete_memo(evidence).model_copy(
        update={
            "question": "Explain cooling and thermal implications for Rubin.",
            "cooling_implications": [],
        }
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "missing_cooling_implications" in codes


def test_evaluator_does_not_warn_when_irrelevant_topic_absent():
    evidence = _full_evidence()
    memo = _complete_memo(evidence).model_copy(
        update={
            "question": "Explain power and grid implications for Rubin.",
            "networking_implications": [],
            "rack_architecture_implications": [],
        }
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "missing_networking_implications" not in codes
    assert "missing_rack_architecture_implications" not in codes


def test_evaluator_warns_on_missing_citations():
    evidence = _full_evidence()
    memo = _complete_memo(evidence).model_copy(
        update={
            "question": "Explain power and cooling implications for Rubin.",
            "confirmed_facts": ["Rubin is a rack-scale platform."],
            "power_implications": ["Power planning matters."],
            "cooling_implications": ["Cooling planning matters."],
        }
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "missing_confirmed_fact_citations" in codes
    assert "missing_power_citations" in codes
    assert "missing_cooling_citations" in codes


def test_cooling_section_with_citation_passes_evaluator():
    """Cooling Implications section with a citation should not produce a citation warning."""
    evidence = assign_evidence_ids(_full_evidence())
    citation = _citation(evidence[0])
    memo = ResearchMemo(
        title="Memo",
        question="What are the cooling implications of Rubin NVL72 racks?",
        executive_summary="Summary.",
        confirmed_facts=[f"Fact. {citation}"],
        inferences=["Inference."],
        power_implications=[f"Power implication. {citation}"],
        cooling_implications=[f"Cooling implication. {citation}"],
        networking_implications=[f"Networking implication. {citation}"],
        rack_architecture_implications=[f"Rack implication. {citation}"],
        open_questions=["Question."],
        source_notes=evidence,
        evidence=evidence,
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "missing_cooling_citations" not in codes


def test_rack_architecture_section_with_citation_passes_evaluator():
    """Rack Architecture Implications section with a citation should not produce a citation warning."""
    evidence = assign_evidence_ids(_full_evidence())
    citation = _citation(evidence[0])
    memo = ResearchMemo(
        title="Memo",
        question="What are the rack architecture implications of Rubin NVL72?",
        executive_summary="Summary.",
        confirmed_facts=[f"Fact. {citation}"],
        inferences=["Inference."],
        power_implications=[f"Power implication. {citation}"],
        cooling_implications=[f"Cooling implication. {citation}"],
        networking_implications=[f"Networking implication. {citation}"],
        rack_architecture_implications=[f"Rack implication. {citation}"],
        open_questions=["Question."],
        source_notes=evidence,
        evidence=evidence,
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "missing_rack_architecture_citations" not in codes


def test_evaluator_recognizes_cooling_and_rack_citations_when_present():
    """Evaluator must not warn on cooling or rack architecture citations when both are present."""
    evidence = assign_evidence_ids(_full_evidence())
    citation = _citation(evidence[0])
    memo = ResearchMemo(
        title="Memo",
        question="What are the DC power and cooling implications of NVIDIA Rubin NVL72 racks?",
        executive_summary="Summary.",
        confirmed_facts=[f"Fact. {citation}"],
        inferences=["Inference."],
        power_implications=[f"Power implication. {citation}"],
        cooling_implications=[f"Cooling implication. {citation}"],
        networking_implications=[f"Networking implication. {citation}"],
        rack_architecture_implications=[f"Rack implication. {citation}"],
        open_questions=["Question."],
        source_notes=evidence,
        evidence=evidence,
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "missing_cooling_citations" not in codes
    assert "missing_rack_architecture_citations" not in codes


def test_evaluator_warns_on_unknown_evidence_ids():
    evidence = _full_evidence()
    memo = _complete_memo(evidence).model_copy(
        update={
            "confirmed_facts": ["Rubin is a rack-scale platform. [Source: a.md, Evidence: E999]"],
        }
    )
    documents = [_document("a.md"), _document("b.md"), _document("c.md")]

    warnings = evaluate_memo(memo, documents)
    codes = {warning.code for warning in warnings}

    assert "unknown_evidence_citation" in codes


def _complete_memo(evidence: list[EvidenceItem]) -> ResearchMemo:
    evidence = assign_evidence_ids(evidence)
    citation = _citation(evidence[0]) if evidence else ""
    return ResearchMemo(
        title="Memo",
        question="Explain Rubin infrastructure",
        executive_summary="Summary.",
        confirmed_facts=[f"Fact. {citation}".strip()],
        inferences=["Inference."],
        power_implications=[f"Power implication. {citation}".strip()],
        cooling_implications=[f"Cooling implication. {citation}".strip()],
        networking_implications=[f"Networking implication. {citation}".strip()],
        rack_architecture_implications=[f"Rack implication. {citation}".strip()],
        open_questions=["Question."],
        source_notes=evidence,
        evidence=evidence,
    )


def _document(name: str) -> SourceDocument:
    return SourceDocument(
        path=Path(name),
        title=Path(name).stem,
        extension=Path(name).suffix,
        text="Source text about power, cooling, and networking.",
    )


def _evidence(source_document: str, snippet: str, category: str = "power") -> EvidenceItem:
    return EvidenceItem(
        claim="Infrastructure claim.",
        source_document=source_document,
        evidence_snippet=snippet,
        category=category,
        relevance="Relevant to the user question.",
        confidence="medium",
        relevance_score=4,
        source_quality_score=4,
        specificity_score=4,
        overall_score=4.0,
    )


def _citation(item: EvidenceItem) -> str:
    return f"[Source: {item.source_document}, Evidence: {item.evidence_id}]"


def test_citation_regex_matches_filename_with_commas():
    """_CITATION_RE must match source names that contain commas (e.g. real NVIDIA PDF titles)."""
    from dc_power_agent.evaluator import _CITATION_RE  # type: ignore[attr-defined]

    comma_filename = (
        "Inside the NVIDIA Vera Rubin Platform_ Six New Chips, One AI Supercomputer "
        "_ NVIDIA Technical Blog.pdf"
    )
    citation_str = f"[Source: {comma_filename}, Evidence: E001]"
    assert _CITATION_RE.search(citation_str), (
        "Regex must match citations whose source filename contains commas"
    )
    assert _CITATION_RE.search(citation_str).group(1) == "E001"


def test_cooling_section_citation_with_comma_filename_passes_evaluator():
    """Evaluator must not warn when cooling section has a citation with a comma in the filename."""
    comma_doc = "Six New Chips, One AI Supercomputer _ NVIDIA Technical Blog.pdf"
    ev = assign_evidence_ids([
        _evidence(comma_doc, "Liquid cooling thermal management.", "cooling"),
        _evidence(comma_doc, "Power distribution analysis.", "power"),
        _evidence("b.md", "Networking cluster architecture.", "networking"),
        _evidence("b.md", "Rack architecture deployment.", "rack architecture"),
        _evidence("c.md", "Additional power analysis.", "power"),
    ])
    citation = _citation(ev[0])
    question = "What are the DC power and cooling implications of NVIDIA Rubin NVL72 racks?"
    memo = ResearchMemo(
        title="Memo",
        question=question,
        executive_summary="Summary.",
        confirmed_facts=[f"Fact. {citation}"],
        inferences=["Inference."],
        power_implications=[f"Power implication. {_citation(ev[1])}"],
        cooling_implications=[f"Cooling implication. {citation}"],
        networking_implications=[f"Networking implication. {_citation(ev[2])}"],
        rack_architecture_implications=[f"Rack implication. {_citation(ev[3])}"],
        open_questions=["Question."],
        source_notes=ev,
        evidence=ev,
    )
    docs = [_document(comma_doc), _document("b.md"), _document("c.md")]
    warnings = evaluate_memo(memo, docs)
    codes = {w.code for w in warnings}
    assert "missing_cooling_citations" not in codes, (
        f"Unexpected cooling citation warning. Codes: {codes}"
    )
    assert "missing_rack_architecture_citations" not in codes, (
        f"Unexpected rack architecture citation warning. Codes: {codes}"
    )


def test_evaluator_detects_missing_citations_in_cooling_and_rack_sections():
    """Evaluator warns when cooling/rack sections exist but have no citations."""
    question = "What are the DC power and cooling implications of NVIDIA Rubin NVL72 racks?"
    ev = assign_evidence_ids([
        _evidence("a.md", "Cooling infrastructure thermal analysis.", "cooling"),
        _evidence("b.md", "Power distribution planning.", "power"),
        _evidence("c.md", "Rack architecture deployment.", "rack architecture"),
    ])
    citation = _citation(ev[0])
    memo = ResearchMemo(
        title="Memo",
        question=question,
        executive_summary="Summary.",
        confirmed_facts=[f"Fact. {citation}"],
        inferences=["Inference."],
        power_implications=[f"Power implication. {_citation(ev[1])}"],
        cooling_implications=["Cooling implication without any citation."],
        networking_implications=[f"Networking implication. {citation}"],
        rack_architecture_implications=["Rack implication without any citation."],
        open_questions=["Question."],
        source_notes=ev,
        evidence=ev,
    )
    docs = [_document("a.md"), _document("b.md"), _document("c.md")]
    warnings = evaluate_memo(memo, docs)
    codes = {w.code for w in warnings}
    assert "missing_cooling_citations" in codes
    assert "missing_rack_architecture_citations" in codes


def _full_evidence() -> list[EvidenceItem]:
    return [
        _evidence("a.md", "Rubin compute architecture affects system planning.", "architecture"),
        _evidence("a.md", "Rack architecture affects power planning.", "rack architecture"),
        _evidence("a.md", "Power distribution affects rack design.", "power"),
        _evidence("b.md", "Cooling infrastructure affects deployment planning.", "cooling"),
        _evidence("b.md", "Networking architecture affects cluster scale.", "networking"),
        _evidence("b.md", "NVLink networking affects rack-scale communication.", "networking"),
        _evidence("c.md", "Architecture constraints affect AI data center design.", "architecture"),
        _evidence("c.md", "Power smoothing affects facility integration.", "power"),
        _evidence("c.md", "Liquid cooling affects thermal planning.", "cooling"),
        _evidence("c.md", "NVL72 rack architecture affects deployment shape.", "rack architecture"),
    ]
