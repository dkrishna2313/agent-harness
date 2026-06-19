import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.agent import DcPowerAgent
from research_agent.claude_client import ClaudeClient, parse_or_repair_json
from research_agent.schemas import SourceDocument
from research_agent.trace import build_trace


def test_claude_client_successful_structured_responses():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_client(
            [
                {
                    "research_questions": ["What is Rubin?"],
                    "key_topics": ["architecture", "power"],
                    "source_priorities": ["rubin.md"],
                },
                {
                    "evidence_items": [
                        {
                            "claim": "Rubin uses rack-scale architecture.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "Rubin uses rack-scale architecture.",
                            "category": "rack architecture",
                            "relevance": "Relevant to Rubin infrastructure.",
                            "confidence": "high",
                        }
                    ]
                },
                {
                    "executive_summary": "Claude summary.",
                    "confirmed_facts": ["Fact."],
                    "inferences": ["Inference."],
                    "power_implications": ["Power."],
                    "cooling_implications": ["Cooling."],
                    "networking_implications": ["Networking."],
                    "rack_architecture_implications": ["Rack."],
                    "open_questions": ["Question."],
                },
            ]
        ),
    )
    documents = [_document()]

    plan = client.create_research_plan("Explain Rubin", documents)
    evidence = client.extract_evidence("Explain Rubin", documents)
    memo = client.synthesize_memo("Explain Rubin", evidence)

    assert plan.key_topics == ["architecture", "power"]
    assert evidence[0].evidence_id == "E001"
    assert evidence[0].category == "rack architecture"
    assert memo.executive_summary == "Claude summary."
    assert memo.networking_implications == ["Networking."]
    assert len(client.call_traces) == 3
    assert all(trace.success for trace in client.call_traces)
    assert client.call_traces[0].token_usage == {"input_tokens": 10, "output_tokens": 20}


def test_claude_client_failure_records_trace():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_failing_client(RuntimeError("network unavailable")),
    )

    with pytest.raises(RuntimeError):
        client.extract_evidence("Explain Rubin", [_document()])

    assert client.call_traces
    assert client.call_traces[0].success is False
    assert "network unavailable" in client.call_traces[0].error


def test_agent_adds_warning_when_claude_fails():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_failing_client(RuntimeError("model unavailable")),
    )

    memo = DcPowerAgent(client=client).analyze("Explain Rubin", [_document()])

    assert any("Claude warning:" in warning for warning in memo.evaluation_warnings)
    assert memo.claude_response_success is False
    assert memo.evidence


def test_trace_includes_claude_metadata():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_client(
            [
                {
                    "research_questions": ["What is Rubin?"],
                    "key_topics": ["architecture"],
                    "source_priorities": ["rubin.md"],
                },
                {
                    "evidence_items": [
                        {
                            "claim": "Rubin uses rack-scale architecture.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "Rubin uses rack-scale architecture.",
                            "category": "rack architecture",
                            "relevance": "Relevant to Rubin infrastructure.",
                            "confidence": "high",
                        }
                    ]
                },
                {
                    "executive_summary": "Claude summary.",
                    "confirmed_facts": ["Fact."],
                    "inferences": ["Inference."],
                    "power_implications": ["Power."],
                    "cooling_implications": ["Cooling."],
                    "networking_implications": ["Networking."],
                    "rack_architecture_implications": ["Rack."],
                    "open_questions": ["Question."],
                },
            ]
        ),
    )
    documents = [_document()]
    memo = DcPowerAgent(client=client).analyze("Explain Rubin", documents)

    trace = build_trace(
        question="Explain Rubin",
        source_directory="sources",
        output_path="outputs/rubin.md",
        documents=documents,
        memo=memo,
        mock_mode=False,
    )

    assert trace["model_name"] == "test-sonnet"
    assert trace["question_topics_detected"] == []
    assert trace["claude_request_timestamp"]
    assert trace["claude_response_success"] is True
    assert trace["claude_token_usage"] == {"input_tokens": 30, "output_tokens": 60}
    assert len(trace["claude_calls"]) == 3
    assert trace["evidence_items"][0]["evidence_id"] == "E001"


def test_parse_json_inside_markdown_fence():
    raw = """```json
{"research_questions":["What is Rubin?"],"key_topics":["architecture"],"source_priorities":["rubin.md"]}
```"""

    parsed = parse_or_repair_json(raw, "research_plan", {})

    assert parsed["key_topics"] == ["architecture"]


def test_parse_json_with_leading_prose():
    raw = """Here is the JSON:

{"research_questions":["What is Rubin?"],"key_topics":["power"],"source_priorities":["rubin.md"]}
"""

    parsed = parse_or_repair_json(raw, "research_plan", {})

    assert parsed["key_topics"] == ["power"]


def test_malformed_json_repair_success():
    raw = '{"research_questions":["What is Rubin?],"key_topics":["power"],"source_priorities":["rubin.md"]}'

    parsed = parse_or_repair_json(
        raw,
        "research_plan",
        {
            "repair": lambda _prompt: (
                '{"research_questions":["What is Rubin?"],'
                '"key_topics":["power"],'
                '"source_priorities":["rubin.md"]}'
            )
        },
    )

    assert parsed["research_questions"] == ["What is Rubin?"]


def test_malformed_json_repair_failure_produces_one_warning_only():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_text_client(
            [
                '{"research_questions":["What is Rubin?"],"key_topics":["architecture"],"source_priorities":["rubin.md"]}',
                '{"evidence_items":[{"claim":"broken"',
                '{"evidence_items":[{"claim":"still broken"',
                {
                    "executive_summary": "Claude summary.",
                    "confirmed_facts": ["Fact. [Source: rubin.md, Evidence: E001]"],
                    "inferences": ["Inference."],
                    "power_implications": ["Power. [Source: rubin.md, Evidence: E001]"],
                    "cooling_implications": ["Cooling. [Source: rubin.md, Evidence: E001]"],
                    "networking_implications": ["Networking. [Source: rubin.md, Evidence: E001]"],
                    "rack_architecture_implications": ["Rack. [Source: rubin.md, Evidence: E001]"],
                    "open_questions": ["Question."],
                },
            ]
        ),
    )

    memo = DcPowerAgent(client=client).analyze("Explain Rubin", [_document()])
    claude_warnings = [
        warning
        for warning in memo.evaluation_warnings
        if warning.startswith("Claude warning: evidence extraction failed")
    ]

    assert len(claude_warnings) == 1
    assert memo.evidence


def test_fallback_evidence_generation_works_after_claude_parse_failure():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_text_client(
            [
                '{"research_questions":["What is Rubin?"],"key_topics":["architecture"],"source_priorities":["rubin.md"]}',
                '{"evidence_items":[{"claim":"broken"',
                '{"evidence_items":[{"claim":"still broken"',
                {
                    "executive_summary": "Claude summary.",
                    "confirmed_facts": ["Fact. [Source: rubin.md, Evidence: E001]"],
                    "inferences": ["Inference."],
                    "power_implications": ["Power. [Source: rubin.md, Evidence: E001]"],
                    "cooling_implications": ["Cooling. [Source: rubin.md, Evidence: E001]"],
                    "networking_implications": ["Networking. [Source: rubin.md, Evidence: E001]"],
                    "rack_architecture_implications": ["Rack. [Source: rubin.md, Evidence: E001]"],
                    "open_questions": ["Question."],
                },
            ]
        ),
    )

    memo = DcPowerAgent(client=client).analyze("Explain Rubin", [_document()])

    assert memo.evidence
    assert memo.evidence[0].evidence_id == "E001"
    assert any(item.source_document == "rubin.md" for item in memo.evidence)


def test_memo_synthesis_json_failure_falls_back_to_valid_memo():
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_text_client(
            [
                '{"research_questions":["What is Rubin?"],"key_topics":["architecture"],"source_priorities":["rubin.md"]}',
                {
                    "evidence_items": [
                        {
                            "claim": "Rubin uses rack-scale architecture.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "Rubin uses rack-scale architecture.",
                            "category": "rack architecture",
                            "relevance": "Relevant to Rubin infrastructure.",
                            "confidence": "high",
                        }
                    ]
                },
                '{"executive_summary":"broken"',
                '{"executive_summary":"still broken"',
            ]
        ),
    )

    memo = DcPowerAgent(client=client).analyze("Explain Rubin", [_document()])
    memo_warnings = [
        warning
        for warning in memo.evaluation_warnings
        if warning.startswith("Claude warning: memo synthesis failed")
    ]

    assert len(memo_warnings) == 1
    assert memo.executive_summary
    assert memo.source_notes
    assert memo.source_notes[0].evidence_id == "E001"


def test_extract_evidence_truncated_response_raises_and_triggers_fallback():
    """Regression: stop_reason=max_tokens must raise, not silently return empty evidence."""
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_truncated_client(
            [
                {
                    "research_questions": ["What is Rubin?"],
                    "key_topics": ["power"],
                    "source_priorities": ["rubin.md"],
                },
            ]
        ),
    )

    # Direct extract_evidence call must raise RuntimeError.
    with pytest.raises(RuntimeError, match="stop_reason=max_tokens"):
        client.extract_evidence("Explain Rubin", [_document()])

    # The call trace must record failure so callers know the truncation happened.
    assert client.call_traces
    assert client.call_traces[0].success is False


def test_truncated_evidence_response_falls_back_to_deterministic_extraction():
    """Regression: when extract_evidence is truncated, the agent falls back and still produces evidence."""
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_client_with_truncated_evidence(
            research_plan={
                "research_questions": ["What is Rubin?"],
                "key_topics": ["power"],
                "source_priorities": ["rubin.md"],
            },
            memo={
                "executive_summary": "Claude summary.",
                "confirmed_facts": ["Fact."],
                "inferences": ["Inference."],
                "power_implications": ["Power."],
                "cooling_implications": ["Cooling."],
                "networking_implications": ["Networking."],
                "rack_architecture_implications": ["Rack."],
                "open_questions": ["Question."],
            },
        ),
    )

    memo = DcPowerAgent(client=client).analyze("Explain Rubin power", [_document()])

    # Evidence must come from the deterministic fallback.
    assert memo.evidence, "Fallback extraction must produce evidence items"
    assert all(item.evidence_id for item in memo.evidence), "All evidence items must have IDs"
    # A warning about the truncated extraction must be present.
    assert any("evidence extraction failed" in w for w in memo.evaluation_warnings)


def test_extract_evidence_produces_evidence_items_and_assigns_ids():
    """Regression: a successful extract_evidence call returns EvidenceItems with sequential IDs."""
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_client(
            [
                {
                    "evidence_items": [
                        {
                            "claim": "Rubin NVL72 rack power reaches 120kW.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "Rubin NVL72 rack power reaches 120kW with liquid cooling.",
                            "category": "power",
                            "relevance": "Directly relevant.",
                            "confidence": "high",
                        },
                        {
                            "claim": "Rubin uses NVLink 6 for scale-up networking.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "NVLink 6 switch enables rack-scale connectivity.",
                            "category": "networking",
                            "relevance": "Relevant infrastructure context.",
                            "confidence": "high",
                        },
                    ]
                }
            ]
        ),
    )

    evidence = client.extract_evidence("Explain Rubin power", [_document()])

    assert len(evidence) == 2
    assert evidence[0].evidence_id == "E001"
    assert evidence[1].evidence_id == "E002"
    assert evidence[0].category == "power"
    assert evidence[1].category == "networking"


def test_evidence_items_with_invalid_category_are_discarded_not_fatal():
    """Regression: one item with an invalid category must not discard all items."""
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_client(
            [
                {
                    "evidence_items": [
                        {
                            "claim": "Valid item.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "Rubin NVL72 uses 120kW liquid cooling.",
                            "category": "power",
                            "relevance": "Relevant.",
                            "confidence": "high",
                        },
                        {
                            "claim": "Invalid category item.",
                            "source_document": "rubin.md",
                            "evidence_snippet": "Some text.",
                            "category": "invalid_category_value",  # not in EvidenceCategory
                            "relevance": "Relevant.",
                            "confidence": "high",
                        },
                    ]
                }
            ]
        ),
    )

    evidence = client.extract_evidence("Explain Rubin power", [_document()])

    # The valid item must survive; the invalid one is silently discarded.
    assert len(evidence) == 1
    assert evidence[0].claim == "Valid item."
    assert evidence[0].evidence_id == "E001"


def test_evidence_reaches_ranking_and_synthesis_via_claude_path():
    """Regression: evidence extracted by Claude is ranked and passed to synthesis."""
    client = ClaudeClient(
        api_key="test-key",
        model="test-sonnet",
        anthropic_client=_fake_client(
            [
                {
                    "research_questions": ["What is Rubin?"],
                    "key_topics": ["power"],
                    "source_priorities": ["rubin.md"],
                },
                {
                    "evidence_items": [
                        {
                            "claim": f"Claim {i}.",
                            "source_document": "rubin.md",
                            "evidence_snippet": f"NVIDIA Rubin NVL72 rack power snippet {i}.",
                            "category": "power",
                            "relevance": "Directly relevant.",
                            "confidence": "high",
                        }
                        for i in range(1, 8)
                    ]
                },
                {
                    "executive_summary": "Summary.",
                    "confirmed_facts": ["Fact. [Source: rubin.md, Evidence: E001]"],
                    "inferences": ["Inference."],
                    "power_implications": ["Power. [Source: rubin.md, Evidence: E001]"],
                    "cooling_implications": [],
                    "networking_implications": [],
                    "rack_architecture_implications": [],
                    "open_questions": ["Question."],
                },
            ]
        ),
    )

    memo = DcPowerAgent(client=client, top_evidence=5).analyze(
        "Explain Rubin rack power", [_document()]
    )

    # All 7 items must appear in source_notes (full evidence pool).
    assert len(memo.source_notes) == 7
    # Only 5 were sent to synthesis.
    assert memo.metadata["evidence_items_used_for_synthesis"] == 5
    assert memo.metadata["evidence_items_total"] == 7
    # IDs are sequential.
    ids = [item.evidence_id for item in memo.source_notes]
    assert ids == [f"E{i:03d}" for i in range(1, 8)]


def test_top_evidence_5_passes_exactly_5_items_when_more_exist():
    """Regression: top_evidence=5 must pass exactly 5 items to synthesis, not more or fewer."""
    from research_agent.agent import select_top_evidence
    from research_agent.schemas import EvidenceItem

    # Scores range 1.0–5.0 in 0.4 increments across 10 items.
    scores = [round(1.0 + i * 0.4, 1) for i in range(10)]  # 1.0, 1.4, … 4.6
    items = [
        EvidenceItem(
            evidence_id=f"E{i:03d}",
            claim=f"Claim {i}.",
            source_document="rubin.md",
            evidence_snippet="Rubin rack power.",
            category="power",
            relevance="Relevant.",
            confidence="high",
            overall_score=scores[i - 1],
        )
        for i in range(1, 11)  # 10 items
    ]

    selected = select_top_evidence(items, top_n=5)

    assert len(selected) == 5
    # Highest-scored items (top 5 of 10) are selected.
    min_selected_score = min(item.overall_score for item in selected)
    max_unselected_score = max(
        item.overall_score for item in items if item not in selected
    )
    assert min_selected_score > max_unselected_score


def _document() -> SourceDocument:
    return SourceDocument(
        path=Path("rubin.md"),
        title="rubin",
        extension=".md",
        text="Rubin uses rack-scale architecture with power, cooling, and networking.",
    )


def _fake_client(payloads):
    return SimpleNamespace(messages=_FakeMessages(payloads))


def _fake_text_client(payloads):
    return SimpleNamespace(messages=_FakeMessages(payloads, text_only=True))


def _failing_client(exc):
    return SimpleNamespace(messages=_FailingMessages(exc))


class _FakeMessages:
    def __init__(self, payloads, text_only=False):
        self.payloads = list(payloads)
        self.text_only = text_only

    def create(self, **_kwargs):
        payload = self.payloads.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        if not self.text_only and _kwargs.get("tools"):
            tool = _kwargs["tools"][0]
            tool_input = json.loads(text)
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name=tool["name"],
                        input=tool_input,
                    )
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )


class _FailingMessages:
    def __init__(self, exc):
        self.exc = exc

    def create(self, **_kwargs):
        raise self.exc


def _truncated_client(payloads):
    """Fake client that returns stop_reason=max_tokens with an empty tool input."""
    return SimpleNamespace(messages=_TruncatedMessages(payloads))


def _fake_client_with_truncated_evidence(*, research_plan: dict, memo: dict):
    """Fake client: research_plan succeeds, extract_evidence is truncated, memo succeeds."""
    return SimpleNamespace(
        messages=_MixedMessages(research_plan=research_plan, memo=memo)
    )


class _TruncatedMessages:
    """Returns stop_reason=max_tokens with an empty tool input for the first call."""

    def __init__(self, payloads):
        self.payloads = list(payloads)

    def create(self, **_kwargs):
        # Consume a payload to advance state, but return a truncated response.
        if self.payloads:
            self.payloads.pop(0)
        return SimpleNamespace(
            stop_reason="max_tokens",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="extract_evidence",
                    input={},  # truncated — empty tool input
                )
            ],
            usage=SimpleNamespace(input_tokens=500, output_tokens=6000),
        )


class _MixedMessages:
    """research_plan → normal; extract_evidence → truncated; synthesize_memo → normal."""

    def __init__(self, *, research_plan: dict, memo: dict):
        self._sequence = [
            ("normal", research_plan),
            ("truncated", {}),
            ("normal", memo),
        ]

    def create(self, **_kwargs):
        mode, payload = self._sequence.pop(0)
        if mode == "truncated":
            return SimpleNamespace(
                stop_reason="max_tokens",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="extract_evidence",
                        input={},
                    )
                ],
                usage=SimpleNamespace(input_tokens=500, output_tokens=6000),
            )
        text = json.dumps(payload)
        if _kwargs.get("tools"):
            tool = _kwargs["tools"][0]
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name=tool["name"],
                        input=json.loads(text),
                    )
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
