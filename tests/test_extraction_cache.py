"""Tests for ExtractionCache and Haiku extraction wiring (J5.8)."""

from __future__ import annotations

import json
import pathlib

import pytest

from research_agent.extraction_cache import ExtractionCache
from research_agent.schemas import Chunk, EvidenceItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_name="test.pdf",
        chunk_number=0,
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _make_item(evidence_id: str = "E001") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        claim="test claim",
        source_document="test.pdf",
        evidence_snippet="snippet",
        category="power",
        relevance="direct",
        confidence="high",
        entity="TestCo",
        scope="global",
    )


# ---------------------------------------------------------------------------
# Cache: basic get/put round-trip
# ---------------------------------------------------------------------------

def test_cache_miss_returns_none(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    assert cache.get("What is power?", chunks) is None


def test_cache_put_then_get(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    items = [_make_item("E001"), _make_item("E002")]

    cache.put("What is power?", chunks, items)
    result = cache.get("What is power?", chunks)

    assert result is not None
    assert len(result) == 2
    assert result[0].evidence_id == "E001"
    assert result[1].evidence_id == "E002"


def test_cache_key_is_question_sensitive(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    chunks = [_make_chunk("c1")]
    items = [_make_item()]

    cache.put("Question A", chunks, items)
    assert cache.get("Question B", chunks) is None


def test_cache_key_is_chunk_sensitive(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    items = [_make_item()]

    cache.put("Question", [_make_chunk("c1")], items)
    assert cache.get("Question", [_make_chunk("c2")]) is None


def test_cache_key_order_independent(tmp_path):
    """Sorted chunk IDs — order of chunk list doesn't matter."""
    cache = ExtractionCache(cache_dir=tmp_path)
    items = [_make_item()]
    c1, c2 = _make_chunk("aaa"), _make_chunk("bbb")

    cache.put("Q", [c1, c2], items)
    result = cache.get("Q", [c2, c1])  # reversed order
    assert result is not None


def test_cache_creates_directory(tmp_path):
    nested = tmp_path / "deep" / "nested"
    cache = ExtractionCache(cache_dir=nested)
    cache.put("Q", [_make_chunk("c1")], [_make_item()])
    assert nested.exists()


def test_cache_file_is_valid_json(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    chunks = [_make_chunk("c1")]
    items = [_make_item()]
    cache.put("Q", chunks, items)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert isinstance(data, list)
    assert data[0]["evidence_id"] == "E001"


def test_cache_corrupt_file_returns_none(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    chunks = [_make_chunk("c1")]
    cache.put("Q", chunks, [_make_item()])

    # corrupt the file
    files = list(tmp_path.glob("*.json"))
    files[0].write_text("not json!!!")

    assert cache.get("Q", chunks) is None


def test_cache_invalidate(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    chunks = [_make_chunk("c1")]
    cache.put("Q", chunks, [_make_item()])

    removed = cache.invalidate("Q", chunks)
    assert removed is True
    assert cache.get("Q", chunks) is None


def test_cache_invalidate_nonexistent(tmp_path):
    cache = ExtractionCache(cache_dir=tmp_path)
    result = cache.invalidate("Q", [_make_chunk("c1")])
    assert result is False


# ---------------------------------------------------------------------------
# ClaudeClient: extraction_model and cache wiring
# ---------------------------------------------------------------------------

def test_claude_client_has_extraction_model():
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient, DEFAULT_EXTRACTION_MODEL

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic)
    assert client.extraction_model == DEFAULT_EXTRACTION_MODEL


def test_claude_client_extraction_model_override():
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic, extraction_model="claude-haiku-4-5-20251001")
    assert client.extraction_model == "claude-haiku-4-5-20251001"


def test_claude_client_cache_disabled_by_default():
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic)
    assert client._extraction_cache is None


def test_claude_client_cache_enabled_when_requested():
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient
    from research_agent.extraction_cache import ExtractionCache

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic, use_extraction_cache=True)
    assert isinstance(client._extraction_cache, ExtractionCache)


def test_claude_client_cache_can_be_disabled():
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic, use_extraction_cache=False)
    assert client._extraction_cache is None


def test_extract_evidence_from_chunks_uses_cache(tmp_path):
    """Cache hit: LLM call is skipped entirely."""
    from unittest.mock import MagicMock, patch
    from research_agent.claude_client import ClaudeClient
    from research_agent.extraction_cache import ExtractionCache

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic, use_extraction_cache=True)
    cache = ExtractionCache(cache_dir=tmp_path)
    client._extraction_cache = cache

    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    cached_items = [_make_item("E001")]
    cache.put("What is power?", chunks, cached_items)

    result = client.extract_evidence_from_chunks("What is power?", chunks)

    assert result == cached_items
    mock_anthropic.messages.create.assert_not_called()


def test_extract_evidence_from_chunks_writes_cache_on_miss(tmp_path):
    """Cache miss: LLM call runs and result is stored."""
    from unittest.mock import MagicMock, patch
    from research_agent.claude_client import ClaudeClient
    from research_agent.extraction_cache import ExtractionCache
    from research_agent.schemas import assign_evidence_ids

    mock_anthropic = MagicMock()
    client = ClaudeClient(anthropic_client=mock_anthropic, use_extraction_cache=True)
    cache = ExtractionCache(cache_dir=tmp_path)
    client._extraction_cache = cache

    chunks = [_make_chunk("c1")]

    # Mock the LLM to return one item
    fake_response = MagicMock()
    fake_response.stop_reason = "tool_use"
    fake_response.usage.output_tokens = 100
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "evidence_items": [{
            "claim": "GPU power is 700W",
            "source_document": "spec.pdf",
            "evidence_snippet": "700W TDP",
            "category": "power",
            "relevance": "direct",
            "confidence": "high",
            "entity": "NVIDIA",
            "scope": "product",
        }]
    }
    fake_response.content = [tool_block]
    mock_anthropic.messages.create.return_value = fake_response

    result = client.extract_evidence_from_chunks("GPU power?", chunks)

    assert len(result) == 1
    assert result[0].claim == "GPU power is 700W"

    # Verify cache was written
    cached = cache.get("GPU power?", chunks)
    assert cached is not None
    assert len(cached) == 1


def test_extract_evidence_from_chunks_uses_extraction_model(tmp_path):
    """Extraction call uses extraction_model, not synthesis model."""
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient
    from research_agent.extraction_cache import ExtractionCache

    mock_anthropic = MagicMock()
    client = ClaudeClient(
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
        extraction_model="claude-haiku-4-5-20251001",
        use_extraction_cache=False,
    )

    fake_response = MagicMock()
    fake_response.stop_reason = "tool_use"
    fake_response.usage.output_tokens = 50
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"evidence_items": []}
    fake_response.content = [tool_block]
    mock_anthropic.messages.create.return_value = fake_response

    client.extract_evidence_from_chunks("Q?", [_make_chunk("c1")])

    call_kwargs = mock_anthropic.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_synthesize_memo_still_uses_main_model():
    """synthesize_memo must NOT use the extraction model."""
    from unittest.mock import MagicMock
    from research_agent.claude_client import ClaudeClient

    mock_anthropic = MagicMock()
    client = ClaudeClient(
        anthropic_client=mock_anthropic,
        model="claude-sonnet-4-6",
        extraction_model="claude-haiku-4-5-20251001",
    )

    fake_response = MagicMock()
    fake_response.stop_reason = "tool_use"
    fake_response.usage.output_tokens = 200
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "executive_summary": "summary",
        "confirmed_facts": [],
        "inferences": [],
        "power_implications": [],
        "cooling_implications": [],
        "networking_implications": [],
        "rack_architecture_implications": [],
        "open_questions": [],
    }
    fake_response.content = [tool_block]
    mock_anthropic.messages.create.return_value = fake_response

    client.synthesize_memo("Q?", [])

    call_kwargs = mock_anthropic.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"
