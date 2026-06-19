"""Tests for K1.0 – Minimal Internet Retrieval.

All external network calls are mocked so these tests run offline.
Module-level sentinel names (_DDGS, _requests, _trafilatura) in
research_agent.web_search are patched via unittest.mock.patch.

API note (post-instrumentation):
  DuckDuckGoSearchProvider.search() -> (list[SearchResult], error | None)
  download_web_document()           -> (WebDocument | None, error | None)
  web_retrieve()                    -> (list[WebDocument], trace_dict)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from research_agent.web_cache import WebPageCache
from research_agent.web_search import (
    DuckDuckGoSearchProvider,
    SearchResult,
    WebDocument,
    _extract_title,
    download_web_document,
    web_retrieve,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_HTML = """\
<html>
<head><title>SMR Reactor Report</title></head>
<body><p>Small modular reactors offer a promising construction timeline of 24 months.</p></body>
</html>
"""

_FAKE_EXTRACTED_TEXT = (
    "Small modular reactors offer a promising construction timeline of 24 months."
)

_FAKE_SEARCH_RESULTS = [
    {"title": "SMR Overview", "href": "https://example.com/smr", "body": "A brief snippet."},
    {"title": "Reactor Design", "href": "https://example.com/design", "body": "Another snippet."},
]


# ---------------------------------------------------------------------------
# SearchResult / WebDocument dataclasses
# ---------------------------------------------------------------------------


class TestDataClasses:

    def test_search_result_fields(self):
        r = SearchResult(title="My Title", url="https://example.com", snippet="A snippet")
        assert r.title == "My Title"
        assert r.url == "https://example.com"
        assert r.snippet == "A snippet"

    def test_search_result_default_snippet(self):
        r = SearchResult(title="T", url="https://example.com")
        assert r.snippet == ""

    def test_web_document_fields(self):
        d = WebDocument(url="https://example.com", title="T", text="body text")
        assert d.text == "body text"
        assert d.fetched_at == ""  # default


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------


class TestExtractTitle:

    def test_extracts_simple_title(self):
        html = "<html><head><title>Hello World</title></head><body/></html>"
        assert _extract_title(html) == "Hello World"

    def test_returns_empty_on_no_title(self):
        assert _extract_title("<html><body>no title</body></html>") == ""

    def test_strips_whitespace(self):
        html = "<title>  Spaces  </title>"
        assert _extract_title(html) == "Spaces"


# ---------------------------------------------------------------------------
# DuckDuckGoSearchProvider — returns (results, error)
# ---------------------------------------------------------------------------


class TestDuckDuckGoSearchProvider:

    def _make_ddgs_mock(self, return_value):
        mock_ddgs_class = MagicMock()
        mock_instance = MagicMock()
        mock_ddgs_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_ddgs_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = return_value
        return mock_ddgs_class, mock_instance

    def test_returns_results_when_library_available(self):
        mock_ddgs_class, _ = self._make_ddgs_mock(_FAKE_SEARCH_RESULTS)

        with patch("research_agent.web_search._DDGS", mock_ddgs_class):
            results, error = DuckDuckGoSearchProvider().search(
                "SMR construction timeline", max_results=5
            )

        assert error is None
        assert len(results) == 2
        assert results[0].title == "SMR Overview"
        assert results[0].url == "https://example.com/smr"

    def test_returns_error_when_library_missing(self):
        with patch("research_agent.web_search._DDGS", None):
            results, error = DuckDuckGoSearchProvider().search("anything")
        assert results == []
        assert error is not None
        assert "not installed" in error.lower()

    def test_returns_error_on_search_exception(self):
        mock_ddgs_class, mock_instance = self._make_ddgs_mock(None)
        mock_instance.text.side_effect = RuntimeError("rate limited")

        with patch("research_agent.web_search._DDGS", mock_ddgs_class):
            results, error = DuckDuckGoSearchProvider().search("SMR", max_results=3)

        assert results == []
        assert error is not None
        assert "RuntimeError" in error or "rate limited" in error

    def test_skips_results_without_href(self):
        mock_ddgs_class, _ = self._make_ddgs_mock([
            {"title": "Good", "href": "https://example.com/ok", "body": "s"},
            {"title": "Bad", "href": "", "body": "s"},
            {"title": "NoHref", "body": "s"},
        ])

        with patch("research_agent.web_search._DDGS", mock_ddgs_class):
            results, error = DuckDuckGoSearchProvider().search("q")

        assert len(results) == 1
        assert results[0].url == "https://example.com/ok"
        assert error is None


# ---------------------------------------------------------------------------
# download_web_document — returns (doc | None, error | None)
# ---------------------------------------------------------------------------


class TestDownloadWebDocument:

    def _make_mock_requests(self, html: str = _FAKE_HTML):
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp
        return mock_requests

    def test_returns_document_on_success(self):
        mock_requests = self._make_mock_requests()
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.return_value = _FAKE_EXTRACTED_TEXT

        with (
            patch("research_agent.web_search._requests", mock_requests),
            patch("research_agent.web_search._trafilatura", mock_trafilatura),
        ):
            doc, error = download_web_document("https://example.com/smr", timeout=5)

        assert error is None
        assert doc is not None
        assert doc.url == "https://example.com/smr"
        assert doc.title == "SMR Reactor Report"
        assert "24 months" in doc.text
        assert doc.fetched_at != ""

    def test_returns_error_when_requests_missing(self):
        with patch("research_agent.web_search._requests", None):
            doc, error = download_web_document("https://example.com")
        assert doc is None
        assert error is not None
        assert "not installed" in error.lower()

    def test_returns_error_when_trafilatura_missing(self):
        mock_requests = self._make_mock_requests()
        with (
            patch("research_agent.web_search._requests", mock_requests),
            patch("research_agent.web_search._trafilatura", None),
        ):
            doc, error = download_web_document("https://example.com")
        assert doc is None
        assert error is not None
        assert "not installed" in error.lower()

    def test_returns_error_on_http_error(self):
        mock_requests = MagicMock()
        mock_requests.get.side_effect = ConnectionError("connection refused")
        mock_trafilatura = MagicMock()

        with (
            patch("research_agent.web_search._requests", mock_requests),
            patch("research_agent.web_search._trafilatura", mock_trafilatura),
        ):
            doc, error = download_web_document("https://example.com")

        assert doc is None
        assert error is not None
        assert "fetch error" in error.lower()

    def test_returns_error_on_empty_extraction(self):
        mock_requests = self._make_mock_requests()
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.return_value = ""

        with (
            patch("research_agent.web_search._requests", mock_requests),
            patch("research_agent.web_search._trafilatura", mock_trafilatura),
        ):
            doc, error = download_web_document("https://example.com")

        assert doc is None
        assert error is not None

    def test_returns_error_on_extraction_exception(self):
        mock_requests = self._make_mock_requests()
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract.side_effect = Exception("parse error")

        with (
            patch("research_agent.web_search._requests", mock_requests),
            patch("research_agent.web_search._trafilatura", mock_trafilatura),
        ):
            doc, error = download_web_document("https://example.com")

        assert doc is None
        assert error is not None
        assert "extraction error" in error.lower()


# ---------------------------------------------------------------------------
# web_retrieve (pipeline) — (docs, trace_dict)
# ---------------------------------------------------------------------------

# Internal mock return helpers: provider.search → (results, error);
# download_web_document → (doc, error)

def _mock_search(results, error=None):
    """Return a side_effect callable for provider.search that returns (results, error)."""
    def _side_effect(query, *, max_results=5):
        return results, error
    return _side_effect


class TestWebRetrieve:

    def test_returns_documents_and_structured_trace(self):
        sr = [SearchResult(title="T", url="https://example.com/page", snippet="s")]
        doc = WebDocument(url="https://example.com/page", title="T", text="body text here")

        mock_provider_class = MagicMock(return_value=MagicMock())
        mock_provider_class.return_value.search.return_value = (sr, None)

        with (
            patch("research_agent.web_search.DuckDuckGoSearchProvider", mock_provider_class),
            patch("research_agent.web_search.download_web_document", return_value=(doc, None)),
        ):
            docs, trace = web_retrieve("SMR construction", max_results=3, max_pages=3)

        assert len(docs) == 1
        assert docs[0].text == "body text here"

        # --- trace structure checks ---
        assert trace["query"] == "SMR construction"
        assert "provider" in trace  # value depends on mock
        assert trace["search_error"] is None
        assert trace["search_results_returned"] == 1
        assert trace["documents_fetched"] == 1
        assert trace["extracted_characters"] == len("body text here")
        assert "https://example.com/page" in trace["downloaded_urls"]
        assert trace["failed_urls"] == []
        assert trace["dependency_status"]["all_available"] in (True, False)

        # --- download_details per-URL record ---
        assert len(trace["download_details"]) == 1
        rec = trace["download_details"][0]
        assert rec["url"] == "https://example.com/page"
        assert rec["error"] is None
        assert rec["chars"] == len("body text here")

    def test_trace_records_search_error(self):
        mock_provider_class = MagicMock(return_value=MagicMock())
        mock_provider_class.return_value.search.return_value = ([], "rate limited")

        with patch("research_agent.web_search.DuckDuckGoSearchProvider", mock_provider_class):
            docs, trace = web_retrieve("anything")

        assert docs == []
        assert trace["search_error"] == "rate limited"
        assert trace["documents_fetched"] == 0
        assert trace["search_results_returned"] == 0

    def test_trace_records_download_errors(self):
        sr = [
            SearchResult(title="OK", url="https://example.com/ok", snippet=""),
            SearchResult(title="Bad", url="https://example.com/bad", snippet=""),
        ]
        doc_ok = WebDocument(url="https://example.com/ok", title="OK", text="good content")

        def fake_download(url, *, timeout=20):
            if url == "https://example.com/ok":
                return doc_ok, None
            return None, "fetch error: ConnectionError: timeout"

        mock_provider_class = MagicMock(return_value=MagicMock())
        mock_provider_class.return_value.search.return_value = (sr, None)

        with (
            patch("research_agent.web_search.DuckDuckGoSearchProvider", mock_provider_class),
            patch("research_agent.web_search.download_web_document", side_effect=fake_download),
        ):
            docs, trace = web_retrieve("q", max_pages=5)

        assert len(docs) == 1
        assert trace["documents_fetched"] == 1
        assert len(trace["failed_urls"]) == 1
        assert trace["failed_urls"][0]["url"] == "https://example.com/bad"
        assert "error" in trace["failed_urls"][0]

    def test_dep_status_in_trace_when_missing(self):
        with (
            patch("research_agent.web_search._DDGS", None),
            patch("research_agent.web_search._dep_status", {
                "duckduckgo_search": "missing: No module named 'duckduckgo_search'",
                "requests": "ok",
                "trafilatura": "ok",
            }),
        ):
            docs, trace = web_retrieve("test query")

        assert trace["dependency_status"]["duckduckgo_search"].startswith("missing")
        assert trace["dependency_status"]["all_available"] is False

    def test_cache_hit_skips_download(self):
        sr = [SearchResult(title="T", url="https://example.com/cached", snippet="")]
        cached_data = {
            "url": "https://example.com/cached",
            "title": "Cached Title",
            "text": "cached body",
        }

        class FakeCache:
            def get(self, url):
                return cached_data if url == "https://example.com/cached" else None
            def set(self, url, data):
                pass

        mock_provider_class = MagicMock(return_value=MagicMock())
        mock_provider_class.return_value.search.return_value = (sr, None)
        mock_download = MagicMock(return_value=(None, "should not be called"))

        with (
            patch("research_agent.web_search.DuckDuckGoSearchProvider", mock_provider_class),
            patch("research_agent.web_search.download_web_document", mock_download),
        ):
            docs, trace = web_retrieve("q", cache=FakeCache())

        mock_download.assert_not_called()
        assert len(docs) == 1
        assert docs[0].text == "cached body"
        assert trace["cache_hits"] == 1
        rec = trace["download_details"][0]
        assert rec["source"] == "cache"

    def test_cache_set_called_on_new_download(self):
        sr = [SearchResult(title="T", url="https://example.com/new", snippet="")]
        doc = WebDocument(url="https://example.com/new", title="T", text="fresh")
        stored: dict = {}

        class FakeCache:
            def get(self, url):
                return None
            def set(self, url, data):
                stored[url] = data

        mock_provider_class = MagicMock(return_value=MagicMock())
        mock_provider_class.return_value.search.return_value = (sr, None)

        with (
            patch("research_agent.web_search.DuckDuckGoSearchProvider", mock_provider_class),
            patch("research_agent.web_search.download_web_document", return_value=(doc, None)),
        ):
            web_retrieve("q", cache=FakeCache())

        assert "https://example.com/new" in stored
        assert stored["https://example.com/new"]["text"] == "fresh"

    def test_trace_end_to_end_pipeline_fields(self):
        """Verify every stage field exists in the trace: query → results → downloads → summary."""
        sr = [
            SearchResult(title="A", url="https://a.com/pg", snippet="snip a"),
            SearchResult(title="B", url="https://b.com/pg", snippet="snip b"),
        ]
        doc_a = WebDocument(url="https://a.com/pg", title="A", text="content from A " * 10)

        def fake_dl(url, *, timeout=20):
            if url == "https://a.com/pg":
                return doc_a, None
            return None, "fetch error: timeout"

        mock_provider_class = MagicMock(return_value=MagicMock())
        mock_provider_class.return_value.search.return_value = (sr, None)

        with (
            patch("research_agent.web_search.DuckDuckGoSearchProvider", mock_provider_class),
            patch("research_agent.web_search.download_web_document", side_effect=fake_dl),
        ):
            docs, trace = web_retrieve("pipeline test", max_results=5, max_pages=5)

        # Every stage must be represented
        assert "dependency_status" in trace
        assert "query" in trace
        assert "provider" in trace
        assert "search_error" in trace
        assert "search_results_returned" in trace
        assert "search_results" in trace
        assert "pages_attempted" in trace
        assert "downloaded_urls" in trace
        assert "failed_urls" in trace
        assert "cache_hits" in trace
        assert "documents_fetched" in trace
        assert "extracted_characters" in trace
        assert "download_details" in trace

        assert trace["pages_attempted"] == 2
        assert trace["documents_fetched"] == 1
        assert len(trace["failed_urls"]) == 1
        assert trace["extracted_characters"] > 0
        assert trace["search_results"][0]["snippet"] == "snip a"


# ---------------------------------------------------------------------------
# WebPageCache
# ---------------------------------------------------------------------------


class TestWebPageCache:

    def test_get_returns_none_on_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            assert cache.get("https://example.com/missing") is None

    def test_set_then_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            data = {"url": "https://example.com", "title": "T", "text": "body"}
            cache.set("https://example.com", data)
            result = cache.get("https://example.com")
            assert result == data

    def test_different_urls_stored_separately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            cache.set("https://a.com", {"text": "aaa"})
            cache.set("https://b.com", {"text": "bbb"})
            assert cache.get("https://a.com") == {"text": "aaa"}
            assert cache.get("https://b.com") == {"text": "bbb"}

    def test_cache_key_is_url_hash_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            url = "https://example.com/test"
            expected_key = hashlib.sha256(url.encode()).hexdigest()[:16]
            cache.set(url, {"text": "x"})
            cache_file = pathlib.Path(tmpdir) / f"{expected_key}.json"
            assert cache_file.exists()

    def test_invalidate_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            url = "https://example.com/to-delete"
            cache.set(url, {"text": "to be removed"})
            assert cache.get(url) is not None
            removed = cache.invalidate(url)
            assert removed is True
            assert cache.get(url) is None

    def test_invalidate_missing_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            assert cache.invalidate("https://example.com/not-there") is False

    def test_creates_cache_dir_on_demand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = pathlib.Path(tmpdir) / "nested" / "cache"
            cache = WebPageCache(cache_dir)
            cache.set("https://example.com", {"text": "hi"})
            assert cache_dir.exists()

    def test_get_handles_corrupt_file_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WebPageCache(tmpdir)
            url = "https://example.com/corrupt"
            key = hashlib.sha256(url.encode()).hexdigest()[:16]
            (pathlib.Path(tmpdir) / f"{key}.json").write_text("not json!!!")
            assert cache.get(url) is None


# ---------------------------------------------------------------------------
# Chunk schema: source_type / source_url backward compatibility
# ---------------------------------------------------------------------------


class TestChunkSourceFields:

    def test_default_source_type_is_local(self):
        from research_agent.schemas import Chunk

        c = Chunk(
            chunk_id="c001",
            document_name="doc.pdf",
            chunk_number=0,
            text="hello",
            start_offset=0,
            end_offset=5,
        )
        assert c.source_type == "local"
        assert c.source_url == ""

    def test_web_chunk_source_fields(self):
        from research_agent.schemas import Chunk

        c = Chunk(
            chunk_id="web_0000_0000",
            document_name="https://example.com/page",
            chunk_number=0,
            text="web content",
            start_offset=0,
            end_offset=11,
            source_type="web",
            source_url="https://example.com/page",
        )
        assert c.source_type == "web"
        assert c.source_url == "https://example.com/page"


# ---------------------------------------------------------------------------
# Agent helper: _web_docs_to_chunks
# ---------------------------------------------------------------------------


class TestWebDocsToChunks:

    def test_converts_doc_to_chunk(self):
        from research_agent.agent import _web_docs_to_chunks
        from research_agent.web_search import WebDocument

        docs = [WebDocument(url="https://example.com/p", title="T", text="content here")]
        chunks = _web_docs_to_chunks(docs)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.source_type == "web"
        assert c.source_url == "https://example.com/p"
        assert c.document_name == "https://example.com/p"
        assert c.text == "content here"

    def test_skips_empty_text_docs(self):
        from research_agent.agent import _web_docs_to_chunks
        from research_agent.web_search import WebDocument

        docs = [
            WebDocument(url="https://a.com", title="A", text="content"),
            WebDocument(url="https://b.com", title="B", text="   "),
        ]
        chunks = _web_docs_to_chunks(docs)
        assert len(chunks) == 1
        assert chunks[0].source_url == "https://a.com"


# ---------------------------------------------------------------------------
# Profile: WebSearchConfig loading
# ---------------------------------------------------------------------------


class TestWebSearchConfigProfile:

    def test_default_web_search_disabled(self):
        from research_agent.profile import load_profile

        profile = load_profile("smr")
        assert profile.web_search.enabled is False

    def test_web_search_config_from_yaml(self, tmp_path):
        yaml_content = """\
name: test_ws
description: Test
web_search:
  enabled: true
  max_results: 8
  max_pages: 3
  timeout_seconds: 30
"""
        profile_file = tmp_path / "test_ws.yaml"
        profile_file.write_text(yaml_content)

        from research_agent.profile import load_profile

        profile = load_profile(str(profile_file))
        assert profile.web_search.enabled is True
        assert profile.web_search.max_results == 8
        assert profile.web_search.max_pages == 3
        assert profile.web_search.timeout_seconds == 30
