"""K1.0 – Web search and page retrieval.

Public API
----------
SearchResult             – one item from a web search (title, url, snippet)
WebDocument              – fetched and extracted web page (url, title, text)
DuckDuckGoSearchProvider – search via duckduckgo_search
download_web_document    – fetch a URL and extract main text via trafilatura
web_retrieve             – search + fetch pipeline; returns (docs, trace_dict)

External dependencies (all guarded – unavailability is surfaced in the trace):
  duckduckgo-search  – pip install duckduckgo-search
  requests           – pip install requests
  trafilatura        – pip install trafilatura
"""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional external dependency guards
# ---------------------------------------------------------------------------
# Imported at module load time so tests can patch the module-level names
# (_DDGS, _requests, _trafilatura) without touching sys.modules.

_dep_status: dict[str, str] = {}  # name -> "ok" | "missing" | error message

# Prefer the new 'ddgs' package (successor to 'duckduckgo-search').
# Fall back to the legacy 'duckduckgo_search' if only that is installed.
# Both expose an identical DDGS().__enter__ / .text() context-manager API.
try:
    from ddgs import DDGS as _DDGS  # type: ignore[import-untyped]
    _dep_status["duckduckgo_search"] = "ok (ddgs)"
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS  # type: ignore[import-untyped]
        _dep_status["duckduckgo_search"] = "ok (duckduckgo_search legacy)"
        LOGGER.warning(
            "Using deprecated duckduckgo_search package — upgrade with: "
            "pip install ddgs"
        )
    except ImportError as _e:
        _DDGS = None  # type: ignore[assignment]
        _dep_status["duckduckgo_search"] = f"missing: {_e}"
        LOGGER.warning("ddgs not installed — web search disabled. pip install ddgs")

try:
    import requests as _requests  # type: ignore[import-untyped]
    _dep_status["requests"] = "ok"
except ImportError as _e:
    _requests = None  # type: ignore[assignment]
    _dep_status["requests"] = f"missing: {_e}"
    LOGGER.warning("requests not installed — page download disabled. pip install requests")

try:
    import trafilatura as _trafilatura  # type: ignore[import-untyped]
    _dep_status["trafilatura"] = "ok"
except ImportError as _e:
    _trafilatura = None  # type: ignore[assignment]
    _dep_status["trafilatura"] = f"missing: {_e}"
    LOGGER.warning("trafilatura not installed — text extraction disabled. pip install trafilatura")


def _deps_available() -> bool:
    return _DDGS is not None and _requests is not None and _trafilatura is not None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One result from a web search."""

    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class WebDocument:
    """Fetched and text-extracted web page."""

    url: str
    title: str
    text: str
    fetched_at: str = ""


# ---------------------------------------------------------------------------
# Search provider
# ---------------------------------------------------------------------------


class DuckDuckGoSearchProvider:
    """Web search via the duckduckgo_search library (DDGS context manager API)."""

    name: str = "DuckDuckGoSearchProvider"

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> tuple[list[SearchResult], str | None]:
        """Return ``(results, error)`` for *query*.

        *error* is ``None`` on success or a string describing the failure.
        Never raises.
        """
        LOGGER.info(
            "[WEB SEARCH] provider=%s  query=%r  max_results=%d",
            self.name,
            query,
            max_results,
        )

        if _DDGS is None:
            msg = (
                "duckduckgo_search library not installed. "
                "Install with: pip install duckduckgo-search"
            )
            LOGGER.error("[WEB SEARCH] DISABLED — %s", msg)
            return [], msg

        try:
            with _DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            LOGGER.error(
                "[WEB SEARCH] search exception for %r — %s\n%s",
                query,
                msg,
                traceback.format_exc(),
            )
            return [], msg

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
            for r in raw
            if r.get("href")
        ]
        LOGGER.info(
            "[WEB SEARCH] provider=%s  raw_count=%d  valid_count=%d",
            self.name,
            len(raw),
            len(results),
        )
        return results, None


# ---------------------------------------------------------------------------
# Page downloader
# ---------------------------------------------------------------------------


def download_web_document(
    url: str,
    *,
    timeout: int = 20,
) -> tuple[WebDocument | None, str | None]:
    """Fetch *url* and extract its main text.

    Returns ``(document, error)``.  *error* is ``None`` on success or a string
    describing what failed.  Never raises.
    """
    LOGGER.info("[WEB DOWNLOAD] fetching url=%s  timeout=%ds", url, timeout)

    if _requests is None:
        msg = "requests library not installed. pip install requests"
        LOGGER.error("[WEB DOWNLOAD] DISABLED — %s", msg)
        return None, msg

    if _trafilatura is None:
        msg = "trafilatura library not installed. pip install trafilatura"
        LOGGER.error("[WEB DOWNLOAD] DISABLED — %s", msg)
        return None, msg

    # --- HTTP fetch ---
    try:
        resp = _requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-harness/1.0)"},
        )
        resp.raise_for_status()
        LOGGER.info(
            "[WEB DOWNLOAD] fetch ok  url=%s  status=%s  html_chars=%d",
            url,
            resp.status_code,
            len(resp.text),
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        LOGGER.error(
            "[WEB DOWNLOAD] fetch failed  url=%s — %s\n%s",
            url,
            msg,
            traceback.format_exc(),
        )
        return None, f"fetch error: {msg}"

    # --- text extraction ---
    try:
        text = _trafilatura.extract(resp.text) or ""
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        LOGGER.error(
            "[WEB DOWNLOAD] extraction exception  url=%s — %s\n%s",
            url,
            msg,
            traceback.format_exc(),
        )
        return None, f"extraction error: {msg}"

    if not text.strip():
        msg = "trafilatura returned empty text (page may be JS-rendered or paywalled)"
        LOGGER.warning("[WEB DOWNLOAD] no text extracted  url=%s — %s", url, msg)
        return None, msg

    title = _extract_title(resp.text) or url
    LOGGER.info(
        "[WEB DOWNLOAD] extracted  url=%s  title=%r  chars=%d",
        url,
        title,
        len(text),
    )
    return (
        WebDocument(
            url=url,
            title=title,
            text=text,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        ),
        None,
    )


def _extract_title(html: str) -> str:
    """Extract the first <title> value from *html*."""
    m = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------


def web_retrieve(
    query: str,
    *,
    max_results: int = 5,
    max_pages: int = 5,
    timeout_seconds: int = 20,
    cache=None,  # WebPageCache | None  (avoid circular import from type annotation)
) -> tuple[list[WebDocument], dict]:
    """Search DuckDuckGo and fetch the top pages for *query*.

    Returns
    -------
    (documents, trace_dict)
        ``documents`` is the list of successfully fetched :class:`WebDocument`
        objects (may be empty on total failure).
        ``trace_dict`` is a fully structured observability record covering
        every stage: dependency check → search → per-URL download → summary.
    """
    LOGGER.info(
        "[WEB RETRIEVE] starting  query=%r  max_results=%d  max_pages=%d  timeout=%ds",
        query,
        max_results,
        max_pages,
        timeout_seconds,
    )

    provider = DuckDuckGoSearchProvider()

    # Stage 1: dependency check
    dep_check = {
        "duckduckgo_search": _dep_status.get("duckduckgo_search", "ok"),
        "requests": _dep_status.get("requests", "ok"),
        "trafilatura": _dep_status.get("trafilatura", "ok"),
        "all_available": _deps_available(),
    }
    if not dep_check["all_available"]:
        missing = [k for k, v in dep_check.items() if k != "all_available" and v != "ok"]
        LOGGER.error(
            "[WEB RETRIEVE] aborting — missing dependencies: %s",
            missing,
        )

    LOGGER.info(
        "[WEB RETRIEVE] provider=%s  deps_ok=%s",
        provider.name,
        dep_check["all_available"],
    )

    # Stage 2: search
    search_results, search_error = provider.search(query, max_results=max_results)

    # Stage 3: per-URL fetch
    documents: list[WebDocument] = []
    download_records: list[dict] = []
    cache_hits: int = 0

    for result in search_results[:max_pages]:
        url = result.url
        if not url:
            continue

        rec: dict = {"url": url, "title": result.title, "source": "unknown", "error": None, "chars": 0}

        # --- cache check ---
        if cache is not None:
            cached = cache.get(url)
            if cached is not None:
                doc = WebDocument(
                    url=cached.get("url", url),
                    title=cached.get("title", url),
                    text=cached.get("text", ""),
                )
                documents.append(doc)
                cache_hits += 1
                rec["source"] = "cache"
                rec["chars"] = len(doc.text)
                LOGGER.info("[WEB RETRIEVE] cache hit  url=%s  chars=%d", url, rec["chars"])
                download_records.append(rec)
                continue

        # --- live download ---
        doc, dl_error = download_web_document(url, timeout=timeout_seconds)
        rec["source"] = "live"
        if doc is not None:
            documents.append(doc)
            rec["chars"] = len(doc.text)
            rec["title"] = doc.title
            if cache is not None:
                cache.set(url, {"url": doc.url, "title": doc.title, "text": doc.text})
        else:
            rec["error"] = dl_error

        download_records.append(rec)

    downloaded_urls = [r["url"] for r in download_records if r["error"] is None]
    failed_urls = [{"url": r["url"], "error": r["error"]} for r in download_records if r["error"]]

    trace: dict = {
        # ---- dependency availability ----
        "dependency_status": dep_check,
        # ---- search stage ----
        "query": query,
        "provider": provider.name,
        "search_error": search_error,
        "search_results_returned": len(search_results),
        "search_results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet[:200]}
            for r in search_results
        ],
        # ---- download stage ----
        "pages_attempted": len(download_records),
        "downloaded_urls": downloaded_urls,
        "failed_urls": failed_urls,
        "cache_hits": cache_hits,
        # ---- extraction summary ----
        "documents_fetched": len(documents),
        "extracted_characters": sum(len(d.text) for d in documents),
        "download_details": download_records,
    }

    LOGGER.info(
        "[WEB RETRIEVE] done  query=%r  results=%d  attempted=%d  fetched=%d  "
        "chars=%d  cache_hits=%d  failures=%d",
        query,
        len(search_results),
        len(download_records),
        len(documents),
        trace["extracted_characters"],
        cache_hits,
        len(failed_urls),
    )

    return documents, trace
