"""Embedding providers for semantic retrieval (J8.4).

Provider hierarchy
------------------
EmbeddingProvider (ABC)
  └── LocalEmbeddingProvider   — sentence-transformers, no API key required
                                  default model: all-MiniLM-L6-v2 (384-dim, 80 MB)

The only public-facing API callers need:
  provider = get_provider()
  vecs = provider.embed(["text one", "text two"])
  or
  vec = provider.embed_one("single text")

Embeddings are persisted to KnowledgeStore via write_embedding(evidence_id, vector).
On subsequent runs, already-embedded items are skipped (incremental).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

LOGGER = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Maps text strings to float vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier used in RetrievalResult.semantic_model."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimensionality."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one float list per text, same order."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers embedding provider (local, no API key required).

    Default model all-MiniLM-L6-v2:
      - 384-dimensional embeddings
      - ~80 MB download on first use (cached at ~/.cache/huggingface/)
      - Fast CPU inference (~5 ms/item)
      - Good semantic quality for domain-specific retrieval
    """

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic retrieval. "
                "Install: pip3 install 'sentence-transformers>=3.0' numpy"
            ) from exc
        self._model_name = model
        LOGGER.info("embedding: loading model %r", model)
        self._st = SentenceTransformer(model)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._st.get_sentence_embedding_dimension()  # type: ignore[attr-defined]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._st.encode(  # type: ignore[attr-defined]
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]


_PROVIDER_CACHE: dict[str, EmbeddingProvider] = {}


def get_provider(model: str | None = None) -> EmbeddingProvider:
    """Return the default embedding provider (LocalEmbeddingProvider).

    The model is loaded once per process per model name and cached.
    Subsequent calls with the same model name return the cached instance.
    The model weights are also cached on disk at ~/.cache/huggingface/hub/.
    """
    key = model or DEFAULT_LOCAL_MODEL
    if key not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[key] = LocalEmbeddingProvider(key)
    return _PROVIDER_CACHE[key]


def embed_evidence_batch(
    items: list,  # list[Evidence] — no circular import
    store: object,  # KnowledgeStore
    provider: EmbeddingProvider,
    *,
    force: bool = False,
    batch_size: int = 64,
) -> tuple[int, int]:
    """Embed evidence statements and persist vectors to the KnowledgeStore.

    Parameters
    ----------
    items:
        Evidence items to embed.
    store:
        KnowledgeStore instance — must support has_embedding / write_embedding.
    provider:
        Embedding provider to use.
    force:
        If True, regenerate even if an embedding already exists.
    batch_size:
        Items per model call.

    Returns
    -------
    (embedded_count, skipped_count)
    """
    to_embed = [ev for ev in items if force or not store.has_embedding(ev.evidence_id)]  # type: ignore[attr-defined]
    skipped = len(items) - len(to_embed)

    embedded = 0
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        statements = [ev.statement for ev in batch]
        vectors = provider.embed(statements)
        for ev, vec in zip(batch, vectors):
            store.write_embedding(ev.evidence_id, vec)  # type: ignore[attr-defined]
        embedded += len(batch)
        LOGGER.info(
            "embedding: %d/%d embedded (batch %d–%d)",
            embedded, len(to_embed), i + 1, i + len(batch),
        )

    return embedded, skipped
