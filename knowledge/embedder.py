"""Evidence embedding backends.

The EmbeddingBackend protocol is the stable interface.
Implementations are swappable — the KnowledgeBuilder selects one at startup.

Available backends:
  LocalEmbedder   — uses sentence-transformers (requires: pip install sentence-transformers)
  NullEmbedder    — skips embedding; logs a warning; stores nothing

J8.1: LocalEmbedder if sentence-transformers is installed, NullEmbedder otherwise.
J8.3: Qdrant-backed index will be added; embeddings generated here are loaded into it.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)

EMBEDDING_DIM_LOCAL = 384  # all-MiniLM-L6-v2


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for any embedding backend."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float] | None:
        """Return a float vector or None if embedding is unavailable."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed multiple texts. Returns a list of same length as input."""
        ...


# ---------------------------------------------------------------------------
# LocalEmbedder — sentence-transformers
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """Embedding backend using sentence-transformers (all-MiniLM-L6-v2).

    Produces 384-dimensional float32 vectors.
    Model is downloaded once to ~/.cache/huggingface/ on first use.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._model = SentenceTransformer(self.MODEL_NAME)
            LOGGER.info("embedder: loaded %s", self.MODEL_NAME)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM_LOCAL

    def embed(self, text: str) -> list[float] | None:
        try:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception as exc:
            LOGGER.warning("embedder: failed to embed text — %s", exc)
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in vecs]
        except Exception as exc:
            LOGGER.warning("embedder: batch embedding failed — %s", exc)
            return [None] * len(texts)


# ---------------------------------------------------------------------------
# NullEmbedder — no-op, used when no backend is available
# ---------------------------------------------------------------------------


class NullEmbedder:
    """No-op embedder. Embeddings are not generated.

    Use this during development or when no embedding provider is configured.
    Evidence records will not be embedded; the manifest tracks this.
    """

    @property
    def dimension(self) -> int:
        return 0

    def embed(self, text: str) -> list[float] | None:
        return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_default_embedder() -> EmbeddingBackend:
    """Return the best available embedding backend.

    Prefers LocalEmbedder if sentence-transformers is installed.
    Falls back to NullEmbedder with a warning.
    """
    try:
        import sentence_transformers  # noqa: F401  # type: ignore[import]
        backend = LocalEmbedder()
        LOGGER.info("embedder: using LocalEmbedder (sentence-transformers)")
        return backend
    except (ImportError, RuntimeError):
        LOGGER.warning(
            "embedder: sentence-transformers not available — embeddings will be skipped. "
            "Install with: pip install sentence-transformers"
        )
        return NullEmbedder()
