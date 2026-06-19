"""Load and extract text from local source documents."""

from __future__ import annotations

from pathlib import Path

from .schemas import SourceCollection, SourceDocument, SourceLoadError

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_sources(source_dir: str | Path, *, max_chars_per_doc: int | None = None) -> SourceCollection:
    """Load supported local documents from a directory.

    Loading is best-effort: unsupported files are ignored, and extraction
    failures are returned as errors instead of aborting the whole run.
    """

    root = Path(source_dir)
    if not root.exists():
        return SourceCollection(
            root=root,
            errors=[
                SourceLoadError(
                    path=root,
                    message="Source directory does not exist.",
                    exception_type="FileNotFoundError",
                )
            ],
        )

    if not root.is_dir():
        return SourceCollection(
            root=root,
            errors=[
                SourceLoadError(
                    path=root,
                    message="Source path is not a directory.",
                    exception_type="NotADirectoryError",
                )
            ],
        )

    documents: list[SourceDocument] = []
    errors: list[SourceLoadError] = []

    for path in _iter_source_files(root):
        try:
            text = extract_text(path)
            if max_chars_per_doc is not None:
                text = text[:max_chars_per_doc]
            if not text.strip():
                raise ValueError("No extractable text found.")
            documents.append(
                SourceDocument(
                    path=path,
                    title=path.stem,
                    extension=path.suffix.lower(),
                    text=text.strip(),
                )
            )
        except Exception as exc:  # pragma: no cover - exact exception depends on parser internals
            errors.append(
                SourceLoadError(
                    path=path,
                    message=str(exc),
                    exception_type=exc.__class__.__name__,
                )
            )

    return SourceCollection(root=root, documents=documents, errors=errors)


def extract_text(path: str | Path) -> str:
    """Extract text from a supported document."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()

    if suffix in {".md", ".txt"}:
        return source_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        return _extract_pdf_text(source_path)

    raise ValueError(f"Unsupported source extension: {suffix}")


def _iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError("Install pypdf to extract PDF sources.") from exc

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(f"[Page {page_number}]\n{text.strip()}")
    return "\n\n".join(chunks)
