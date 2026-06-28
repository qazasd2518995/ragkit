"""Document loaders for plain-text, markdown and PDF files."""
from __future__ import annotations

from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


def load_file(path: str | Path) -> str:
    """Return the text content of a supported file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in SUPPORTED:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix} (supported: {sorted(SUPPORTED)})")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pypdf to read PDFs: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def iter_documents(paths: list[str]) -> list[tuple[str, str]]:
    """Expand paths (files or directories) into (text, source) pairs."""
    out: list[tuple[str, str]] = []
    for raw in paths:
        p = Path(raw)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and f.suffix.lower() in SUPPORTED:
                out.append((load_file(f), str(f)))
    return out
