"""Stable links to document locations (PDF page fragments and asset paths)."""

from pathlib import Path


def pdf_doc_link(source_path: Path, page: int | None) -> str:
    """Return a file URI with optional PDF page fragment (common in desktop viewers)."""
    uri = source_path.expanduser().resolve().as_uri()
    if page is not None and page >= 1:
        return f"{uri}#page={page}"
    return uri


def image_file_link(image_path: Path | None) -> str | None:
    """Return file URI for an extracted image, if the path exists."""
    if not image_path:
        return None
    p = Path(image_path).expanduser().resolve()
    if p.is_file():
        return p.as_uri()
    return str(p)
