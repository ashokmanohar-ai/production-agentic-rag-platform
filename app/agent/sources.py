from collections.abc import Iterable
from app.models import SourceItem


def extract_sources(documents: Iterable[dict[str, object]]) -> list[SourceItem]:
    """Create deterministic source lineage from structured retrieval documents."""
    seen: set[tuple[str, str]] = set()
    sources: list[SourceItem] = []
    for doc in documents:
        document_id = str(doc.get("document_id", "")).strip()
        chunk_id = str(doc.get("chunk_id", "")).strip()
        if not document_id or not chunk_id:
            continue
        key = (document_id, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        raw_authors = doc.get("authors", [])
        authors = [str(v) for v in raw_authors] if isinstance(raw_authors, list) else []
        sources.append(SourceItem(
            document_id=document_id,
            chunk_id=chunk_id,
            title=str(doc.get("title", "Untitled")),
            url=str(doc["url"]) if doc.get("url") else None,
            authors=authors,
            score=float(doc.get("score", 0.0)),
            category=str(doc["category"]) if doc.get("category") else None,
        ))
    return sources
