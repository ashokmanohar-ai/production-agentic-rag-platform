from app.knowledge.models import DocumentStatus
from app.knowledge.registry import DocumentRegistry


def test_registry_tracks_hash_and_status() -> None:
    registry = DocumentRegistry()
    record = registry.create("doc-1", "a.txt", "text/plain", "abc", 3, "requirements")
    assert record.status == DocumentStatus.queued
    assert registry.duplicate_for("abc") == "doc-1"
    updated = registry.update("doc-1", DocumentStatus.indexed, chunks_indexed=2)
    assert updated.chunks_indexed == 2
    assert registry.get("doc-1") == updated
