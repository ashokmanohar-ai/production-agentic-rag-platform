from app.agent.sources import extract_sources


def test_extract_sources_deduplicates_and_preserves_lineage() -> None:
    docs = [
        {"document_id": "d1", "chunk_id": "c1", "title": "Paper", "url": "https://example.test/p", "authors": ["A"], "score": 0.9},
        {"document_id": "d1", "chunk_id": "c1", "title": "Paper duplicate", "score": 0.8},
        {"document_id": "d1", "chunk_id": "c2", "title": "Paper", "score": 0.7},
        {"title": "invalid"},
    ]
    sources = extract_sources(docs)
    assert [(s.document_id, s.chunk_id) for s in sources] == [("d1", "c1"), ("d1", "c2")]
    assert sources[0].url == "https://example.test/p"
