from src.ingestion import ingest_text
from src.retrieval import retrieve


def test_empty_collection_returns_no_chunks(collection, fake_openai):
    assert retrieve("anything", collection=collection, client=fake_openai) == []
    assert fake_openai.embeddings.calls == []


def test_returns_chunks_with_metadata_best_first(collection, fake_openai):
    ingest_text("alpha facts", "a.txt", collection=collection, client=fake_openai)
    ingest_text("beta facts", "b.md", collection=collection, client=fake_openai)

    chunks = retrieve("alpha facts", k=5, collection=collection, client=fake_openai)

    assert len(chunks) == 2
    top = chunks[0]
    assert (top.text, top.source, top.chunk_index, top.id) == (
        "alpha facts",
        "a.txt",
        0,
        "a.txt::chunk-0",
    )
    assert top.score > chunks[1].score
    assert top.score > 0.99
    assert top.label == "a.txt (chunk 0)"


def test_k_limits_results(collection, fake_openai):
    for i in range(3):
        ingest_text(f"doc {i}", f"{i}.txt", collection=collection, client=fake_openai)
    assert len(retrieve("doc", k=2, collection=collection, client=fake_openai)) == 2
