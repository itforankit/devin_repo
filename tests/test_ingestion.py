from __future__ import annotations

import io

import pytest
from docx import Document

from src import ingestion
from src.config import EMBEDDING_MODEL
from tests.conftest import fake_vector


def _pattern(n: int) -> str:
    return "".join(chr(65 + i % 26) for i in range(n))


class TestChunking:
    def test_short_text_is_single_chunk(self):
        assert ingestion.chunk_text("hello world") == ["hello world"]

    def test_empty_text_yields_no_chunks(self):
        assert ingestion.chunk_text("   \n\n ") == []

    def test_chunks_respect_size_and_overlap(self):
        text = _pattern(2600)
        chunks = ingestion.chunk_text(text)
        assert all(len(c) <= ingestion.CHUNK_SIZE for c in chunks)
        assert chunks[0] == text[:1000]
        assert chunks[1] == text[800:1800]
        for a, b in zip(chunks, chunks[1:], strict=False):
            assert a[-ingestion.CHUNK_OVERLAP :] == b[: ingestion.CHUNK_OVERLAP]
        assert chunks[-1].endswith(text[-100:])

    def test_prefers_paragraph_boundaries(self):
        para_a, para_b = "a" * 600, "b" * 600
        chunks = ingestion.chunk_text(f"{para_a}\n\n{para_b}")
        assert chunks == [para_a, para_b]

    def test_custom_sizes(self):
        chunks = ingestion.chunk_text(_pattern(250), chunk_size=100, chunk_overlap=20)
        assert [len(c) for c in chunks] == [100, 100, 90]
        assert chunks[0][-20:] == chunks[1][:20]


class TestLoaderDispatch:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("doc.pdf", "pdf"),
            ("doc.PDF", "pdf"),
            ("doc.docx", "docx"),
            ("notes.txt", "text"),
            ("README.md", "text"),
        ],
    )
    def test_dispatch_by_extension(self, monkeypatch, filename, expected):
        monkeypatch.setattr(
            ingestion,
            "LOADERS",
            {
                ".pdf": lambda s: "pdf",
                ".docx": lambda s: "docx",
                ".txt": lambda s: "text",
                ".md": lambda s: "text",
            },
        )
        assert ingestion.load_document(io.BytesIO(b""), filename=filename) == expected

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestion.load_document(path)

    def test_stream_requires_filename(self):
        with pytest.raises(ValueError, match="filename"):
            ingestion.load_document(io.BytesIO(b"x"))

    def test_loads_real_txt_and_md(self, tmp_path):
        (tmp_path / "a.txt").write_text("plain text")
        (tmp_path / "b.md").write_text("# Title\nbody")
        assert ingestion.load_document(tmp_path / "a.txt") == "plain text"
        assert ingestion.load_document(str(tmp_path / "b.md")) == "# Title\nbody"

    def test_loads_real_docx(self, tmp_path):
        doc = Document()
        doc.add_paragraph("First paragraph")
        doc.add_paragraph("Second paragraph")
        path = tmp_path / "file.docx"
        doc.save(path)
        assert ingestion.load_document(path) == "First paragraph\nSecond paragraph"

    def test_loads_pdf_via_pypdf(self, monkeypatch):
        class FakePage:
            def __init__(self, text):
                self._text = text

            def extract_text(self):
                return self._text

        class FakeReader:
            def __init__(self, source):
                self.pages = [FakePage("page one"), FakePage(None), FakePage("page three")]

        monkeypatch.setattr(ingestion, "PdfReader", FakeReader)
        text = ingestion.load_document(io.BytesIO(b"%PDF"), filename="x.pdf")
        assert text == "page one\n\n\n\npage three"


class TestEmbeddingAndStorage:
    def test_embed_texts_batches(self, fake_openai):
        texts = [f"t{i}" for i in range(5)]
        vectors = ingestion.embed_texts(texts, client=fake_openai, batch_size=2)
        assert vectors == [fake_vector(t) for t in texts]
        calls = fake_openai.embeddings.calls
        assert [len(c["input"]) for c in calls] == [2, 2, 1]
        assert all(c["model"] == EMBEDDING_MODEL for c in calls)

    def test_ingest_file_stores_chunks_with_metadata(self, tmp_path, fake_openai, collection):
        path = tmp_path / "guide.txt"
        path.write_text(_pattern(2600))
        result = ingestion.ingest_file(path, collection=collection, client=fake_openai)

        assert result.source == "guide.txt"
        assert result.num_chunks == 3
        assert collection.count() == 3
        stored = collection.get(include=["documents", "metadatas", "embeddings"])
        by_id = dict(zip(stored["ids"], range(len(stored["ids"])), strict=True))
        i0 = by_id["guide.txt::chunk-0"]
        assert stored["metadatas"][i0] == {"source": "guide.txt", "chunk_index": 0}
        assert stored["documents"][i0] == _pattern(2600)[:1000]
        assert list(stored["embeddings"][i0]) == pytest.approx(fake_vector(stored["documents"][i0]))

    def test_reingest_replaces_previous_chunks(self, fake_openai, collection):
        ingestion.ingest_text(_pattern(2600), "doc.txt", collection=collection, client=fake_openai)
        assert collection.count() == 3
        ingestion.ingest_text("short now", "doc.txt", collection=collection, client=fake_openai)
        assert collection.count() == 1

    def test_ingest_bytes_and_directory(self, tmp_path, fake_openai, collection):
        ingestion.ingest_bytes(
            "upload.md", b"# hi\nthere", collection=collection, client=fake_openai
        )
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.txt").write_text("alpha")
        (tmp_path / "docs" / "skip.csv").write_text("x")
        results = ingestion.ingest_directory(
            tmp_path / "docs", collection=collection, client=fake_openai
        )
        assert [r.source for r in results] == ["a.txt"]
        sources = {m["source"] for m in collection.get()["metadatas"]}
        assert sources == {"upload.md", "a.txt"}

    def test_empty_document_stores_nothing(self, fake_openai, collection):
        result = ingestion.ingest_text("", "empty.txt", collection=collection, client=fake_openai)
        assert result.num_chunks == 0
        assert fake_openai.embeddings.calls == []
        assert collection.count() == 0
