"""Document ingestion: load -> chunk -> embed -> persist to Chroma."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import chromadb
from chromadb.config import Settings
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src import config

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 100

Source = Path | BinaryIO


def load_pdf(source: Source) -> str:
    reader = PdfReader(source)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_docx(source: Source) -> str:
    doc = Document(source)
    return "\n".join(p.text for p in doc.paragraphs)


def load_text(source: Source) -> str:
    data = source.read_bytes() if isinstance(source, Path) else source.read()
    return data.decode("utf-8", errors="replace")


LOADERS: dict[str, Callable[[Source], str]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
}
SUPPORTED_EXTENSIONS = tuple(LOADERS)


def load_document(source: str | Path | BinaryIO, filename: str | None = None) -> str:
    """Load text from a file path or binary stream, dispatching on file extension.

    For streams, ``filename`` supplies the extension.
    """
    if isinstance(source, str | Path):
        source = Path(source)
        filename = filename or source.name
    if not filename:
        raise ValueError("filename is required when loading from a stream")
    ext = Path(filename).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader(source)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [c for c in splitter.split_text(text) if c.strip()]


def embed_texts(
    texts: Sequence[str],
    client=None,
    model: str = config.EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    client = client or config.get_openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


def get_collection(persist_dir: str | Path | None = None, name: str = config.DEFAULT_COLLECTION):
    """Return (creating if needed) the persistent Chroma collection."""
    path = Path(persist_dir) if persist_dir else config.chroma_dir()
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )


@dataclass
class IngestResult:
    source: str
    num_chunks: int


def ingest_text(text: str, source: str, collection=None, client=None) -> IngestResult:
    """Chunk, embed and store ``text`` under ``source`` (replacing earlier chunks)."""
    collection = collection if collection is not None else get_collection()
    chunks = chunk_text(text)
    collection.delete(where={"source": source})
    if not chunks:
        return IngestResult(source=source, num_chunks=0)
    embeddings = embed_texts(chunks, client=client)
    collection.upsert(
        ids=[f"{source}::chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
    )
    return IngestResult(source=source, num_chunks=len(chunks))


def ingest_file(path: str | Path, collection=None, client=None) -> IngestResult:
    path = Path(path)
    return ingest_text(load_document(path), path.name, collection=collection, client=client)


def ingest_bytes(filename: str, data: bytes, collection=None, client=None) -> IngestResult:
    """Ingest an in-memory upload (e.g. from Streamlit's file uploader)."""
    text = load_document(io.BytesIO(data), filename=filename)
    return ingest_text(text, Path(filename).name, collection=collection, client=client)


def ingest_directory(directory: str | Path, collection=None, client=None) -> list[IngestResult]:
    directory = Path(directory)
    return [
        ingest_file(p, collection=collection, client=client)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for p in args.paths:
        results = ingest_directory(p) if p.is_dir() else [ingest_file(p)]
        for r in results:
            print(f"{r.source}: {r.num_chunks} chunks")
