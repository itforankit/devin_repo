from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from src.ingestion import get_collection

DIM = 8


def fake_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest[:DIM]]


class FakeEmbeddings:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": list(input)})
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=fake_vector(t)) for i, t in enumerate(input)]
        )


class FakeOpenAI:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def fake_openai():
    return FakeOpenAI()


@pytest.fixture
def collection(tmp_path):
    return get_collection(tmp_path / "chroma")
