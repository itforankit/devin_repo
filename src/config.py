"""Shared configuration: environment variables and lazily-created clients."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_COLLECTION = "documents"


def chat_model() -> str:
    return os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def chroma_dir() -> Path:
    return Path(os.getenv("CHROMA_DIR", "./chroma_db"))


@lru_cache(maxsize=1)
def get_openai_client():
    """Return an OpenAI client using ``OPENAI_API_KEY`` from the environment."""
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or add it to a local .env file.")
    return OpenAI()
