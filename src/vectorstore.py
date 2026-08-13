"""FAISS vector store creation, persistence, and loading."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_embeddings(model: str, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model, api_key=api_key)


def build_vectorstore(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    persist_dir: str | Path,
) -> FAISS:
    """Embed chunks, build a FAISS index, and persist it to disk."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_path))
    logger.info("Saved FAISS index (%d vectors) to %s", len(chunks), persist_path)
    return vectorstore


def load_vectorstore(persist_dir: str | Path, embeddings: OpenAIEmbeddings) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    persist_path = Path(persist_dir)
    if not (persist_path / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index found at {persist_path}. Run ingest.py first."
        )
    return FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
