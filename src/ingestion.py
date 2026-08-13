"""Document loading and chunking.

Supports PDF, TXT, and Markdown files out of the box. Each chunk keeps
metadata (source filename, page number when available) so downstream
answers can cite exactly where information came from.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_documents(source_dir: str | Path) -> list[Document]:
    """Load every supported file under `source_dir` into LangChain Documents."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Data directory not found: {source_path}")

    documents: list[Document] = []
    files = sorted(p for p in source_path.rglob("*") if p.is_file())

    for file_path in files:
        loader_cls = _LOADERS.get(file_path.suffix.lower())
        if loader_cls is None:
            logger.warning("Skipping unsupported file type: %s", file_path.name)
            continue

        # TextLoader defaults to the OS locale encoding (cp1252 on Windows),
        # which mangles non-ASCII characters (e.g. "₹", "—"). Force UTF-8.
        kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
        loader = loader_cls(str(file_path), **kwargs)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
        documents.extend(docs)
        logger.info("Loaded %s (%d page(s)/section(s))", file_path.name, len(docs))

    if not documents:
        raise ValueError(f"No supported documents (.pdf, .txt, .md) found in {source_path}")

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into overlapping chunks sized for embedding + retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def load_and_split(
    source_dir: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Convenience wrapper: load then split in one call."""
    documents = load_documents(source_dir)
    return split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
