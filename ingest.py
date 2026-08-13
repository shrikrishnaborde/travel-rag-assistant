#!/usr/bin/env python
"""CLI: build (or rebuild) the FAISS index from the travel knowledge base.

Usage:
    python ingest.py --data-dir data/knowledge_base
    python ingest.py --data-dir data/knowledge_base --persist-dir faiss_index --chunk-size 800
"""
from __future__ import annotations

import argparse
import logging
import sys

from config import settings
from src.ingestion import load_and_split
from src.vectorstore import build_vectorstore, get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the travel knowledge base into a FAISS vector index.")
    parser.add_argument("--data-dir", default=settings.data_dir, help="Directory of .pdf/.txt/.md knowledge base files")
    parser.add_argument("--persist-dir", default=settings.persist_dir, help="Where to save the FAISS index")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    logger.info("Loading and chunking documents from %s ...", args.data_dir)
    chunks = load_and_split(args.data_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    logger.info("Embedding %d chunks with %s ...", len(chunks), settings.embedding_model)
    embeddings = get_embeddings(settings.embedding_model, settings.openai_api_key)
    build_vectorstore(chunks, embeddings, args.persist_dir)

    logger.info("Done. Index ready at %s — run `streamlit run app.py` to query it.", args.persist_dir)


if __name__ == "__main__":
    main()
