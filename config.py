"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "8"))
    persist_dir: str = os.getenv("PERSIST_DIR", "faiss_index")
    data_dir: str = os.getenv("DATA_DIR", "data/knowledge_base")
    db_path: str = os.getenv("DB_PATH", "wanderly.db")


settings = Settings()
