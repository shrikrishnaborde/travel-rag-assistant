"""Central configuration, loaded from environment variables / .env / Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _resolve(key: str, default: str = "") -> str:
    """Env var / .env value if set, otherwise fall back to Streamlit Cloud's
    st.secrets. Streamlit only mirrors secrets.toml into os.environ once
    st.secrets has actually been accessed, which isn't guaranteed to have
    happened yet at import time -- reading st.secrets directly here is the
    reliable way to pick up a Cloud deployment's secrets on first load.
    """
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(key, default))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = _resolve("OPENAI_API_KEY")
    chat_model: str = _resolve("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = _resolve("EMBEDDING_MODEL", "text-embedding-3-small")
    chunk_size: int = int(_resolve("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(_resolve("CHUNK_OVERLAP", "150"))
    retriever_k: int = int(_resolve("RETRIEVER_K", "8"))
    persist_dir: str = _resolve("PERSIST_DIR", "faiss_index")
    data_dir: str = _resolve("DATA_DIR", "data/knowledge_base")
    db_path: str = _resolve("DB_PATH", "wanderly.db")


settings = Settings()
