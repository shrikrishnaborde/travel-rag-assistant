"""Shared state, caching, and RAG plumbing used by both app pages.

Kept separate from the page files so `pages/trip_builder.py` and
`pages/knowledge_base.py` both work off the same cached vectorstore/chain
instead of re-deriving them, and so this glue logic isn't duplicated per page.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import settings
from src import db
from src.ingestion import load_and_split
from src.kb_admin import humanize_filename, save_document_metadata, seed_if_empty
from src.rag_chain import build_rag_chain, get_source_citations
from src.vectorstore import build_vectorstore, get_embeddings, load_vectorstore

SEARCH_DEPTH_OPTIONS = {"Fast": 4, "Balanced (recommended)": 8, "Thorough": 12}

_TITLE_MAX_LEN = 48


def _new_conversation_entry() -> dict:
    return {"title": None, "messages": [], "created_at": datetime.now()}


@st.cache_resource(show_spinner=False)
def get_db_ready() -> bool:
    """Initialize the SQLite engine and seed default categories/locations
    on a fresh database. Cached process-wide so this runs exactly once,
    the same pattern used for the vectorstore below.
    """
    db.init_engine(settings.db_path)
    seed_if_empty()
    return True


def init_session_state() -> None:
    get_db_ready()
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "session_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.session_state.conversations[new_id] = _new_conversation_entry()
    if "kb_version" not in st.session_state:
        st.session_state.kb_version = 0
    if "search_depth_label" not in st.session_state:
        st.session_state.search_depth_label = "Balanced (recommended)"


def get_current_messages() -> list:
    return st.session_state.conversations[st.session_state.session_id]["messages"]


def list_conversations() -> list[tuple[str, dict]]:
    """(session_id, conversation) pairs, most recently created first."""
    items = st.session_state.conversations.items()
    return sorted(items, key=lambda kv: kv[1]["created_at"], reverse=True)


def start_new_conversation() -> None:
    new_id = str(uuid.uuid4())
    st.session_state.session_id = new_id
    st.session_state.conversations[new_id] = _new_conversation_entry()


def switch_conversation(session_id: str) -> None:
    st.session_state.session_id = session_id


@st.cache_resource(show_spinner=False)
def get_embeddings_cached():
    return get_embeddings(settings.embedding_model, settings.openai_api_key)


@st.cache_resource(show_spinner=False)
def get_vectorstore_cached(version: int):
    """Load the persisted index if it exists, otherwise build it from the
    bundled knowledge base. Cached process-wide (keyed on `version`) so it
    survives Streamlit reruns and is shared across sessions -- rebuilding on
    every page interaction would be wasteful and slow. `version` is bumped
    after an admin re-index so the cache picks up the new index.
    """
    embeddings = get_embeddings_cached()
    try:
        return load_vectorstore(settings.persist_dir, embeddings)
    except FileNotFoundError:
        chunks = load_and_split(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
        return build_vectorstore(chunks, embeddings, settings.persist_dir)


@st.cache_resource(show_spinner=False)
def get_chain_cached(version: int, k: int):
    retriever = get_vectorstore_cached(version).as_retriever(search_kwargs={"k": k})
    return build_rag_chain(retriever, settings.chat_model, settings.openai_api_key)


def get_knowledge_base_status() -> tuple[bool, object, str | None]:
    """Returns (ready, vectorstore_or_None, error_message_or_None)."""
    if not settings.openai_api_key:
        return False, None, "OPENAI_API_KEY is not set."
    try:
        return True, get_vectorstore_cached(st.session_state.kb_version), None
    except Exception as e:  # noqa: BLE001 -- surface any provisioning failure to the UI
        return False, None, str(e)


def ask(question: str, k: int, title: str | None = None) -> None:
    conversation = st.session_state.conversations[st.session_state.session_id]
    if not conversation["messages"]:
        # First message of this conversation -- fix its sidebar title now.
        conversation["title"] = title or (
            question if len(question) <= _TITLE_MAX_LEN else question[: _TITLE_MAX_LEN - 1] + "…"
        )

    conversation["messages"].append({"role": "user", "content": question, "citations": None})
    with st.spinner("Looking up travel details and building your answer..."):
        chain = get_chain_cached(st.session_state.kb_version, k)
        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )
    answer = result["answer"]
    citations = get_source_citations(result["source_documents"])
    conversation["messages"].append({"role": "assistant", "content": answer, "citations": citations})


def rebuild_knowledge_base(uploaded_files, category: str | None = None, location: str | None = None) -> None:
    """Save any admin-uploaded files into the library (tagged with `category`
    and `location` if given), then re-index the whole library from disk.

    Uploaded files are written directly into `settings.data_dir` -- not a
    temp copy -- so they show up in the library on every future load, not
    just for this one indexing pass.
    """
    data_path = Path(settings.data_dir)
    for f in uploaded_files or []:
        (data_path / f.name).write_bytes(f.getvalue())
        if category and location:
            save_document_metadata(f.name, humanize_filename(f.name), category, location)

    embeddings = get_embeddings_cached()
    chunks = load_and_split(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
    build_vectorstore(chunks, embeddings, settings.persist_dir)

    get_vectorstore_cached.clear()
    get_chain_cached.clear()
    st.session_state.kb_version += 1
