"""Entry point: page config, shared styling, and navigation.

The actual screens live in pages/trip_builder.py (the day-to-day tool) and
pages/knowledge_base.py (how the data pipeline works, what's indexed, and
how to add/rebuild it). Shared caching and RAG plumbing lives in
src/app_core.py so both pages work off the same cached vectorstore/chain.
"""
from __future__ import annotations

import streamlit as st

from config import settings
from src.app_core import (
    get_knowledge_base_status,
    init_session_state,
    list_conversations,
    start_new_conversation,
    switch_conversation,
)
from src.kb_admin import list_bundled_files

st.set_page_config(page_title="Wanderly Travels — AI Package Assistant", page_icon="🧳", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.25rem; max-width: 1100px; }

    /* ---------- Sidebar shell ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7faf9 0%, #eef4f2 100%);
        border-right: 1px solid #dde6e3;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { background: transparent; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }

    /* ---------- Page nav (Trip Builder / Knowledge Base links) ----------
    Streamlit's *built-in* nav widget (position="sidebar") truncates long
    labels with an ellipsis and can't be restyled reliably, so it's turned
    off (position="hidden") in favor of plain st.page_link calls placed and
    styled here -- full control, no truncation. */
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
        border-radius: 9px; margin-bottom: 0.3rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {
        font-weight: 600; font-size: 0.88rem; white-space: normal;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
        background: #f0fdfa;
    }
    /* The active page renders as plain text (see app.py) rather than a link
    -- st.page_link doesn't expose a stable "current page" attribute to
    style against, so this avoids relying on Streamlit's internal, unstable
    generated class names. */
    .wt-nav-current {
        display: flex; align-items: center; gap: 0.5rem;
        border-radius: 9px; padding: 0.5rem 0.7rem; margin-bottom: 0.3rem;
        background: #ecfdf5; color: #0f766e; font-weight: 600; font-size: 0.88rem;
    }

    /* ---------- Brand header ---------- */
    .wt-brand { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.1rem; }
    .wt-brand-icon {
        display: flex; align-items: center; justify-content: center;
        width: 44px; height: 44px; border-radius: 13px; font-size: 1.5rem;
        background: linear-gradient(135deg, #0f766e, #14b8a6);
        box-shadow: 0 3px 10px rgba(15, 118, 110, 0.28);
    }
    .wt-brand-text h2 { margin: 0; font-size: 1.08rem; font-weight: 700; color: #0f172a; line-height: 1.2; }
    .wt-brand-text span { font-size: 0.76rem; color: #64748b; }

    /* ---------- Status card ---------- */
    .wt-status-card {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0.65rem 0.85rem; border-radius: 11px;
        background: #ffffff; border: 1px solid #a7f3d0;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
        margin-bottom: 1.3rem;
    }
    .wt-status-card.wt-error { border-color: #fecaca; }
    .wt-status-dot { width: 9px; height: 9px; border-radius: 50%; background: #10b981; flex-shrink: 0; }
    .wt-status-card.wt-error .wt-status-dot { background: #ef4444; }
    .wt-status-title { font-weight: 650; font-size: 0.84rem; color: #065f46; }
    .wt-status-card.wt-error .wt-status-title { color: #991b1b; }
    .wt-status-sub { font-size: 0.74rem; color: #4b7f74; margin-top: 1px; }
    .wt-status-card.wt-error .wt-status-sub { color: #b91c1c; }

    /* ---------- Section labels ---------- */
    .wt-section-label {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; color: #94a3b8;
        margin: 1.15rem 0 0.5rem 0;
    }

    /* ---------- Sidebar buttons (secondary style) ---------- */
    [data-testid="stSidebar"] .stButton button {
        border-radius: 9px; border: 1px solid #d6ded9; background: #ffffff;
        color: #1f2937; font-weight: 550; box-shadow: none;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        border-color: #0f766e; color: #0f766e; background: #f0fdfa;
    }

    /* ---------- Card-style containers (admin expanders) ---------- */
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stExpander"] {
        border-radius: 11px; border: 1px solid #dde6e3; background: #ffffff;
    }

    /* ---------- Main hero ---------- */
    .wt-hero { margin-bottom: 1.4rem; }
    .wt-hero-eyebrow {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #0f766e; margin-bottom: 0.2rem;
    }
    .wt-hero h1 { margin: 0; font-size: 1.85rem; color: #0f172a; }
    .wt-hero p { margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.92rem; }

    /* ---------- Document cards (Knowledge Base page) ---------- */
    .wt-doc-card { text-align: left; }
    .wt-doc-label { font-weight: 650; font-size: 0.95rem; color: #0f172a; margin-bottom: 0.3rem; }
    .wt-doc-location {
        display: inline-block; font-size: 0.7rem; font-weight: 600; color: #7c3aed;
        background: #f5f3ff; border: 1px solid #ddd6fe;
        border-radius: 999px; padding: 0.05rem 0.5rem; margin-bottom: 0.4rem;
    }
    .wt-doc-meta { font-size: 0.76rem; color: #94a3b8; }

    /* ---------- Category headings (Knowledge Base page) ---------- */
    .wt-category-heading {
        font-weight: 700; font-size: 1rem; color: #0f172a;
        margin: 1.6rem 0 0.7rem 0; display: flex; align-items: center; gap: 0.5rem;
    }
    .wt-category-heading:first-of-type { margin-top: 0.4rem; }
    .wt-category-count {
        font-size: 0.72rem; font-weight: 600; color: #0f766e;
        background: #f0fdfa; border: 1px solid #a7f3d0;
        border-radius: 999px; padding: 0.05rem 0.55rem;
    }

    /* ---------- Chat & citations ---------- */
    [data-testid="stChatMessage"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

trip_builder_page = st.Page("pages/trip_builder.py", title="Trip Builder", icon="🧳", default=True)
knowledge_base_page = st.Page("pages/knowledge_base.py", title="Knowledge Base", icon="📚")
pages = st.navigation([trip_builder_page, knowledge_base_page], position="hidden")

init_session_state()
kb_ready, vectorstore, kb_error = get_knowledge_base_status()

with st.sidebar:
    st.markdown(
        """
        <div class="wt-brand">
            <div class="wt-brand-icon">🧳</div>
            <div class="wt-brand-text">
                <h2>Wanderly Travels</h2>
                <span>AI Package Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for page, icon, label in [
        (trip_builder_page, "🧳", "Trip Builder"),
        (knowledge_base_page, "📚", "Knowledge Base"),
    ]:
        if pages.title == label:
            st.markdown(f'<div class="wt-nav-current">{icon} {label}</div>', unsafe_allow_html=True)
        else:
            st.page_link(page, label=label, icon=icon, use_container_width=True)

    if not settings.openai_api_key:
        st.markdown(
            '<div class="wt-status-card wt-error"><div class="wt-status-dot"></div>'
            '<div><div class="wt-status-title">Setup needed</div>'
            '<div class="wt-status-sub">Add your API key in .env</div></div></div>',
            unsafe_allow_html=True,
        )
    elif kb_ready and vectorstore is not None:
        doc_count = len(list_bundled_files(settings.data_dir))
        sub = f"{doc_count} documents ready" if doc_count else "Ready to plan trips"
        st.markdown(
            f'<div class="wt-status-card"><div class="wt-status-dot"></div>'
            f'<div><div class="wt-status-title">Assistant ready</div>'
            f'<div class="wt-status-sub">{sub}</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="wt-status-card wt-error"><div class="wt-status-dot"></div>'
            '<div><div class="wt-status-title">Setup error</div>'
            f'<div class="wt-status-sub">{kb_error or "See logs for details"}</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="wt-section-label">Session</div>', unsafe_allow_html=True)
    if st.button("🗑️ New conversation", use_container_width=True):
        start_new_conversation()
        st.rerun()

    st.markdown('<div class="wt-section-label">Recent</div>', unsafe_allow_html=True)
    conversations = list_conversations()
    if len(conversations) <= 1:
        st.caption("Your conversations will show up here.")
    else:
        for session_id, conversation in conversations:
            title = conversation["title"] or "New conversation"
            if session_id == st.session_state.session_id:
                st.markdown(f'<div class="wt-nav-current">💬 {title}</div>', unsafe_allow_html=True)
            elif st.button(f"💬 {title}", key=f"conv_{session_id}", use_container_width=True):
                switch_conversation(session_id)
                st.rerun()

pages.run()
