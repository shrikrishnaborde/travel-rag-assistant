"""Knowledge Base page: the documents behind every recommendation, grouped
by category (with destination as a second, filterable dimension), and a
simple way to add more -- including brand-new categories and destinations,
since both now live in a real database instead of a fixed list. Deliberately
free of any implementation detail (chunking, embeddings, vector search) --
that's plumbing, not something an agency admin managing this content needs
to think about.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import streamlit as st

from config import settings
from src.app_core import get_knowledge_base_status, init_session_state, rebuild_knowledge_base
from src.kb_admin import list_bundled_files, list_categories, list_locations

_ADD_NEW_CATEGORY = "+ Add new category"
_ADD_NEW_LOCATION = "+ Add new destination"

init_session_state()
kb_ready, vectorstore, kb_error = get_knowledge_base_status()


def _open_preview(doc: dict) -> None:
    """Open a document's contents in a wide modal -- a card-width inline
    preview is too narrow to read comfortably."""

    @st.dialog(doc["label"], width="large")
    def _dialog():
        content = Path(settings.data_dir, doc["name"]).read_text(encoding="utf-8")
        st.caption(f'{doc["icon"]} {doc["category"]} · 📍 {doc["location"]} · {doc["type"]} · {doc["size_kb"]} KB')
        st.markdown(content)

    _dialog()


st.markdown(
    """
    <div class="wt-hero">
        <div class="wt-hero-eyebrow">Admin</div>
        <h1>Knowledge Base</h1>
        <p>The documents your assistant draws on to build every travel package.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not settings.openai_api_key:
    st.warning("Add `OPENAI_API_KEY` to a `.env` file (see `.env.example`), then restart the app.")
elif not kb_ready:
    st.error(f"Something's not right with the content library: {kb_error}")

categories = list_categories()  # [(name, icon), ...]
locations = list_locations()  # [name, ...]

bundled = list_bundled_files(settings.data_dir)
present_locations = sorted(
    {f["location"] for f in bundled},
    key=lambda loc: locations.index(loc) if loc in locations else len(locations),
)

col_count, col_filter = st.columns([3, 1])
with col_count:
    st.caption(f"{len(bundled)} documents · organized by category")
with col_filter:
    location_filter = st.selectbox(
        "Filter by destination", ["All destinations"] + present_locations, label_visibility="collapsed"
    )

if location_filter != "All destinations":
    bundled = [f for f in bundled if f["location"] == location_filter]

by_category: dict[str, list[dict]] = defaultdict(list)
for f in bundled:
    by_category[f["category"]].append(f)

for category, icon in categories:
    docs = by_category.get(category)
    if not docs:
        continue
    st.markdown(
        f'<div class="wt-category-heading">{icon} {category} '
        f'<span class="wt-category-count">{len(docs)}</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, f in enumerate(docs):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="wt-doc-card"><div class="wt-doc-label">{f["label"]}</div>'
                    f'<span class="wt-doc-location">📍 {f["location"]}</span>'
                    f'<div class="wt-doc-meta">{f["type"]} · {f["size_kb"]} KB</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("👁️ Preview", key=f"preview_{f['name']}", use_container_width=True):
                    _open_preview(f)

if not bundled:
    st.info(f"No documents tagged **{location_filter}** yet.")

st.divider()

st.markdown('<div class="wt-section-label">Add documents</div>', unsafe_allow_html=True)
st.caption(
    "Have a new destination's hotel list, a pricing sheet, or an updated policy? "
    "Upload it, tag its category and destination (creating a new one if it doesn't "
    "exist yet), and update the library — your assistant will start using it right away."
)
uploaded_files = st.file_uploader(
    "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed"
)

col_category, col_location = st.columns(2)
with col_category:
    category_names = [name for name, _icon in categories]
    category_choice = st.selectbox("Category", category_names + [_ADD_NEW_CATEGORY])
    new_category_name = ""
    if category_choice == _ADD_NEW_CATEGORY:
        new_category_name = st.text_input("New category name", placeholder="e.g. Visa Assistance")
with col_location:
    location_choice = st.selectbox("Destination", locations + [_ADD_NEW_LOCATION])
    new_location_name = ""
    if location_choice == _ADD_NEW_LOCATION:
        new_location_name = st.text_input("New destination name", placeholder="e.g. Ooty")
st.caption("Applied to every file uploaded above.")

if st.button("📥 Update library", type="primary", disabled=not settings.openai_api_key):
    final_category = new_category_name.strip() if category_choice == _ADD_NEW_CATEGORY else category_choice
    final_location = new_location_name.strip() if location_choice == _ADD_NEW_LOCATION else location_choice

    if not uploaded_files:
        st.warning("Choose at least one file to upload first.")
    elif category_choice == _ADD_NEW_CATEGORY and not final_category:
        st.warning("Enter a category name, or pick one from the list.")
    elif location_choice == _ADD_NEW_LOCATION and not final_location:
        st.warning("Enter a destination name, or pick one from the list.")
    else:
        with st.spinner("Updating your library..."):
            rebuild_knowledge_base(uploaded_files, category=final_category, location=final_location)
        st.success(f"Library updated — added under **{final_category} · {final_location}**.")
        st.rerun()
