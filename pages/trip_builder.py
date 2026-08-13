"""Trip Builder page: the customer-intake form + follow-up chat."""
import datetime

import streamlit as st

from src.app_core import (
    SEARCH_DEPTH_OPTIONS,
    ask,
    get_current_messages,
    get_knowledge_base_status,
    init_session_state,
)
from src.request_builder import TravelRequest, build_request_text

init_session_state()
kb_ready, _vectorstore, kb_error = get_knowledge_base_status()
retriever_k = SEARCH_DEPTH_OPTIONS[st.session_state.search_depth_label]
messages = get_current_messages()

st.markdown(
    """
    <div class="wt-hero">
        <div class="wt-hero-eyebrow">Trip Builder</div>
        <h1>Plan a trip for your customer</h1>
        <p>Fill in what the customer told you — get a costed, cited package back in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not kb_ready:
    st.info(
        f"The assistant isn't ready yet ({kb_error}). Open **📚 Knowledge Base** "
        "in the sidebar to check its status."
    )
else:
    with st.expander("📝 New package request", expanded=len(messages) == 0):
        with st.form("package_request_form"):
            col1, col2 = st.columns(2)
            with col1:
                destination = st.selectbox(
                    "Destination",
                    ["Goa", "Manali", "Kerala (Munnar & Alleppey)", "Other (type below)"],
                )
                destination_other = st.text_input("If Other, enter destination")
                start_date = st.date_input(
                    "Travel start date",
                    value=datetime.date.today() + datetime.timedelta(days=30),
                )
                duration_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=4)
            with col2:
                num_adults = st.number_input("Number of adults", min_value=1, max_value=20, value=2)
                num_children = st.number_input("Number of children", min_value=0, max_value=20, value=0)
                budget_inr = st.number_input(
                    "Total budget (₹)", min_value=0, max_value=10_000_000, value=50_000, step=1000
                )
                hotel_category = st.selectbox(
                    "Preferred hotel category",
                    ["No preference", "Budget", "Standard", "Deluxe", "Luxury"],
                )

            transport_mode = st.selectbox(
                "Preferred mode of transport", ["No preference", "Flight", "Train", "Bus", "Cab"]
            )
            interests = st.multiselect(
                "Interests",
                ["Sightseeing", "Adventure", "Beaches", "Shopping", "Religious"],
                default=["Sightseeing"],
            )

            submitted = st.form_submit_button("✨ Generate Travel Package", type="primary")

        if submitted:
            req = TravelRequest(
                destination=destination_other.strip() if destination == "Other (type below)" else destination,
                start_date=start_date.strftime("%d %B %Y"),
                duration_days=int(duration_days),
                num_adults=int(num_adults),
                num_children=int(num_children),
                budget_inr=int(budget_inr) if budget_inr else None,
                hotel_category=hotel_category if hotel_category != "No preference" else None,
                transport_mode=transport_mode if transport_mode != "No preference" else None,
                interests=interests,
            )
            ask(build_request_text(req), retriever_k, title=f"{req.destination} trip package")
            st.rerun()

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        page_str = f", page {c['page']}" if c["page"] else ""
                        st.markdown(f"**[{c['index']}] {c['source']}{page_str}**")
                        st.caption(c["snippet"] + "...")

    question = st.chat_input('Ask a follow-up (e.g. "make it cheaper", "what\'s the cancellation policy?")')
    if question:
        ask(question, retriever_k)
        st.rerun()
