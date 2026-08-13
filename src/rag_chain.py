"""Conversational RAG chain: history-aware retrieval + citation-grounded generation.

Pipeline:
  1. A history-aware retriever rewrites the latest user question using the
     chat history (so "what about its pricing?" resolves to a standalone
     query) before hitting the vector store.
  2. Retrieved chunks are numbered and injected into the prompt so the LLM
     can cite them inline as [1], [2], ...
  3. `RunnableWithMessageHistory` keeps per-session chat memory in process
     memory, keyed by session_id.

The chain is built with plain LCEL (`RunnablePassthrough.assign`) rather than
the `create_stuff_documents_chain` helper, because we need the raw retrieved
`Document` objects (for the UI's source panel) alongside the numbered context
string (for the LLM's citation tags) in the same output.
"""
from __future__ import annotations

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question as a standalone question that can be understood without "
            "the chat history. Do NOT answer the question, only reformulate it "
            "if needed, otherwise return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Wanderly Travels' AI travel-package assistant. You help "
            "travel agents put together personalized travel packages for "
            "customers, using only the knowledge base sources provided below "
            "(hotel details, transport options, sightseeing packages, past "
            "itineraries, destination guides, pricing rules, and travel "
            "policies).\n\n"
            "Rules:\n"
            "- Only use facts found in the numbered context sources below. "
            "Never invent a hotel, price, or policy that isn't in the context.\n"
            "- Every factual claim (a price, a hotel name, a policy term) must "
            "be followed by its source tag, e.g. [1] or [1][3], matching the "
            "numbered sources.\n"
            "- If the customer's budget doesn't comfortably fit, say so "
            "explicitly and suggest the closest grounded alternative (cheaper "
            "hotel tier, different transport, shorter trip) rather than "
            "silently exceeding the budget.\n"
            "- If the context does not contain enough information to answer, "
            "say so clearly instead of guessing.\n\n"
            "When asked to build, recommend, or generate a travel package, "
            "structure your answer with these sections (omit a section only "
            "if the context truly has nothing relevant to it):\n"
            "  ## Recommended Hotel(s)\n"
            "  ## Day-wise Itinerary\n"
            "  ## Sightseeing & Activities\n"
            "  ## Transport Recommendations\n"
            "  ## Estimated Total Cost\n"
            "  ## Important Travel Instructions\n"
            "  ## Cancellation & Refund Policy\n"
            "For follow-up questions that are not a full package request "
            "(e.g. \"what's the refund policy?\"), answer directly and "
            "concisely instead of repeating the full structure.\n\n"
            "Context sources:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_SESSION_STORE: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]


def format_docs_for_citation(docs: list[Document]) -> str:
    """Render retrieved chunks as a numbered block the LLM can cite by index."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if page is not None else source
        lines.append(f"[{i}] (Source: {location})\n{doc.page_content}")
    return "\n\n".join(lines)


def get_source_citations(docs: list[Document]) -> list[dict]:
    """Build UI-friendly citation metadata matching the [n] tags in an answer."""
    citations = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        citations.append(
            {
                "index": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": page + 1 if page is not None else None,
                "snippet": doc.page_content[:300].strip(),
            }
        )
    return citations


def _build_history_aware_retriever(llm: ChatOpenAI, retriever: VectorStoreRetriever):
    """Rewrite the question using chat history, then retrieve.

    Skips the rewrite step entirely on the first turn (no history yet), since
    an LLM call to "contextualize" a question with no history is wasted work.
    """
    contextualize_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    return RunnableBranch(
        (
            lambda x: not x.get("chat_history"),
            (lambda x: x["input"]) | retriever,
        ),
        contextualize_chain | retriever,
    )


def build_rag_chain(
    retriever: VectorStoreRetriever,
    chat_model: str,
    api_key: str,
    temperature: float = 0.0,
) -> RunnableWithMessageHistory:
    """Assemble the full conversational, citation-aware RAG chain.

    Invoking the returned chain with
        {"input": "<question>"}, config={"configurable": {"session_id": "<id>"}}
    yields a dict with:
        - "answer": the generated, citation-tagged answer string
        - "source_documents": the retrieved Document chunks, in citation order
    """
    llm = ChatOpenAI(model=chat_model, api_key=api_key, temperature=temperature)

    history_aware_retriever = _build_history_aware_retriever(llm, retriever)
    answer_generation = ANSWER_PROMPT | llm | StrOutputParser()

    retrieval_chain = (
        RunnablePassthrough.assign(source_documents=history_aware_retriever)
        .assign(context=lambda x: format_docs_for_citation(x["source_documents"]))
        .assign(answer=answer_generation)
    )

    return RunnableWithMessageHistory(
        retrieval_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
