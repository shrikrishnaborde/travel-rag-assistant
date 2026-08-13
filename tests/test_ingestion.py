from langchain_core.documents import Document

from src.ingestion import load_documents, split_documents


def test_load_documents_from_knowledge_base():
    docs = load_documents("data/knowledge_base")
    sources = {d.metadata["source"] for d in docs}

    # 3 guides + 14 individual hotel properties + 3 transport + 3 sightseeing
    # + 3 itineraries + 2 general (pricing, policies) = 28.
    assert len(docs) == 28
    assert "goa_sea_breeze_inn.md" in sources
    assert "kerala_premium_houseboat.md" in sources
    assert "manali_transport.md" in sources
    assert "travel_policies.md" in sources
    # The old combined-per-destination hotel docs no longer exist.
    assert "goa_hotels.md" not in sources
    assert "kerala_hotels.md" not in sources


def test_split_documents_respects_chunk_size():
    long_text = "Sentence number %d provides some filler content for testing. " * 200
    doc = Document(page_content=long_text, metadata={"source": "synthetic.txt"})

    chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 + 50 for c in chunks)  # allow splitter slack
    assert all(c.metadata["source"] == "synthetic.txt" for c in chunks)


def test_split_documents_assigns_sequential_chunk_ids():
    doc = Document(page_content="short text", metadata={"source": "a.txt"})
    chunks = split_documents([doc], chunk_size=1000, chunk_overlap=0)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
