from langchain_core.documents import Document

from src.rag_chain import format_docs_for_citation, get_source_citations


def _sample_docs():
    return [
        Document(page_content="The Team plan costs $349/month.", metadata={"source": "pricing_plans.txt"}),
        Document(page_content="Uptime SLA is 99.95%.", metadata={"source": "support_policy.txt", "page": 0}),
    ]


def test_format_docs_for_citation_numbers_sequentially():
    formatted = format_docs_for_citation(_sample_docs())
    assert "[1] (Source: pricing_plans.txt)" in formatted
    assert "[2] (Source: support_policy.txt, page 1)" in formatted
    assert "$349/month" in formatted
    assert "99.95%" in formatted


def test_format_docs_for_citation_empty_list():
    assert format_docs_for_citation([]) == ""


def test_get_source_citations_matches_index_and_page_offset():
    citations = get_source_citations(_sample_docs())

    assert len(citations) == 2
    assert citations[0] == {
        "index": 1,
        "source": "pricing_plans.txt",
        "page": None,
        "snippet": "The Team plan costs $349/month.",
    }
    assert citations[1]["page"] == 1  # stored 0-indexed, displayed 1-indexed
    assert citations[1]["index"] == 2
