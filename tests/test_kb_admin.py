import pytest

from src import db, kb_admin
from src.kb_admin import (
    get_or_create_category,
    get_or_create_location,
    list_bundled_files,
    list_categories,
    list_locations,
    save_document_metadata,
)


@pytest.fixture(autouse=True)
def _fresh_seeded_db():
    """Every test gets its own in-memory database, seeded the same way the
    real app bootstraps on first run -- full isolation, no shared state
    between tests or with the real wanderly.db file."""
    db.init_engine(":memory:")
    kb_admin.seed_if_empty()
    yield


def test_list_bundled_files_matches_knowledge_base_dir():
    files = list_bundled_files("data/knowledge_base")
    names = {f["name"] for f in files}
    labels = {f["label"] for f in files}
    categories = {f["category"] for f in files}
    locations = {f["location"] for f in files}

    assert len(files) == 28
    assert "goa_sea_breeze_inn.md" in names
    assert "kerala_premium_houseboat.md" in names
    assert "travel_policies.md" in names
    assert "Sea Breeze Inn" in labels
    assert "The Grand Riviera" in labels
    assert "Premium Houseboat" in labels
    assert categories == {name for name, _icon in list_categories()} - {"Other"}
    assert locations == {"Goa", "Manali", "Kerala", "General"}
    assert all(f["type"] == "MD" for f in files)
    assert all(f["size_kb"] > 0 for f in files)
    assert all(f["icon"] for f in files)


def test_hotels_are_split_one_document_per_property_not_one_combined_doc():
    files = list_bundled_files("data/knowledge_base")
    hotel_docs = [f for f in files if f["category"] == "Hotels"]
    by_location = {}
    for f in hotel_docs:
        by_location.setdefault(f["location"], []).append(f["label"])

    # 5 Goa + 4 Manali + 5 Kerala properties, each its own document.
    assert len(hotel_docs) == 14
    assert len(by_location["Goa"]) == 5
    assert len(by_location["Manali"]) == 4
    assert len(by_location["Kerala"]) == 5
    assert len(hotel_docs) == len({f["name"] for f in hotel_docs})  # every property is a distinct file


def test_save_and_load_document_metadata_roundtrip(tmp_path):
    (tmp_path / "ooty_special_offers.pdf").write_bytes(b"content")

    save_document_metadata("ooty_special_offers.pdf", "Ooty Special Offers", "Hotels", "Ooty")

    files = list_bundled_files(tmp_path)
    assert files == [
        {
            "name": "ooty_special_offers.pdf",
            "label": "Ooty Special Offers",
            "category": "Hotels",
            "icon": "🏨",
            "location": "Ooty",
            "type": "PDF",
            "size_kb": round(len(b"content") / 1024, 1),
        }
    ]


def test_save_document_metadata_creates_brand_new_category_and_location():
    """The whole point of moving off a hardcoded dict: an admin can file a
    document under a category/destination that never existed before."""
    before_categories = {name for name, _icon in list_categories()}
    before_locations = set(list_locations())
    assert "Visa Assistance" not in before_categories
    assert "Ooty" not in before_locations

    save_document_metadata("ooty_visa.md", "Ooty Visa Requirements", "Visa Assistance", "Ooty")

    assert "Visa Assistance" in {name for name, _icon in list_categories()}
    assert "Ooty" in list_locations()


def test_save_document_metadata_updates_existing_document_in_place():
    save_document_metadata("goa_sea_breeze_inn.md", "Sea Breeze Inn", "Hotels", "Goa")
    save_document_metadata("goa_sea_breeze_inn.md", "Sea Breeze Inn (Updated)", "Hotels", "Goa")

    files = list_bundled_files("data/knowledge_base")
    match = [f for f in files if f["name"] == "goa_sea_breeze_inn.md"]
    assert match[0]["label"] == "Sea Breeze Inn (Updated)"


def test_get_or_create_category_is_idempotent():
    get_or_create_category("Visa Assistance", icon="🛂")
    get_or_create_category("Visa Assistance", icon="🛂")

    matches = [name for name, icon in list_categories() if name == "Visa Assistance"]
    assert matches == ["Visa Assistance"]


def test_get_or_create_location_is_idempotent():
    get_or_create_location("Ooty")
    get_or_create_location("Ooty")

    assert list_locations().count("Ooty") == 1


def test_uploaded_file_without_saved_metadata_defaults_to_other(tmp_path):
    (tmp_path / "mystery.md").write_text("content", encoding="utf-8")

    files = list_bundled_files(tmp_path)

    other_icon = dict(list_categories())["Other"]
    assert files[0]["category"] == "Other"
    assert files[0]["location"] == "Other"
    assert files[0]["icon"] == other_icon
    assert files[0]["label"] == "Mystery"


def test_locations_list_includes_all_destinations_and_general():
    assert list_locations() == ["Goa", "Manali", "Kerala", "General", "Other"]


class _FakeDocstore:
    def __init__(self, docs_by_id):
        self._dict = docs_by_id


class _FakeDoc:
    def __init__(self, source):
        self.metadata = {"source": source}


class _FakeVectorstore:
    def __init__(self, sources):
        self.docstore = _FakeDocstore({str(i): _FakeDoc(s) for i, s in enumerate(sources)})


def test_summarize_indexed_sources_counts_chunks_per_file():
    from src.kb_admin import summarize_indexed_sources

    vs = _FakeVectorstore(["goa_hotels.md", "goa_hotels.md", "goa_transport.md"])
    summary = summarize_indexed_sources(vs)

    assert summary == [
        {"source": "goa_hotels.md", "chunks_indexed": 2},
        {"source": "goa_transport.md", "chunks_indexed": 1},
    ]


def test_summarize_indexed_sources_handles_missing_docstore():
    from src.kb_admin import summarize_indexed_sources

    class _Empty:
        pass

    assert summarize_indexed_sources(_Empty()) == []
