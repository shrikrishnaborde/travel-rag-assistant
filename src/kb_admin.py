"""Knowledge-base admin operations: categories, destinations, and documents.

This is the business-logic layer between the Streamlit pages and `src/db.py`
(the persistence layer). Deliberately framework-agnostic (no Streamlit
import) so it's testable on its own.

Categories and destinations are no longer a fixed set baked into app code --
they're rows in a real SQLite database (see `src/db.py`), seeded once with
sensible defaults and freely extensible from the admin UI from then on.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src import db

# Seed data: the initial set of categories/destinations and how the bundled
# sample documents are tagged. This runs once (see `seed_if_empty`) to
# populate empty tables -- after that, all of it lives in the database and
# can be extended at runtime, not just at this fixed set.
_SEED_CATEGORIES = [
    ("Destination Guides", "🌍"),
    ("Hotels", "🏨"),
    ("Transport Options", "🚗"),
    ("Sightseeing & Activities", "🗺️"),
    ("Sample Itineraries", "📋"),
    ("Pricing & Add-ons", "💰"),
    ("Policies & Cancellations", "📜"),
    ("Other", "📄"),
]

_SEED_LOCATIONS = ["Goa", "Manali", "Kerala", "General", "Other"]

_SEED_DOCUMENTS: dict[str, dict[str, str]] = {
    "goa_guide.md": {"category": "Destination Guides", "location": "Goa", "title": "Goa Destination Guide"},
    "manali_guide.md": {"category": "Destination Guides", "location": "Manali", "title": "Manali Destination Guide"},
    "kerala_guide.md": {"category": "Destination Guides", "location": "Kerala", "title": "Kerala Destination Guide"},
    "goa_sea_breeze_inn.md": {"category": "Hotels", "location": "Goa", "title": "Sea Breeze Inn"},
    "goa_palm_grove_residency.md": {"category": "Hotels", "location": "Goa", "title": "Palm Grove Residency"},
    "goa_coral_bay_resort.md": {"category": "Hotels", "location": "Goa", "title": "Coral Bay Resort"},
    "goa_anjuna_sands_boutique_hotel.md": {"category": "Hotels", "location": "Goa", "title": "Anjuna Sands Boutique Hotel"},
    "goa_grand_riviera.md": {"category": "Hotels", "location": "Goa", "title": "The Grand Riviera"},
    "manali_hillside_cottages.md": {"category": "Hotels", "location": "Manali", "title": "Hillside Cottages"},
    "manali_apple_orchard_retreat.md": {"category": "Hotels", "location": "Manali", "title": "Apple Orchard Retreat"},
    "manali_solang_valley_view_resort.md": {"category": "Hotels", "location": "Manali", "title": "Solang Valley View Resort"},
    "manali_snow_peak_luxury_resort.md": {"category": "Hotels", "location": "Manali", "title": "Snow Peak Luxury Resort"},
    "kerala_tea_valley_homestay.md": {"category": "Hotels", "location": "Kerala", "title": "Tea Valley Homestay"},
    "kerala_misty_hills_resort.md": {"category": "Hotels", "location": "Kerala", "title": "Misty Hills Resort"},
    "kerala_lakeview_backwater_resort.md": {"category": "Hotels", "location": "Kerala", "title": "Lakeview Backwater Resort"},
    "kerala_premium_houseboat.md": {"category": "Hotels", "location": "Kerala", "title": "Premium Houseboat"},
    "kerala_cardamom_county_luxury_resort.md": {"category": "Hotels", "location": "Kerala", "title": "Cardamom County Luxury Resort"},
    "goa_transport.md": {"category": "Transport Options", "location": "Goa", "title": "Goa Transport"},
    "manali_transport.md": {"category": "Transport Options", "location": "Manali", "title": "Manali Transport"},
    "kerala_transport.md": {"category": "Transport Options", "location": "Kerala", "title": "Kerala Transport"},
    "goa_sightseeing.md": {"category": "Sightseeing & Activities", "location": "Goa", "title": "Goa Sightseeing & Activities"},
    "manali_sightseeing.md": {"category": "Sightseeing & Activities", "location": "Manali", "title": "Manali Sightseeing & Activities"},
    "kerala_sightseeing.md": {"category": "Sightseeing & Activities", "location": "Kerala", "title": "Kerala Sightseeing & Activities"},
    "goa_itinerary.md": {"category": "Sample Itineraries", "location": "Goa", "title": "Goa Itinerary"},
    "manali_itinerary.md": {"category": "Sample Itineraries", "location": "Manali", "title": "Manali Itinerary"},
    "kerala_itinerary.md": {"category": "Sample Itineraries", "location": "Kerala", "title": "Kerala Itinerary"},
    "pricing_and_addons.md": {"category": "Pricing & Add-ons", "location": "General", "title": "Pricing & Add-ons"},
    "travel_policies.md": {"category": "Policies & Cancellations", "location": "General", "title": "Policies & Cancellations"},
}


def humanize_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return stem.strip().title()


def seed_if_empty() -> None:
    """Populate categories/locations/documents on a fresh database. No-op
    once seeded -- everything from here on is real rows, editable at runtime."""
    with db.get_session() as session:
        if session.query(db.Category).count() > 0:
            return

        categories = {name: db.Category(name=name, icon=icon) for name, icon in _SEED_CATEGORIES}
        locations = {name: db.Location(name=name) for name in _SEED_LOCATIONS}
        session.add_all(categories.values())
        session.add_all(locations.values())
        session.flush()  # assign primary keys before referencing them below

        for filename, meta in _SEED_DOCUMENTS.items():
            session.add(
                db.Document(
                    filename=filename,
                    title=meta["title"],
                    category=categories[meta["category"]],
                    location=locations[meta["location"]],
                )
            )


def list_categories() -> list[tuple[str, str]]:
    """[(name, icon), ...] in creation order."""
    with db.get_session() as session:
        rows = session.query(db.Category).order_by(db.Category.id).all()
        return [(c.name, c.icon) for c in rows]


def list_locations() -> list[str]:
    """Location names in creation order."""
    with db.get_session() as session:
        rows = session.query(db.Location).order_by(db.Location.id).all()
        return [loc.name for loc in rows]


def get_or_create_category(name: str, icon: str = "📄") -> None:
    with db.get_session() as session:
        existing = session.query(db.Category).filter_by(name=name).one_or_none()
        if existing is None:
            session.add(db.Category(name=name, icon=icon))


def get_or_create_location(name: str) -> None:
    with db.get_session() as session:
        existing = session.query(db.Location).filter_by(name=name).one_or_none()
        if existing is None:
            session.add(db.Location(name=name))


def save_document_metadata(filename: str, title: str, category_name: str, location_name: str) -> None:
    """Create or update a document's category/location tag, creating the
    category/location itself first if it's new (e.g. an admin filing a
    document under a brand-new destination)."""
    with db.get_session() as session:
        category = session.query(db.Category).filter_by(name=category_name).one_or_none()
        if category is None:
            category = db.Category(name=category_name, icon="📄")
            session.add(category)
            session.flush()

        location = session.query(db.Location).filter_by(name=location_name).one_or_none()
        if location is None:
            location = db.Location(name=location_name)
            session.add(location)
            session.flush()

        doc = session.query(db.Document).filter_by(filename=filename).one_or_none()
        if doc is None:
            session.add(db.Document(filename=filename, title=title, category=category, location=location))
        else:
            doc.title = title
            doc.category = category
            doc.location = location


def list_bundled_files(data_dir: str | Path) -> list[dict]:
    """List the documents currently in the content library -- one row per
    file actually present on disk, joined with its category/location/title
    from the database (falling back to a humanized filename under "Other"
    for any file that predates a database record)."""
    with db.get_session() as session:
        doc_rows = {d.filename: d for d in session.query(db.Document).all()}

        files = []
        for path in sorted(Path(data_dir).glob("*")):
            if not path.is_file():
                continue
            doc = doc_rows.get(path.name)
            category = doc.category.name if doc else "Other"
            icon = doc.category.icon if doc else "📄"
            location = doc.location.name if doc else "Other"
            label = doc.title if doc else humanize_filename(path.name)
            files.append(
                {
                    "name": path.name,
                    "label": label,
                    "category": category,
                    "icon": icon,
                    "location": location,
                    "type": path.suffix.lstrip(".").upper(),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
        return files


def summarize_indexed_sources(vectorstore) -> list[dict]:
    """Count indexed chunks per source file from a built FAISS vectorstore.

    Internal/diagnostic use only (not shown in the main UI, which stays in
    business language) -- LangChain's FAISS wrapper doesn't expose a public
    "list all documents" method, so this reads the underlying docstore
    directly.
    """
    docstore = getattr(vectorstore, "docstore", None)
    doc_map = getattr(docstore, "_dict", None) or {}

    counts = Counter(
        doc.metadata.get("source", "unknown") for doc in doc_map.values()
    )
    return [
        {"source": source, "chunks_indexed": count}
        for source, count in sorted(counts.items())
    ]
