"""SQLite-backed persistence for the knowledge base's categorization metadata.

Document *content* stays on disk as files in data/knowledge_base -- that's
what the RAG ingestion pipeline reads. This module only tracks which
category and destination each file belongs to, and the set of valid
categories/destinations themselves, in real tables an admin can add rows to
at runtime -- replacing what used to be a hardcoded Python dict and a
`categories.json` sidecar file.

Schema: `categories` and `locations` are lookup tables (name + optional
icon); `documents` maps each filename to one category and one location via
foreign keys, so "what categories exist" and "what's tagged under Goa" are
ordinary queries, not app-code constants.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    icon: Mapped[str] = mapped_column(String(8), default="📄")


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True)
    title: Mapped[str] = mapped_column(String(255))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    category: Mapped[Category] = relationship()
    location: Mapped[Location] = relationship()


_engine = None
_session_factory: sessionmaker | None = None


def init_engine(db_path: str | Path) -> None:
    """(Re-)point this module at a database file (or ':memory:') and create
    tables if they don't exist yet. Safe to call more than once -- tests
    call it per-test with a fresh in-memory database for isolation."""
    global _engine, _session_factory
    _engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(_engine)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized -- call init_engine() first.")
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
