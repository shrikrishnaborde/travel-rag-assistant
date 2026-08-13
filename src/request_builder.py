"""Turn a structured customer-requirement form into a natural-language request.

The RAG chain expects a single "input" string (it's the same chain used for
free-text chat follow-ups). This module bridges the structured intake form
in the Streamlit app to that string, so retrieval and generation both work
off a normal sentence instead of a raw dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TravelRequest:
    destination: str
    start_date: str  # ISO or free text, e.g. "2026-12-15"
    duration_days: int
    num_adults: int
    num_children: int = 0
    budget_inr: int | None = None
    hotel_category: str | None = None  # Budget / Standard / Deluxe / Luxury
    transport_mode: str | None = None  # Flight / Train / Bus / Cab / No preference
    interests: list[str] = field(default_factory=list)  # Sightseeing, Adventure, Beaches, Shopping, Religious

    @property
    def total_travellers(self) -> int:
        return self.num_adults + self.num_children


def build_request_text(req: TravelRequest) -> str:
    """Render a TravelRequest as a natural-language package request."""
    parts = [
        f"Build a travel package for {req.destination}, starting {req.start_date}, "
        f"for {req.duration_days} days.",
        f"Travellers: {req.total_travellers} total "
        f"({req.num_adults} adult(s), {req.num_children} child(ren)).",
    ]
    if req.budget_inr:
        parts.append(f"Total budget: approximately ₹{req.budget_inr:,}.")
    if req.hotel_category:
        parts.append(f"Preferred hotel category: {req.hotel_category}.")
    if req.transport_mode and req.transport_mode != "No preference":
        parts.append(f"Preferred mode of transport: {req.transport_mode}.")
    if req.interests:
        parts.append(f"Interests: {', '.join(req.interests)}.")
    parts.append(
        "Recommend suitable hotels, a day-wise itinerary, sightseeing "
        "activities matching the interests above, transport recommendations, "
        "an estimated total cost against the stated budget, important travel "
        "instructions, and the applicable cancellation/refund policy."
    )
    return " ".join(parts)
