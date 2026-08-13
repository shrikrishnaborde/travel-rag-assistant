from src.request_builder import TravelRequest, build_request_text


def test_total_travellers_sums_adults_and_children():
    req = TravelRequest(
        destination="Goa",
        start_date="15 December 2026",
        duration_days=4,
        num_adults=2,
        num_children=2,
    )
    assert req.total_travellers == 4


def test_build_request_text_includes_all_provided_fields():
    req = TravelRequest(
        destination="Goa",
        start_date="15 December 2026",
        duration_days=4,
        num_adults=2,
        num_children=2,
        budget_inr=50000,
        hotel_category="Standard",
        transport_mode="Train",
        interests=["Beaches", "Sightseeing"],
    )
    text = build_request_text(req)

    assert "Goa" in text
    assert "15 December 2026" in text
    assert "4 days" in text
    assert "4 total" in text
    assert "₹50,000" in text
    assert "Standard" in text
    assert "Train" in text
    assert "Beaches, Sightseeing" in text


def test_build_request_text_omits_no_preference_transport():
    req = TravelRequest(
        destination="Manali",
        start_date="1 March 2027",
        duration_days=5,
        num_adults=2,
        transport_mode="No preference",
    )
    text = build_request_text(req)
    assert "Preferred mode of transport" not in text
