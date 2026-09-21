"""Tests for history_store.py - saving, loading and searching past research runs.

Some tests take an argument called `tmp_path`. This is a built-in pytest helper:
pytest creates a fresh empty folder for the test and passes it in, so the tests
can write files without touching the real `history/` folder.

Run with:  pytest tests/test_history_store.py
"""
import json

from history_store import ResearchRecord, build_record, load_records, save_record, search_records


def make_sample_records():
    """Two fake records that the search tests use."""
    bmw = ResearchRecord(
        id="1",
        objective="BMW M4 performance",
        model="m",
        report="The M4 does 0-60 in 3.8s",
        created_at="2026-03-01T00:00:00+00:00",
        sources=("https://bmw.com/m4",),
    )
    tesla = ResearchRecord(
        id="2",
        objective="Tesla Model 3 range",
        model="m",
        report="EPA range is 333 miles",
        created_at="2026-02-01T00:00:00+00:00",
        sources=("https://tesla.com/model3",),
    )
    return [bmw, tesla]


# --- build_record ------------------------------------------------------------

def test_build_record_trims_objective_and_removes_duplicate_sources():
    record = build_record(
        "  Identify the BMW M4 specs  ",
        "gemini-3.1-flash-lite-preview",
        "# Report",
        sources=["https://a.com", "https://a.com", "https://b.com", ""],
        queries_run=4,
        chars_collected=1234,
    )

    assert record.objective == "Identify the BMW M4 specs"
    assert record.sources == ("https://a.com", "https://b.com")
    assert record.queries_run == 4
    assert record.created_at.endswith("+00:00")   # timestamp is in UTC
    assert record.id != ""                         # an id was generated


# --- save_record / load_records ---------------------------------------------

def test_save_then_load_gives_back_the_same_record(tmp_path):
    # Arrange
    record = build_record("Objective one", "model-x", "Body one", sources=["https://x.com"])

    # Act
    path = save_record(record, tmp_path)
    loaded = load_records(tmp_path)

    # Assert
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["objective"] == "Objective one"
    assert loaded == [record]


def test_load_records_returns_newest_first(tmp_path):
    older = ResearchRecord(id="1", objective="old", model="m", report="r",
                           created_at="2026-01-01T00:00:00+00:00")
    newer = ResearchRecord(id="2", objective="new", model="m", report="r",
                           created_at="2026-06-01T00:00:00+00:00")
    save_record(older, tmp_path)
    save_record(newer, tmp_path)

    loaded = load_records(tmp_path)

    assert [r.objective for r in loaded] == ["new", "old"]


def test_load_records_skips_broken_json_files(tmp_path):
    save_record(build_record("Good one", "m", "body"), tmp_path)
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")

    loaded = load_records(tmp_path)

    assert [r.objective for r in loaded] == ["Good one"]


def test_load_records_from_missing_folder_returns_empty_list(tmp_path):
    assert load_records(tmp_path / "does-not-exist") == []


# --- search_records ----------------------------------------------------------

def test_search_with_blank_query_returns_everything():
    records = make_sample_records()

    assert search_records(records, "   ") == records


def test_search_matches_objective_ignoring_case():
    records = make_sample_records()

    result = search_records(records, "bmw")

    assert [r.id for r in result] == ["1"]


def test_search_matches_report_text():
    records = make_sample_records()

    result = search_records(records, "EPA range")

    assert [r.id for r in result] == ["2"]


def test_search_matches_source_url():
    records = make_sample_records()

    result = search_records(records, "tesla.com")

    assert [r.id for r in result] == ["2"]


def test_search_needs_every_word_to_match():
    records = make_sample_records()

    # "bmw" is only in record 1 and "tesla" only in record 2, so nothing has both
    assert search_records(records, "bmw tesla") == []
    # "model" and "333" are both in record 2
    assert [r.id for r in search_records(records, "model 333")] == ["2"]


# --- created_display ---------------------------------------------------------

def test_created_display_shows_raw_text_when_date_is_invalid():
    record = ResearchRecord(id="1", objective="o", model="m", report="r", created_at="not-a-date")

    assert record.created_display == "not-a-date"
