"""Unit tests for the persistent research-history store."""
import json

import pytest

from history_store import (
    ResearchRecord,
    build_record,
    load_records,
    save_record,
    search_records,
)


def test_build_record_stamps_time_and_dedupes_sources():
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
    assert record.created_at.endswith("+00:00")
    assert record.id


def test_save_and_load_round_trips(tmp_path):
    record = build_record("Objective one", "model-x", "Body one", sources=["https://x.com"])
    path = save_record(record, tmp_path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["objective"] == "Objective one"

    loaded = load_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == record


def test_load_records_sorted_newest_first(tmp_path):
    older = ResearchRecord(id="1", objective="old", model="m", report="r",
                           created_at="2026-01-01T00:00:00+00:00")
    newer = ResearchRecord(id="2", objective="new", model="m", report="r",
                           created_at="2026-06-01T00:00:00+00:00")
    save_record(older, tmp_path)
    save_record(newer, tmp_path)

    loaded = load_records(tmp_path)
    assert [r.objective for r in loaded] == ["new", "old"]


def test_load_records_skips_corrupt_files(tmp_path):
    save_record(build_record("Good one", "m", "body"), tmp_path)
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")

    loaded = load_records(tmp_path)
    assert [r.objective for r in loaded] == ["Good one"]


def test_load_records_missing_directory_returns_empty(tmp_path):
    assert load_records(tmp_path / "does-not-exist") == []


@pytest.fixture
def sample_records():
    return [
        ResearchRecord(id="1", objective="BMW M4 performance", model="m",
                       report="The M4 does 0-60 in 3.8s", created_at="2026-03-01T00:00:00+00:00",
                       sources=("https://bmw.com/m4",)),
        ResearchRecord(id="2", objective="Tesla Model 3 range", model="m",
                       report="EPA range is 333 miles", created_at="2026-02-01T00:00:00+00:00",
                       sources=("https://tesla.com/model3",)),
    ]


def test_search_empty_query_returns_all(sample_records):
    assert search_records(sample_records, "   ") == sample_records


def test_search_matches_objective_case_insensitively(sample_records):
    result = search_records(sample_records, "bmw")
    assert [r.id for r in result] == ["1"]


def test_search_matches_report_body(sample_records):
    result = search_records(sample_records, "EPA range")
    assert [r.id for r in result] == ["2"]


def test_search_matches_source_url(sample_records):
    result = search_records(sample_records, "tesla.com")
    assert [r.id for r in result] == ["2"]


def test_search_requires_all_terms(sample_records):
    assert search_records(sample_records, "bmw tesla") == []
    assert [r.id for r in search_records(sample_records, "model 333")] == ["2"]


def test_created_display_falls_back_on_bad_timestamp():
    record = ResearchRecord(id="1", objective="o", model="m", report="r", created_at="not-a-date")
    assert record.created_display == "not-a-date"
