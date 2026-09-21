"""Tests for report_schema.py - the structured, citation-checked research report.

The report gives every source a short ID like S1, S2, S3. Findings cite those
IDs, and validate_report() warns when a citation is missing or made up.

Run with:  pytest tests/test_report_schema.py
"""
from report_schema import (
    Finding,
    ResearchReport,
    assign_source_ids,
    normalise_confidence,
    render_labelled_corpus,
    report_to_markdown,
    validate_report,
)


def make_sources():
    """Three fake sources: two full pages and one search snippet."""
    return {
        "https://full-a.com": {"url": "https://full-a.com", "tier": "browser", "content": "A body"},
        "https://snip-b.com": {"url": "https://snip-b.com", "tier": "snippet", "content": "b snippet"},
        "https://api-c.com": {"url": "https://api-c.com", "tier": "api", "content": "C body"},
    }


def make_report(findings):
    """A minimal report with the given list of findings."""
    return ResearchReport(executive_summary="s", findings=findings, gaps=[])


# --- assign_source_ids -------------------------------------------------------

def test_assign_source_ids_numbers_full_pages_before_snippets():
    pairs = assign_source_ids(make_sources())

    # pairs looks like: [("S1", entry), ("S2", entry), ("S3", entry)]
    ids = [source_id for source_id, entry in pairs]
    assert ids == ["S1", "S2", "S3"]

    # the snippet is always last, no matter what order it was added in
    last_entry = pairs[-1][1]
    assert last_entry["tier"] == "snippet"


def test_assign_source_ids_with_no_sources():
    assert assign_source_ids({}) == []


# --- render_labelled_corpus --------------------------------------------------

def test_render_labelled_corpus_adds_id_prefix_and_respects_limit():
    # Arrange: 6 sources of 400 characters each
    sources = {}
    for i in range(6):
        url = f"https://s{i}.com"
        sources[url] = {"url": url, "tier": "browser", "content": "x" * 400}
    pairs = assign_source_ids(sources)

    # Act
    small = render_labelled_corpus(pairs, max_chars=1000)
    full = render_labelled_corpus(pairs, max_chars=10_000)

    # Assert
    assert small.startswith("[S1] SOURCE (browser): https://s0.com")
    assert small.count("] SOURCE (") < 6      # limit hit, not all 6 fit
    assert full.count("] SOURCE (") == 6      # big limit, all 6 fit


# --- normalise_confidence ----------------------------------------------------

def test_normalise_confidence_lowercases_and_defaults_to_low():
    assert normalise_confidence("HIGH") == "high"
    assert normalise_confidence("  Medium ") == "medium"
    assert normalise_confidence("pretty sure") == "low"   # unknown word -> low
    assert normalise_confidence("") == "low"


# --- validate_report ---------------------------------------------------------

def test_validate_warns_when_finding_has_no_source():
    finding = Finding(claim="A claim here", detail="d", source_ids=[], confidence="low")
    report = make_report([finding])

    warnings = validate_report(report, {"S1"})

    assert any("cites no source" in w for w in warnings)


def test_validate_warns_when_source_id_does_not_exist():
    finding = Finding(claim="A claim", detail="d", source_ids=["S7"], confidence="low")
    report = make_report([finding])

    warnings = validate_report(report, {"S1", "S2"})

    assert any("not in the data" in w and "S7" in w for w in warnings)


def test_validate_warns_when_finding_has_only_one_source():
    finding = Finding(claim="A claim", detail="d", source_ids=["S1"], confidence="high")
    report = make_report([finding])

    warnings = validate_report(report, {"S1", "S2"})

    assert any("single-sourced" in w for w in warnings)


def test_validate_gives_no_warnings_when_finding_has_two_sources():
    finding = Finding(claim="A claim", detail="d", source_ids=["S1", "S2"], confidence="high")
    report = make_report([finding])

    assert validate_report(report, {"S1", "S2"}) == []


def test_validate_warns_when_report_is_empty():
    warnings = validate_report(make_report([]), {"S1"})

    assert any("no findings" in w for w in warnings)


# --- report_to_markdown ------------------------------------------------------

def test_report_to_markdown_includes_every_section():
    # Arrange
    report = ResearchReport(
        executive_summary="The summary.",
        findings=[Finding(claim="X is Y", detail="Because.", source_ids=["S1"], confidence="medium")],
        gaps=["pricing not found"],
    )
    pairs = assign_source_ids(make_sources())

    # Act
    md = report_to_markdown(report, pairs, objective="Investigate X",
                            warnings=["Finding 1 is single-sourced."])

    # Assert
    assert "**Objective:** Investigate X" in md
    assert "Data-quality flags" in md
    assert "## Executive Summary" in md
    assert "## Key Findings" in md
    assert "[S1](https://full-a.com)" in md
    assert "Confidence: Medium" in md
    assert "- pricing not found" in md
    assert "| S3 | https://snip-b.com | search snippet |" in md


def test_report_to_markdown_with_no_findings_and_no_gaps():
    pairs = assign_source_ids(make_sources())

    md = report_to_markdown(make_report([]), pairs, objective="obj", warnings=[])

    assert "_No findings were produced._" in md
    assert "None identified" in md
