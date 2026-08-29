"""Unit tests for the structured research report: IDs, validation, rendering."""
from report_schema import (
    Finding,
    ResearchReport,
    assign_source_ids,
    normalise_confidence,
    render_labelled_corpus,
    report_to_markdown,
    validate_report,
)


def _sources():
    return {
        "https://full-a.com": {"url": "https://full-a.com", "tier": "browser", "content": "A body"},
        "https://snip-b.com": {"url": "https://snip-b.com", "tier": "snippet", "content": "b snippet"},
        "https://api-c.com": {"url": "https://api-c.com", "tier": "api", "content": "C body"},
    }


def test_assign_source_ids_full_pages_before_snippets():
    pairs = assign_source_ids(_sources())
    assert [sid for sid, _ in pairs] == ["S1", "S2", "S3"]
    # snippet ends up last regardless of insertion order
    assert pairs[-1][1]["tier"] == "snippet"
    assert {e["tier"] for _, e in pairs[:2]} == {"browser", "api"}


def test_assign_source_ids_empty():
    assert assign_source_ids({}) == []


def test_render_labelled_corpus_prefixes_ids_and_caps():
    sources = {
        f"https://s{i}.com": {"url": f"https://s{i}.com", "tier": "browser", "content": "x" * 400}
        for i in range(6)
    }
    pairs = assign_source_ids(sources)
    rendered = render_labelled_corpus(pairs, max_chars=1000)
    assert rendered.startswith("[S1] SOURCE (browser): https://s0.com")
    assert rendered.count("] SOURCE (") < 6  # cap hit
    # full render is not capped
    assert render_labelled_corpus(pairs, max_chars=10_000).count("] SOURCE (") == 6


def test_normalise_confidence():
    assert normalise_confidence("HIGH") == "high"
    assert normalise_confidence("  Medium ") == "medium"
    assert normalise_confidence("pretty sure") == "low"
    assert normalise_confidence("") == "low"


def _report(findings):
    return ResearchReport(executive_summary="s", findings=findings, gaps=[])


def test_validate_flags_uncited_finding():
    r = _report([Finding(claim="A claim here", detail="d", source_ids=[], confidence="low")])
    warnings = validate_report(r, {"S1"})
    assert any("cites no source" in w for w in warnings)


def test_validate_flags_unknown_source():
    r = _report([Finding(claim="A claim", detail="d", source_ids=["S7"], confidence="low")])
    warnings = validate_report(r, {"S1", "S2"})
    assert any("not in the data" in w and "S7" in w for w in warnings)


def test_validate_flags_single_source():
    r = _report([Finding(claim="A claim", detail="d", source_ids=["S1"], confidence="high")])
    warnings = validate_report(r, {"S1", "S2"})
    assert any("single-sourced" in w for w in warnings)


def test_validate_clean_when_multi_sourced():
    r = _report([Finding(claim="A claim", detail="d", source_ids=["S1", "S2"], confidence="high")])
    assert validate_report(r, {"S1", "S2"}) == []


def test_validate_flags_empty_report():
    assert any("no findings" in w for w in validate_report(_report([]), {"S1"}))


def test_report_to_markdown_has_all_sections_and_source_table():
    r = ResearchReport(
        executive_summary="The summary.",
        findings=[Finding(claim="X is Y", detail="Because.", source_ids=["S1"], confidence="medium")],
        gaps=["pricing not found"],
    )
    pairs = assign_source_ids(_sources())
    md = report_to_markdown(r, pairs, objective="Investigate X", warnings=["Finding 1 is single-sourced."])

    assert "**Objective:** Investigate X" in md
    assert "Data-quality flags" in md
    assert "## Executive Summary" in md
    assert "## Key Findings" in md
    assert "[S1](https://full-a.com)" in md
    assert "Confidence: Medium" in md
    assert "- pricing not found" in md
    assert "| S3 | https://snip-b.com | search snippet |" in md


def test_report_to_markdown_handles_no_findings_no_gaps():
    md = report_to_markdown(_report([]), assign_source_ids(_sources()), objective="obj", warnings=[])
    assert "_No findings were produced._" in md
    assert "None identified" in md
