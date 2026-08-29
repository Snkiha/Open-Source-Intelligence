"""Structured research report: schema, citation IDs, validation, and rendering.

The reporter node asks the LLM for a `ResearchReport` (executive summary + findings
that each cite source IDs + gaps), then this module checks every finding cites a
real source and renders the result to Markdown with a source table. Kept
import-safe (no Streamlit / Playwright) so it can be unit-tested directly.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from corpus import SourceEntry

# Friendly labels for the source table.
_TIER_LABEL = {
    "browser": "full page",
    "api": "reader API",
    "snippet": "search snippet",
}
_CONFIDENCE_VALUES = ("high", "medium", "low")


class Finding(BaseModel):
    claim: str = Field(description="One specific, self-contained factual claim (a single sentence).")
    detail: str = Field(
        description="1-3 sentences of supporting detail, paraphrased or quoted from the cited sources."
    )
    source_ids: List[str] = Field(
        description="IDs of the sources that support this claim, e.g. ['S1', 'S3']. "
        "Every finding MUST cite at least one source ID that appears in the source data."
    )
    confidence: str = Field(
        description="'high', 'medium', or 'low' — higher when several independent, "
        "reliable sources agree; low when a single or weak source."
    )


class ResearchReport(BaseModel):
    executive_summary: str = Field(
        description="3-6 sentences answering the objective directly, in plain language."
    )
    findings: List[Finding] = Field(
        description="The key findings, each tied to one or more source IDs. Order by importance."
    )
    gaps: List[str] = Field(
        description="Aspects of the objective the source data did NOT answer. Empty list if fully covered."
    )


def normalise_confidence(value: str) -> str:
    """Coerce a free-text confidence to one of high/medium/low (default low)."""
    v = (value or "").strip().lower()
    return v if v in _CONFIDENCE_VALUES else "low"


def assign_source_ids(sources: dict[str, SourceEntry]) -> list[tuple[str, SourceEntry]]:
    """Give every source a stable citation ID (S1, S2, …), full pages before snippets."""
    entries = list(sources.values())
    ordered = (
        [e for e in entries if e["tier"] != "snippet"]
        + [e for e in entries if e["tier"] == "snippet"]
    )
    return [(f"S{i}", entry) for i, entry in enumerate(ordered, start=1)]


def render_labelled_corpus(
    id_pairs: list[tuple[str, SourceEntry]], max_chars: int
) -> str:
    """Render the ID-tagged source blocks the reporter reads, capped at `max_chars`."""
    parts: list[str] = []
    total = 0
    for sid, entry in id_pairs:
        block = f"[{sid}] SOURCE ({entry['tier']}): {entry['url']}\n{entry['content']}\n\n"
        if parts and total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def validate_report(report: ResearchReport, valid_ids: set[str]) -> list[str]:
    """Return human-readable data-quality warnings — uncited, mis-cited, or thin findings."""
    warnings: list[str] = []
    if not report.findings:
        warnings.append("The report contains no findings.")

    for i, finding in enumerate(report.findings, start=1):
        cited = [s.strip() for s in finding.source_ids if s and s.strip()]
        unknown = [s for s in cited if s not in valid_ids]
        if not cited:
            warnings.append(f'Finding {i} ("{_short(finding.claim)}") cites no source.')
        elif unknown:
            warnings.append(
                f"Finding {i} cites source(s) not in the data: {', '.join(unknown)}."
            )
        elif len(set(cited)) == 1:
            warnings.append(f'Finding {i} ("{_short(finding.claim)}") is single-sourced.')
    return warnings


def report_to_markdown(
    report: ResearchReport,
    id_pairs: list[tuple[str, SourceEntry]],
    *,
    objective: str,
    warnings: list[str],
) -> str:
    """Render a `ResearchReport` to the Markdown shown in the UI and saved to history."""
    url_by_id = {sid: entry["url"] for sid, entry in id_pairs}

    def _cite(ids: list[str]) -> str:
        seen = list(dict.fromkeys(s.strip() for s in ids if s and s.strip()))
        if not seen:
            return "_uncited_"
        return ", ".join(
            f"[{sid}]({url_by_id[sid]})" if sid in url_by_id else f"{sid} (unknown)"
            for sid in seen
        )

    lines: list[str] = ["# Research Report", "", f"**Objective:** {objective.strip()}", ""]

    if warnings:
        lines += [
            "> ⚠️ **Data-quality flags**",
            *(f"> - {w}" for w in warnings),
            "",
        ]

    lines += ["## Executive Summary", "", report.executive_summary.strip(), "", "## Key Findings", ""]
    if report.findings:
        for i, finding in enumerate(report.findings, start=1):
            lines += [
                f"**{i}. {finding.claim.strip()}**  ",
                f"{finding.detail.strip()}  ",
                f"*Sources: {_cite(finding.source_ids)} · "
                f"Confidence: {normalise_confidence(finding.confidence).title()}*",
                "",
            ]
    else:
        lines += ["_No findings were produced._", ""]

    lines += ["## Gaps & Open Questions", ""]
    if report.gaps:
        lines += [f"- {g.strip()}" for g in report.gaps] + [""]
    else:
        lines += ["_None identified — the sources covered the objective._", ""]

    lines += ["## Sources", "", "| ID | Source | Access |", "|----|--------|--------|"]
    for sid, entry in id_pairs:
        label = _TIER_LABEL.get(entry["tier"], entry["tier"])
        lines.append(f"| {sid} | {entry['url']} | {label} |")
    lines.append("")

    return "\n".join(lines).strip()
