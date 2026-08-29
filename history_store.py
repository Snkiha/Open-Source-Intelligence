"""Persistent store for completed research runs.

Each finished run is written as a single JSON file under ``history/`` so past
reports can be searched and re-read without running the agent again. Storage is
local to the machine; on ephemeral hosts (e.g. Streamlit Community Cloud) the
history resets whenever the app redeploys.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("osint_agent.history")

HISTORY_DIR = Path("history")

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_SLUG_LEN = 50
_SCHEMA_VERSION = 2  # v2: added report_struct + warnings


@dataclass(frozen=True)
class ResearchRecord:
    """One completed research run, as persisted to disk."""

    id: str
    objective: str
    model: str
    report: str
    created_at: str  # ISO 8601, UTC
    sources: tuple[str, ...] = ()
    queries_run: int = 0
    chars_collected: int = 0
    api_scraped_count: int = 0
    report_struct: dict | None = None  # serialised ResearchReport, if structured
    warnings: tuple[str, ...] = ()  # data-quality flags recorded at report time

    @property
    def created_display(self) -> str:
        """Local-time ``YYYY-MM-DD HH:MM`` for the UI, falling back to the raw value."""
        try:
            return (
                datetime.fromisoformat(self.created_at)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
        except ValueError:
            return self.created_at


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:_MAX_SLUG_LEN].strip("-") or "research"


def build_record(
    objective: str,
    model: str,
    report: str,
    *,
    sources: Iterable[str] = (),
    queries_run: int = 0,
    chars_collected: int = 0,
    api_scraped_count: int = 0,
    report_struct: dict | None = None,
    warnings: Iterable[str] = (),
) -> ResearchRecord:
    """Assemble a record for a run that just finished, stamped with the current time."""
    return ResearchRecord(
        id=uuid.uuid4().hex,
        objective=objective.strip(),
        model=model,
        report=report,
        created_at=datetime.now(timezone.utc).isoformat(),
        sources=tuple(dict.fromkeys(s for s in sources if s)),
        queries_run=queries_run,
        chars_collected=chars_collected,
        api_scraped_count=api_scraped_count,
        report_struct=report_struct or None,
        warnings=tuple(warnings),
    )


def _to_dict(record: ResearchRecord) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "id": record.id,
        "objective": record.objective,
        "model": record.model,
        "report": record.report,
        "created_at": record.created_at,
        "sources": list(record.sources),
        "queries_run": record.queries_run,
        "chars_collected": record.chars_collected,
        "api_scraped_count": record.api_scraped_count,
        "report_struct": record.report_struct,
        "warnings": list(record.warnings),
    }


def _from_dict(data: dict[str, Any]) -> ResearchRecord:
    report_struct = data.get("report_struct")
    return ResearchRecord(
        id=str(data.get("id") or uuid.uuid4().hex),
        objective=str(data.get("objective", "")),
        model=str(data.get("model", "")),
        report=str(data.get("report", "")),
        created_at=str(data.get("created_at", "")),
        sources=tuple(data.get("sources") or ()),
        queries_run=int(data.get("queries_run", 0) or 0),
        chars_collected=int(data.get("chars_collected", 0) or 0),
        api_scraped_count=int(data.get("api_scraped_count", 0) or 0),
        report_struct=report_struct if isinstance(report_struct, dict) else None,
        warnings=tuple(data.get("warnings") or ()),
    )


def save_record(record: ResearchRecord, directory: Path = HISTORY_DIR) -> Path:
    """Write ``record`` to ``directory`` as ``<timestamp>-<slug>-<id8>.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = directory / f"{stamp}-{_slugify(record.objective)}-{record.id[:8]}.json"
    path.write_text(
        json.dumps(_to_dict(record), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved research record %s -> %s", record.id, path.name)
    return path


def load_records(directory: Path = HISTORY_DIR) -> list[ResearchRecord]:
    """Load every saved record, newest first. Unreadable files are skipped."""
    if not directory.exists():
        return []
    records: list[ResearchRecord] = []
    for path in directory.glob("*.json"):
        try:
            records.append(_from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable history file %s: %s", path.name, exc)
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def search_records(
    records: Iterable[ResearchRecord], query: str
) -> list[ResearchRecord]:
    """Filter to records where every whitespace-separated term appears (case-insensitive)
    in the objective, report body, or a source URL. An empty query returns all."""
    terms = query.lower().split()
    result: list[ResearchRecord] = []
    for record in records:
        haystack = "\n".join(
            (record.objective, record.report, *record.sources)
        ).lower()
        if all(term in haystack for term in terms):
            result.append(record)
    return result
