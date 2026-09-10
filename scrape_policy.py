"""Routing policy for the scraping tiers.

Pure functions, no I/O — decides (a) which URLs skip the headless browser and go
straight to the reader API, and (b) whether a page the browser *did* load is
really a bot wall rather than content.
"""
from __future__ import annotations

import urllib.parse

# Chromium cannot render these inline: `page.goto` raises "Download is starting".
_API_FIRST_SUFFIXES = (".pdf",)

# The browser tier "succeeds" on these but returns only UI chrome (~1k chars of
# nav and player text). The reader API extracts the description/transcript.
_API_FIRST_HOSTS = ("youtube.com", "youtu.be")

# Titles used by Cloudflare / Akamai / DataDome / PerimeterX challenge pages.
_BLOCK_TITLE_MARKERS = (
    "just a moment",
    "attention required",
    "access denied",
    "security verification",
    "verify you are human",
    "are you a robot",
    "bot verification",
)

# Phrases that appear at the top of a challenge page's body text.
_BLOCK_BODY_MARKERS = (
    "you have been blocked",
    "you've been blocked",
    "blocked by network security",
    "enable javascript and cookies to continue",
    "checking your browser",
    "checking if the site connection is secure",
    "performing security verification",
    "verify you are human",
)

# Only the opening of the body is inspected so a long article that merely
# *mentions* bot blocking is not misclassified.
_BODY_INSPECT_CHARS = 400


class BlockedPageError(Exception):
    """The browser loaded a bot wall / error page instead of content."""


def api_first_reason(url: str) -> str:
    """Return why `url` should bypass the browser tier ("pdf", "youtube"), or ""."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return ""
    host = parsed.netloc.lower().rsplit("@", 1)[-1].split(":", 1)[0]
    path = parsed.path.lower()

    if path.endswith(_API_FIRST_SUFFIXES):
        return "pdf"
    for known in _API_FIRST_HOSTS:
        if host == known or host.endswith("." + known):
            return "youtube"
    return ""


def is_block_page(status: int | None, title: str, text: str) -> bool:
    """True when the loaded page is an HTTP error or a recognised bot-challenge page."""
    if status is not None and status >= 400:
        return True
    title_l = (title or "").lower()
    if any(marker in title_l for marker in _BLOCK_TITLE_MARKERS):
        return True
    head = (text or "")[:_BODY_INSPECT_CHARS].lower()
    return any(marker in head for marker in _BLOCK_BODY_MARKERS)
