"""Tests for corpus.py - the helpers that collect scraped text into one "corpus".

How to read these tests:
  - Every function that starts with `test_` is one test. pytest finds them automatically.
  - Each test has three parts: Arrange (set up data), Act (call the function),
    Assert (check the result). `assert x == y` fails the test if it is not true.

Run with:  pytest tests/test_corpus.py
"""
from corpus import has_full_page, merge_scraped, render_corpus, seed_snippet, wrap_untrusted

# The scraper only accepts a page if it has at least this many characters.
MIN_USEFUL = 300

# A block of text long enough to count as a "real" page.
LONG_TEXT = "real page content. " * 30   # ~570 characters


# --- seed_snippet ------------------------------------------------------------

def test_seed_snippet_adds_a_new_source():
    # Arrange
    sources = {}

    # Act
    seed_snippet(sources, "https://a.com", "a short snippet")

    # Assert
    assert sources["https://a.com"]["tier"] == "snippet"
    assert sources["https://a.com"]["content"] == "a short snippet"


def test_seed_snippet_ignores_empty_text():
    sources = {}

    seed_snippet(sources, "https://a.com", "")

    assert sources == {}


def test_seed_snippet_does_not_replace_a_full_page():
    # Arrange: we already have a full page for this URL
    sources = {"https://a.com": {"url": "https://a.com", "tier": "browser", "content": LONG_TEXT}}

    # Act: try to add a snippet for the same URL
    seed_snippet(sources, "https://a.com", "stale snippet")

    # Assert: the full page is still there, untouched
    assert sources["https://a.com"]["tier"] == "browser"
    assert sources["https://a.com"]["content"] == LONG_TEXT


# --- has_full_page -----------------------------------------------------------

def test_has_full_page():
    sources = {
        "https://snip.com": {"url": "https://snip.com", "tier": "snippet", "content": "x"},
        "https://full.com": {"url": "https://full.com", "tier": "api", "content": "y"},
    }

    assert has_full_page(sources, "https://full.com") is True
    assert has_full_page(sources, "https://snip.com") is False
    assert has_full_page(sources, "https://missing.com") is False


# --- merge_scraped -----------------------------------------------------------

def test_merge_scraped_upgrades_snippet_to_full_page():
    # Arrange: start with only a snippet
    sources = {"https://a.com": {"url": "https://a.com", "tier": "snippet", "content": "snip"}}

    # Act: merge in a long page
    applied = merge_scraped(sources, "https://a.com", LONG_TEXT, "browser", MIN_USEFUL)

    # Assert: the snippet was replaced by the page
    assert applied is True
    assert sources["https://a.com"]["tier"] == "browser"
    assert sources["https://a.com"]["content"] == LONG_TEXT


def test_merge_scraped_rejects_short_page_and_keeps_snippet():
    sources = {"https://a.com": {"url": "https://a.com", "tier": "snippet", "content": "snip"}}

    applied = merge_scraped(sources, "https://a.com", "bot wall", "browser", MIN_USEFUL)

    assert applied is False
    assert sources["https://a.com"]["tier"] == "snippet"


# --- wrap_untrusted ----------------------------------------------------------

def test_wrap_untrusted_puts_fences_around_text():
    assert wrap_untrusted("hello") == "<SOURCE_DATA>\nhello\n</SOURCE_DATA>"


def test_wrap_untrusted_turns_blank_text_into_none():
    assert wrap_untrusted("   ") == "<SOURCE_DATA>\nNone\n</SOURCE_DATA>"


# --- render_corpus -----------------------------------------------------------

def test_render_corpus_puts_full_pages_before_snippets():
    sources = {
        "https://snip.com": {"url": "https://snip.com", "tier": "snippet", "content": "S"},
        "https://full.com": {"url": "https://full.com", "tier": "browser", "content": "F"},
    }

    rendered = render_corpus(sources, max_chars=10_000)

    # .index() gives the position of the text - a smaller number means it comes first
    assert rendered.index("https://full.com") < rendered.index("https://snip.com")
    assert "-- SOURCE: https://full.com --" in rendered
    assert "-- SOURCE (snippet): https://snip.com --" in rendered


def test_render_corpus_stops_at_the_character_limit():
    # Arrange: 10 sources of 500 characters each = 5,000 characters total
    sources = {}
    for i in range(10):
        url = f"https://s{i}.com"
        sources[url] = {"url": url, "tier": "browser", "content": "x" * 500}

    # Act: only allow 1,200 characters
    rendered = render_corpus(sources, max_chars=1_200)

    # Assert: not every source fit into the text...
    assert rendered.count("-- SOURCE:") < 10
    # ...but the original dict was not changed
    assert len(sources) == 10


def test_render_corpus_always_includes_at_least_one_source():
    sources = {"https://big.com": {"url": "https://big.com", "tier": "browser", "content": "x" * 99_999}}

    rendered = render_corpus(sources, max_chars=100)

    assert "https://big.com" in rendered
