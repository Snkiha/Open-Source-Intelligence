"""Unit tests for the corpus assembly helpers (pure, no Streamlit/Playwright)."""
from corpus import (
    has_full_page,
    merge_scraped,
    render_corpus,
    seed_snippet,
    wrap_untrusted,
)

MIN_USEFUL = 300


def _long(text: str, n: int = MIN_USEFUL) -> str:
    return (text + " ") * (n // len(text) + 1)


def test_seed_snippet_adds_entry_when_absent():
    sources = {}
    seed_snippet(sources, "https://a.com", "a short snippet")
    assert sources["https://a.com"] == {
        "url": "https://a.com", "tier": "snippet", "content": "a short snippet",
    }


def test_seed_snippet_ignores_empty_snippet():
    sources = {}
    seed_snippet(sources, "https://a.com", "")
    assert sources == {}


def test_seed_snippet_does_not_downgrade_a_full_page():
    page = _long("real content")
    sources = {"https://a.com": {"url": "https://a.com", "tier": "browser", "content": page}}
    seed_snippet(sources, "https://a.com", "stale snippet")
    assert sources["https://a.com"]["tier"] == "browser"
    assert sources["https://a.com"]["content"] == page


def test_has_full_page():
    sources = {
        "https://snip.com": {"url": "https://snip.com", "tier": "snippet", "content": "x"},
        "https://full.com": {"url": "https://full.com", "tier": "api", "content": "y"},
    }
    assert has_full_page(sources, "https://full.com") is True
    assert has_full_page(sources, "https://snip.com") is False
    assert has_full_page(sources, "https://missing.com") is False


def test_merge_scraped_upgrades_snippet_to_full_page():
    sources = {"https://a.com": {"url": "https://a.com", "tier": "snippet", "content": "snip"}}
    page = _long("full page body")
    applied = merge_scraped(sources, "https://a.com", page, "browser", MIN_USEFUL)
    assert applied is True
    assert sources["https://a.com"]["tier"] == "browser"
    assert sources["https://a.com"]["content"] == page


def test_merge_scraped_rejects_short_body_and_keeps_snippet():
    sources = {"https://a.com": {"url": "https://a.com", "tier": "snippet", "content": "snip"}}
    applied = merge_scraped(sources, "https://a.com", "bot wall", "browser", MIN_USEFUL)
    assert applied is False
    assert sources["https://a.com"]["tier"] == "snippet"


def test_wrap_untrusted_fences_content():
    assert wrap_untrusted("hello") == "<SOURCE_DATA>\nhello\n</SOURCE_DATA>"


def test_wrap_untrusted_handles_empty():
    assert wrap_untrusted("   ") == "<SOURCE_DATA>\nNone\n</SOURCE_DATA>"


def test_render_corpus_orders_full_pages_before_snippets():
    sources = {
        "https://snip.com": {"url": "https://snip.com", "tier": "snippet", "content": "S"},
        "https://full.com": {"url": "https://full.com", "tier": "browser", "content": "F"},
    }
    rendered = render_corpus(sources, max_chars=10_000)
    assert rendered.index("https://full.com") < rendered.index("https://snip.com")
    assert "-- SOURCE: https://full.com --" in rendered
    assert "-- SOURCE (snippet): https://snip.com --" in rendered


def test_render_corpus_cap_truncates_view_only():
    sources = {
        f"https://s{i}.com": {"url": f"https://s{i}.com", "tier": "browser", "content": "x" * 500}
        for i in range(10)
    }
    rendered = render_corpus(sources, max_chars=1_200)
    # cap hit: not every source made it into the prompt text...
    assert rendered.count("-- SOURCE:") < 10
    # ...but the caller's sources dict is untouched
    assert len(sources) == 10


def test_render_corpus_always_emits_at_least_one_source():
    sources = {"https://big.com": {"url": "https://big.com", "tier": "browser", "content": "x" * 99_999}}
    rendered = render_corpus(sources, max_chars=100)
    assert "https://big.com" in rendered
