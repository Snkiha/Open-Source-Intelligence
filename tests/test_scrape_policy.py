"""Tests for scrape_policy.py - deciding how to fetch a URL and spotting block pages.

  - api_first_reason(url) returns "pdf", "youtube", or "" (empty = use the browser).
  - is_block_page(status, title, body) returns True when a site refused us.

Run with:  pytest tests/test_scrape_policy.py
"""
from scrape_policy import api_first_reason, is_block_page


# --- api_first_reason --------------------------------------------------------

def test_pdf_urls_go_straight_to_api():
    assert api_first_reason("https://intra.ece.ucr.edu/~oymak/multiclass.pdf") == "pdf"
    assert api_first_reason("https://example.com/paper.PDF") == "pdf"
    assert api_first_reason("https://example.com/docs/report.pdf?download=1") == "pdf"


def test_youtube_urls_go_straight_to_api():
    assert api_first_reason("https://www.youtube.com/watch?v=Md4b67HvmRo") == "youtube"
    assert api_first_reason("https://youtube.com/watch?v=abc") == "youtube"
    assert api_first_reason("https://m.youtube.com/watch?v=abc") == "youtube"
    assert api_first_reason("https://youtu.be/abc") == "youtube"


def test_ordinary_urls_use_the_browser():
    assert api_first_reason("https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy/") == ""
    assert api_first_reason("https://sassafras13.github.io/BiCE/") == ""


def test_look_alike_urls_still_use_the_browser():
    # "pdf" appears in the path but the file is not a .pdf
    assert api_first_reason("https://example.com/pdf-guide/") == ""
    # the host only *contains* the word youtube
    assert api_first_reason("https://notyoutube.com/watch?v=abc") == ""
    assert api_first_reason("https://example.com/youtube-strategy") == ""


def test_broken_url_uses_the_browser():
    assert api_first_reason("not a url") == ""


# --- is_block_page -----------------------------------------------------------

def test_cloudflare_403_page_is_blocked():
    body = "Sorry, you have been blocked. You are unable to access medium.com. " * 3

    assert is_block_page(403, "Attention Required! | Cloudflare", body) is True


def test_cloudflare_challenge_page_is_blocked():
    assert is_block_page(403, "Just a moment...", "Performing security verification") is True


def test_reddit_network_block_is_blocked():
    body = "You've been blocked by network security. If you think ..."

    assert is_block_page(200, "", body) is True


def test_http_error_status_is_blocked_even_with_normal_text():
    body = "Some long neutral text " * 50

    assert is_block_page(503, "Service", body) is True


def test_real_article_is_not_blocked():
    body = "Binary cross-entropy is the loss function used for binary classification. " * 20

    assert is_block_page(200, "Binary Cross Entropy | GeeksforGeeks", body) is False


def test_article_that_talks_about_blocking_is_not_blocked():
    # The "blocked" phrase is only checked in the title and the START of the body,
    # so an article that mentions blocking near the end is still a normal page.
    body = "How CDNs work. " * 40 + " Cloudflare may say 'you have been blocked' to bots."

    assert is_block_page(200, "How CDNs work", body) is False


def test_missing_status_code_alone_is_not_blocked():
    assert is_block_page(None, "Home", "Ordinary text " * 40) is False
