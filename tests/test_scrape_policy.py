"""Unit tests for scrape routing policy: block-page detection and API-first URLs."""
import pytest

from scrape_policy import api_first_reason, is_block_page


# -- api_first_reason --------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://intra.ece.ucr.edu/~oymak/multiclass.pdf",
    "https://example.com/paper.PDF",
    "https://example.com/docs/report.pdf?download=1",
])
def test_pdf_urls_go_straight_to_api(url):
    assert api_first_reason(url) == "pdf"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=Md4b67HvmRo",
    "https://youtube.com/watch?v=abc",
    "https://m.youtube.com/watch?v=abc",
    "https://youtu.be/abc",
])
def test_youtube_urls_go_straight_to_api(url):
    assert api_first_reason(url) == "youtube"


@pytest.mark.parametrize("url", [
    "https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy/",
    "https://sassafras13.github.io/BiCE/",
    "https://example.com/pdf-guide/",          # "pdf" in path but not the suffix
    "https://notyoutube.com/watch?v=abc",       # host merely contains the word
    "https://example.com/youtube-strategy",
])
def test_ordinary_urls_use_browser(url):
    assert api_first_reason(url) == ""


def test_malformed_url_uses_browser():
    assert api_first_reason("not a url") == ""


# -- is_block_page -----------------------------------------------------------

def test_cloudflare_403_medium_is_blocked():
    body = ("Please enable cookies. Sorry, you have been blocked You are unable to access "
            "medium.com Why have I been blocked? This website is using a security service " * 3)
    assert is_block_page(403, "Attention Required! | Cloudflare", body)


def test_cloudflare_challenge_is_blocked():
    assert is_block_page(403, "Just a moment...", "www.datacamp.com Performing security verification")


def test_reddit_network_block_is_blocked():
    assert is_block_page(200, "", "You've been blocked by network security. If you think ...")


def test_http_error_status_is_blocked_even_with_neutral_text():
    assert is_block_page(503, "Service", "Some long neutral text " * 50)


def test_real_article_is_not_blocked():
    body = "Binary cross-entropy is the loss function used for binary classification. " * 20
    assert not is_block_page(200, "Binary Cross Entropy/Log Loss | GeeksforGeeks", body)


def test_article_mentioning_blocking_is_not_blocked():
    # Marker phrases must be checked in the title or the *start* of the body only,
    # so an article that discusses bot blocking in passing is not misclassified.
    body = "How CDNs work. " * 40 + " Cloudflare may say 'you have been blocked' to bots."
    assert not is_block_page(200, "How CDNs work", body)


def test_missing_status_is_not_blocked_on_its_own():
    assert not is_block_page(None, "Home", "Ordinary text " * 40)
