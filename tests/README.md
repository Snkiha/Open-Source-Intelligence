# Tests

Plain `pytest` unit tests. No fixtures, no parametrize, no mocks - just functions
with `assert` statements.

## Run them

From the project root:

```bash
pytest tests/
```

Run one file:

```bash
pytest tests/test_corpus.py
```

Run one test by name (`-k` matches part of the name):

```bash
pytest tests/ -k "snippet"
```

Show print output and stop at the first failure:

```bash
pytest tests/ -s -x
```

## How a test works

```python
def test_seed_snippet_adds_a_new_source():
    # Arrange - set up the data
    sources = {}

    # Act - call the function you are testing
    seed_snippet(sources, "https://a.com", "a short snippet")

    # Assert - check what happened
    assert sources["https://a.com"]["tier"] == "snippet"
```

- Any function whose name starts with `test_` is a test. pytest finds them by itself.
- `assert` is the only tool you need. If the condition is false, the test fails and
  pytest prints both sides of the comparison.
- The test name should say what is being checked, so a failure reads like a sentence.

## The one pytest feature used

`tmp_path` in `test_history_store.py` is a built-in pytest argument. When a test
asks for it, pytest hands over a fresh empty folder, so the test can write files
without touching the real `history/` folder.

## Files

| File | What it tests |
|------|---------------|
| `test_corpus.py` | Collecting scraped text into one corpus |
| `test_history_store.py` | Saving, loading and searching past runs |
| `test_report_schema.py` | Source IDs, citation checks, Markdown output |
| `test_scrape_policy.py` | PDF / YouTube routing and block-page detection |
