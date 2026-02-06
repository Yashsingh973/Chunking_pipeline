# Test Report

## Scope

This report covers local unit tests that validate:
- H1/H2 chunking behavior (including H2 fallback when missing).
- Fixture-based ingestion using dummy Markdown files.
- Environment-driven LLM summarizer loading behavior when `LLM_API_URL` is unset.

## Dummy Data Used

Fixtures created in `tests/fixtures/`:
- `doc1.md`: H1 with two H2 sections and `[[PAGE 1]]`.
- `doc2.md`: H1 with no H2, to exercise fallback chunk creation.

## Test Execution

Commands run:

```bash
python -m unittest
python -m unittest discover -s tests
```

## Results

`python -m unittest` runs without import errors after making optional imports in the LLM and MongoDB helpers. `python -m unittest discover -s tests` discovers and executes the ingestion tests successfully.

## What Each Test Validates

- **test_h2_fallback_created_when_missing**: Ensures that an H1 without H2 headers still produces one H2 chunk containing the H1 text.
- **test_h2_chunks_created_for_multiple_sections**: Ensures multiple H2 sections within a single H1 yield distinct H2 chunks.
- **test_load_summarizer_from_env_none_without_url**: Ensures the environment loader returns `None` when `LLM_API_URL` is not configured.
- **test_fixture_markdown_ingestion**: Ensures the fixtures produce 2 H2 chunks for `doc1.md` and a fallback H2 chunk for `doc2.md`.
