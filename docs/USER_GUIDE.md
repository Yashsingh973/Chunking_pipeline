# User Guide: Tree-Based Legal RAG Pipeline

This guide explains how to ingest Markdown (from legal PDFs), persist the tree in MongoDB, and run scoped retrieval with citations.

## 1) Prepare Markdown Inputs

Your Markdown should preserve headings and page markers:

```markdown
# Act Title
[[PAGE 1]]
## Section 1
Statutory text...
```

**Rules**
- `#` = H1 (routing section).
- `##` = H2 (retrieval chunks).
- `[[PAGE N]]` markers are used for citation metadata.

## 2) Optional LLM Summaries (.env)

To generate H1 summaries using an LLM, create a `.env` file:

```bash
LLM_API_URL=https://your-llm-endpoint.example.com/v1/completions
LLM_API_KEY=your_api_key
LLM_MODEL=your_model_name
```

If `LLM_API_URL` is missing, the pipeline falls back to a deterministic local summary.

## 3) Ingest Markdown and Build Indexes

```python
from rag_pipeline import (
    BM25Index,
    IngestionConfig,
    VectorIndex,
    load_summarizer_from_env,
    parse_markdown_to_tree,
)

markdown_text = open("doc1.md", "r", encoding="utf-8").read()
summarizer = load_summarizer_from_env()

tree = parse_markdown_to_tree(
    markdown_text,
    IngestionConfig(
        doc_id="doc_01",
        pdf_name="doc1.pdf",
        summarize_fn=summarizer.summarize if summarizer else None,
    ),
)

bm25 = BM25Index()
vector = VectorIndex()
for h1 in tree.h1_nodes.values():
    bm25.add(h1)
for h2 in tree.h2_nodes.values():
    vector.add(h2)
```

## 4) Persist Tree to MongoDB

```python
from rag_pipeline import DocumentRoot, MongoTreeStore

store = MongoTreeStore(uri="mongodb://localhost:27017", db_name="legal_rag")
store.save_tree(DocumentRoot(pdf_name="doc1.pdf", doc_id="doc_01"), tree)
```

## 5) Run Scoped Retrieval + Inference

```python
from rag_pipeline import InferencePipeline

pipeline = InferencePipeline(tree=tree, bm25=bm25, vector=vector)
payload = pipeline.answer("What powers does the board have?")
print(payload.answer)
```

## 6) Local Test Command

```bash
python -m unittest
```
