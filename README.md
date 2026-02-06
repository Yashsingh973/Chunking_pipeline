# Tree-Based Legal RAG Pipeline

This repository implements a deterministic, hierarchical retrieval pipeline for legal documents. It preserves Markdown heading structure, builds a tree of H1/H2 nodes, and enables scoped hybrid retrieval (BM25 routing + embeddings). The design avoids flat chunking and enforces legal section boundaries for citation-safe answers.

## Architecture Overview

**Ingestion**
- Markdown (from OCR + layout models) is parsed into a tree.
- Each PDF becomes a document root.
- H1 nodes become routing heads with summaries.
- H2 nodes hold statutory language and are embedded.

**Indexes**
- **Index A (BM25)**: H1 head + summary + pdf_name for high-recall routing.
- **Index B (Vector)**: H2 text embeddings with strict H1 scoping.
- **Index C (Lookup Table)**: Parent/child navigation and citation grouping.

**Retrieval Flow**
1. BM25 search on H1 summaries.
2. Scoped vector search within selected H1 nodes.
3. Optional tree expansion for adjacent clauses.
4. Citation formatting with page + PDF metadata.

## Usage

```python
from rag_pipeline import (
    InferencePipeline,
    BM25Index,
    IngestionConfig,
    RetrievalAgent,
    VectorIndex,
    parse_markdown_to_tree,
)

markdown_text = \"\"\"
# Cantonment Code and Reforms
[[PAGE 2]]
## Powers of the Cantonment Board
[[PAGE 3]]
The Cantonment Board shall have authority to levy taxes...
\"\"\"

tree = parse_markdown_to_tree(
    markdown_text,
    IngestionConfig(doc_id="clar_1925", pdf_name="CLAR 1925.pdf"),
)

bm25 = BM25Index()
vector = VectorIndex()
for h1 in tree.h1_nodes.values():
    bm25.add(h1)
for h2 in tree.h2_nodes.values():
    vector.add(h2)

agent = RetrievalAgent(tree=tree, bm25=bm25, vector=vector)
result = agent.retrieve("What powers does the Cantonment Board have?")

for chunk in result.chunks:
    print(chunk.text, chunk.pages)
print(result.citations)
```

## MongoDB Storage

Persist the tree structure and lookup tables in MongoDB to keep the hierarchy auditable and queryable.

```python
from rag_pipeline import DocumentRoot, MongoTreeStore

store = MongoTreeStore(uri="mongodb://localhost:27017", db_name="legal_rag")
store.save_tree(DocumentRoot(pdf_name="CLAR 1925.pdf", doc_id="clar_1925"), tree)

reloaded_tree = store.load_tree(doc_id="clar_1925")
```

## Inference Pipeline

Use the inference pipeline to retrieve scoped context and return a citation-safe answer payload.

```python
pipeline = InferencePipeline(tree=reloaded_tree, bm25=bm25, vector=vector)
payload = pipeline.answer("What powers does the Cantonment Board have?")
print(payload.answer)
```

## Local Testing Steps (1-2 Markdown Files)

1. **Start MongoDB** (local install or container).
   ```bash
   mongod --dbpath /tmp/mongo-legal
   ```
2. **Install Python dependencies** (example uses PyMongo).
   ```bash
   pip install pymongo
   ```
3. **Prepare Markdown files** (`doc1.md`, `doc2.md`) with H1/H2 structure and `[[PAGE N]]` markers.
4. **Run ingestion + storage + retrieval**:
   ```python
   from pathlib import Path
   from rag_pipeline import (
       BM25Index,
       DocumentRoot,
       InferencePipeline,
       IngestionConfig,
       MongoTreeStore,
       VectorIndex,
       parse_markdown_to_tree,
   )

   store = MongoTreeStore(uri="mongodb://localhost:27017", db_name="legal_rag")

   bm25 = BM25Index()
   vector = VectorIndex()

   for idx, path in enumerate([Path("doc1.md"), Path("doc2.md")], start=1):
       markdown_text = path.read_text()
       doc_id = f"doc_{idx:02d}"
       pdf_name = f"{path.stem}.pdf"
       tree = parse_markdown_to_tree(markdown_text, IngestionConfig(doc_id=doc_id, pdf_name=pdf_name))
       store.save_tree(DocumentRoot(pdf_name=pdf_name, doc_id=doc_id), tree)
       for h1 in tree.h1_nodes.values():
           bm25.add(h1)
       for h2 in tree.h2_nodes.values():
           vector.add(h2)

   combined_tree = store.load_tree(doc_id="doc_01")
   pipeline = InferencePipeline(tree=combined_tree, bm25=bm25, vector=vector)
   payload = pipeline.answer("Ask a legal question here")
   print(payload.answer)
   ```
