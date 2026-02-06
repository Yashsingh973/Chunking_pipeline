from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rag_pipeline.indexes import BM25Index, VectorIndex
from rag_pipeline.schemas import H1Node, H2Node, TreeIndex
from rag_pipeline.utils import dedupe_preserve_order


@dataclass(frozen=True)
class RetrievedChunk:
    node_id: str
    text: str
    pdf_name: str
    pages: List[int]
    parent_h1: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    h1_candidates: List[str]
    chunks: List[RetrievedChunk]
    citations: List[str]


class RetrievalAgent:
    def __init__(self, tree: TreeIndex, bm25: BM25Index, vector: VectorIndex) -> None:
        self.tree = tree
        self.bm25 = bm25
        self.vector = vector

    def bm25_search(self, query: str, top_k: int = 3) -> List[str]:
        results = self.bm25.search(query, top_k=top_k)
        return [node_id for node_id, _ in results]

    def vector_search(self, query: str, h1_scope: List[str], top_k: int = 5) -> List[Tuple[H2Node, float]]:
        scoped_h2_ids = []
        for h1_id in h1_scope:
            h1_node = self.tree.h1_nodes.get(h1_id)
            if not h1_node:
                continue
            scoped_h2_ids.extend(h1_node.children)
        scoped_h2_ids = dedupe_preserve_order(scoped_h2_ids)
        return self.vector.search(query, scoped_h2_ids, top_k=top_k)

    def tree_expand(self, node_id: str, window: int = 1) -> List[str]:
        h2_ids = []
        for h1_id, h1_node in self.tree.h1_nodes.items():
            if node_id not in h1_node.children:
                continue
            children = h1_node.children
            idx = children.index(node_id)
            start = max(0, idx - window)
            end = min(len(children), idx + window + 1)
            h2_ids.extend(children[start:end])
        return dedupe_preserve_order(h2_ids)

    def format_citations(self, chunks: List[RetrievedChunk]) -> List[str]:
        citations = []
        for chunk in chunks:
            pages = ", ".join(str(page) for page in chunk.pages) if chunk.pages else "n/a"
            citations.append(f"{chunk.pdf_name} p.{pages} [{chunk.node_id}]")
        return dedupe_preserve_order(citations)

    def retrieve(self, query: str, top_h1: int = 3, top_h2: int = 4) -> RetrievalResult:
        h1_candidates = self.bm25_search(query, top_k=top_h1)

        needs_detail = len(query.split()) > 3
        chunks: List[RetrievedChunk] = []
        if needs_detail:
            h2_results = self.vector_search(query, h1_candidates, top_k=top_h2)
            for node, score in h2_results:
                chunks.append(
                    RetrievedChunk(
                        node_id=node.node_id,
                        text=node.text,
                        pdf_name=node.pdf_name,
                        pages=node.pages,
                        parent_h1=node.parent,
                        score=score,
                    )
                )
        else:
            for h1_id in h1_candidates:
                h1_node = self.tree.h1_nodes.get(h1_id)
                if not h1_node:
                    continue
                chunks.append(
                    RetrievedChunk(
                        node_id=h1_node.node_id,
                        text=f"{h1_node.head}\n{h1_node.summary}",
                        pdf_name=h1_node.pdf_name,
                        pages=h1_node.pages,
                        parent_h1=h1_node.node_id,
                        score=0.0,
                    )
                )

        citations = self.format_citations(chunks)
        return RetrievalResult(
            query=query,
            h1_candidates=h1_candidates,
            chunks=chunks,
            citations=citations,
        )

