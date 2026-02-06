from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from rag_pipeline.schemas import H1Node, H2Node
from rag_pipeline.utils import cosine_similarity, hash_embedding, tokenize


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    documents: Dict[str, str] = field(default_factory=dict)
    doc_freqs: Dict[str, int] = field(default_factory=dict)
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    avg_doc_len: float = 0.0

    def add(self, node: H1Node) -> None:
        text = f"{node.head}\n{node.summary}\n{node.pdf_name}"
        self.documents[node.node_id] = text
        tokens = tokenize(text)
        self.doc_lengths[node.node_id] = len(tokens)
        for token in set(tokens):
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = []
        query_tokens = tokenize(query)
        total_docs = len(self.documents)
        for doc_id, text in self.documents.items():
            doc_tokens = tokenize(text)
            doc_len = self.doc_lengths.get(doc_id, 1)
            token_counts = {token: doc_tokens.count(token) for token in set(doc_tokens)}
            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue
                df = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = token_counts[token]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1))
                score += idf * (tf * (self.k1 + 1) / denom)
            scores.append((doc_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


@dataclass
class VectorIndex:
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, H2Node] = field(default_factory=dict)
    dim: int = 256

    def add(self, node: H2Node) -> None:
        embedding = node.embedding or hash_embedding(node.text, dim=self.dim)
        self.embeddings[node.node_id] = embedding
        self.metadata[node.node_id] = H2Node(
            node_id=node.node_id,
            parent=node.parent,
            level=node.level,
            head=node.head,
            text=node.text,
            pages=node.pages,
            pdf_name=node.pdf_name,
            embedding=embedding,
        )

    def search(self, query: str, scope: List[str], top_k: int = 5) -> List[Tuple[H2Node, float]]:
        query_embedding = hash_embedding(query, dim=self.dim)
        results: List[Tuple[H2Node, float]] = []
        for node_id in scope:
            embedding = self.embeddings.get(node_id)
            if embedding is None:
                continue
            score = cosine_similarity(query_embedding, embedding)
            results.append((self.metadata[node_id], score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

