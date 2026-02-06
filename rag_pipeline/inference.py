from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from rag_pipeline.indexes import BM25Index, VectorIndex
from rag_pipeline.retrieval import RetrievalAgent, RetrievedChunk
from rag_pipeline.schemas import TreeIndex


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    citations: List[str]
    chunks: List[RetrievedChunk]


class InferencePipeline:
    def __init__(self, tree: TreeIndex, bm25: BM25Index, vector: VectorIndex) -> None:
        self.agent = RetrievalAgent(tree=tree, bm25=bm25, vector=vector)

    def answer(self, query: str) -> AnswerPayload:
        result = self.agent.retrieve(query)
        answer_lines = [
            "Retrieved statutory context (no additional interpretation applied):"
        ]
        for chunk in result.chunks:
            answer_lines.append(f"- {chunk.text.strip()}")
        answer_lines.append("")
        answer_lines.append("Citations:")
        for citation in result.citations:
            answer_lines.append(f"- {citation}")
        answer = "\n".join(answer_lines).strip()
        return AnswerPayload(answer=answer, citations=result.citations, chunks=result.chunks)

    def debug_state(self) -> Dict[str, str]:
        return {"policy": "BM25 routing -> scoped vector search -> citation-safe answer"}

