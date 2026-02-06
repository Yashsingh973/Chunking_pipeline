from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DocumentRoot:
    pdf_name: str
    doc_id: str


@dataclass(frozen=True)
class H1Node:
    node_id: str
    level: str
    head: str
    summary: str
    pages: List[int]
    pdf_name: str
    children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class H2Node:
    node_id: str
    parent: str
    level: str
    head: str
    text: str
    pages: List[int]
    pdf_name: str
    embedding: Optional[List[float]] = None


@dataclass(frozen=True)
class TreeIndex:
    h1_nodes: Dict[str, H1Node]
    h2_nodes: Dict[str, H2Node]
    lookup: Dict[str, Dict[str, List[str]]]

