from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from rag_pipeline.schemas import DocumentRoot, H1Node, H2Node, TreeIndex
from rag_pipeline.utils import dedupe_preserve_order, summarize_text


PAGE_MARKER_RE = re.compile(r"\[\[PAGE\s+(\d+)\]\]", re.IGNORECASE)


@dataclass(frozen=True)
class IngestionConfig:
    doc_id: str
    pdf_name: str
    h1_prefix: str = "h1"
    h2_prefix: str = "h2"


def _extract_pages(text: str) -> List[int]:
    pages = [int(match.group(1)) for match in PAGE_MARKER_RE.finditer(text)]
    return sorted(set(pages))


def _split_by_heading(lines: List[str], heading_marker: str) -> List[Tuple[str, List[str]]]:
    sections: List[Tuple[str, List[str]]] = []
    current_head = ""
    current_lines: List[str] = []
    for line in lines:
        if line.startswith(heading_marker + " "):
            if current_head or current_lines:
                sections.append((current_head, current_lines))
            current_head = line[len(heading_marker) + 1 :].strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_head, current_lines))
    return sections


def parse_markdown_to_tree(markdown_text: str, config: IngestionConfig) -> TreeIndex:
    lines = [line.rstrip() for line in markdown_text.splitlines()]
    doc_root = DocumentRoot(pdf_name=config.pdf_name, doc_id=config.doc_id)

    h1_sections = _split_by_heading(lines, "#")
    h1_nodes = {}
    h2_nodes = {}
    lookup = {}

    h1_counter = 0
    for h1_head, h1_lines in h1_sections:
        if not h1_head:
            continue
        h1_counter += 1
        h1_id = f"{config.h1_prefix}_{h1_counter:02d}"

        h2_sections = _split_by_heading(h1_lines, "##")
        h2_ids = []
        h2_counter = 0
        h1_text = "\n".join(h1_lines).strip()
        h1_pages = _extract_pages(h1_text)

        for h2_head, h2_lines in h2_sections:
            if not h2_head:
                continue
            h2_counter += 1
            h2_id = f"{config.h2_prefix}_{h1_counter:02d}_{h2_counter:02d}"
            h2_text = "\n".join(h2_lines).strip()
            h2_pages = _extract_pages(h2_text) or h1_pages
            h2_nodes[h2_id] = H2Node(
                node_id=h2_id,
                parent=h1_id,
                level="H2",
                head=h2_head,
                text=h2_text,
                pages=h2_pages,
                pdf_name=doc_root.pdf_name,
            )
            h2_ids.append(h2_id)

        h1_summary = summarize_text(h1_text)
        h1_nodes[h1_id] = H1Node(
            node_id=h1_id,
            level="H1",
            head=h1_head,
            summary=h1_summary,
            pages=h1_pages,
            pdf_name=doc_root.pdf_name,
            children=h2_ids,
        )
        lookup[h1_id] = {"summary": h1_summary, "children": h2_ids}

    return TreeIndex(h1_nodes=h1_nodes, h2_nodes=h2_nodes, lookup=lookup)


def combine_pages(h1_nodes: List[H1Node], h2_nodes: List[H2Node]) -> List[int]:
    pages = []
    for node in h1_nodes:
        pages.extend(node.pages)
    for node in h2_nodes:
        pages.extend(node.pages)
    return dedupe_preserve_order([page for page in pages if page is not None])

