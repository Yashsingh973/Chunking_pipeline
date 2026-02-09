from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from rag_pipeline.schemas import DocumentRoot, H1Node, H2Node, TreeIndex
from rag_pipeline.utils import dedupe_preserve_order, summarize_text


PAGE_MARKER_RE = re.compile(r"\[\[PAGE\s+(\d+)\]\]", re.IGNORECASE)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class IngestionConfig:
    doc_id: str
    pdf_name: str
    h1_prefix: str = "h1"
    h2_prefix: str = "h2"
    summarize_fn: Optional[Callable[[str], str]] = None


def _extract_pages(text: str) -> List[int]:
    pages = [int(match.group(1)) for match in PAGE_MARKER_RE.finditer(text)]
    return sorted(set(pages))


def _find_headings(lines: List[str]) -> List[Tuple[int, int, str]]:
    headings = []
    for idx, line in enumerate(lines):
        match = HEADER_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        head = match.group(2).strip()
        headings.append((idx, level, head))
    return headings


def parse_markdown_to_tree(markdown_text: str, config: IngestionConfig) -> TreeIndex:
    lines = [line.rstrip() for line in markdown_text.splitlines()]
    doc_root = DocumentRoot(pdf_name=config.pdf_name, doc_id=config.doc_id)

    h1_nodes = {}
    h2_nodes = {}
    lookup = {}

    headings = _find_headings(lines)
    h1_indices = [(idx, head) for idx, level, head in headings if level == 1]

    for h1_counter, (h1_start_idx, h1_head) in enumerate(h1_indices, start=1):
        h1_id = f"{config.h1_prefix}_{h1_counter:02d}"
        h1_end_idx = next(
            (idx for idx, _ in h1_indices if idx > h1_start_idx),
            len(lines),
        )
        h1_block = lines[h1_start_idx + 1 : h1_end_idx]
        h1_text = "\n".join(h1_block).strip()
        h1_pages = _extract_pages(h1_text)

        h2_ids = []
        h2_counter = 0
        h2_headings = [
            (idx, head)
            for idx, level, head in headings
            if level == 2 and h1_start_idx < idx < h1_end_idx
        ]

        if h2_headings:
            first_h2_idx = h2_headings[0][0]
            pre_h2_block = lines[h1_start_idx + 1 : first_h2_idx]
            last_pages = _extract_pages("\n".join(pre_h2_block).strip())
            for h2_start_idx, h2_head in h2_headings:
                h2_end_idx = next(
                    (idx for idx, _ in h2_headings if idx > h2_start_idx),
                    h1_end_idx,
                )
                h2_block = lines[h2_start_idx + 1 : h2_end_idx]
                h2_text = "\n".join(h2_block).strip()
                h2_pages = _extract_pages(h2_text)
                if h2_pages:
                    last_pages = h2_pages
                else:
                    h2_pages = last_pages
                h2_counter += 1
                h2_id = f"{config.h2_prefix}_{h1_counter:02d}_{h2_counter:02d}"
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
        else:
            h2_counter = 1
            h2_id = f"{config.h2_prefix}_{h1_counter:02d}_{h2_counter:02d}"
            h2_nodes[h2_id] = H2Node(
                node_id=h2_id,
                parent=h1_id,
                level="H2",
                head=h1_head,
                text=h1_text,
                pages=h1_pages,
                pdf_name=doc_root.pdf_name,
            )
            h2_ids.append(h2_id)

        summarize = config.summarize_fn or summarize_text
        h1_summary = summarize(h1_text)
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
