from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from rag_pipeline.schemas import DocumentRoot, H1Node, H2Node, TreeIndex


class MongoTreeStore:
    def __init__(self, uri: str, db_name: str = "legal_rag") -> None:
        try:
            from pymongo import MongoClient
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pymongo is required for MongoTreeStore. Install with `pip install pymongo`."
            ) from exc
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.documents = self.db.documents
        self.h1_nodes = self.db.h1_nodes
        self.h2_nodes = self.db.h2_nodes
        self.lookup = self.db.lookup

    def ensure_indexes(self) -> None:
        self.documents.create_index("doc_id", unique=True)
        self.h1_nodes.create_index([("doc_id", 1), ("node_id", 1)], unique=True)
        self.h2_nodes.create_index([("doc_id", 1), ("node_id", 1)], unique=True)
        self.h2_nodes.create_index([("doc_id", 1), ("parent", 1)])
        self.lookup.create_index([("doc_id", 1), ("h1_id", 1)], unique=True)

    def save_tree(self, doc_root: DocumentRoot, tree: TreeIndex) -> None:
        self.ensure_indexes()
        self.documents.update_one(
            {"doc_id": doc_root.doc_id},
            {"$set": {"doc_id": doc_root.doc_id, "pdf_name": doc_root.pdf_name}},
            upsert=True,
        )
        for h1 in tree.h1_nodes.values():
            payload = asdict(h1)
            payload["doc_id"] = doc_root.doc_id
            self.h1_nodes.update_one(
                {"doc_id": doc_root.doc_id, "node_id": h1.node_id},
                {"$set": payload},
                upsert=True,
            )
        for h2 in tree.h2_nodes.values():
            payload = asdict(h2)
            payload["doc_id"] = doc_root.doc_id
            self.h2_nodes.update_one(
                {"doc_id": doc_root.doc_id, "node_id": h2.node_id},
                {"$set": payload},
                upsert=True,
            )
        for h1_id, entry in tree.lookup.items():
            self.lookup.update_one(
                {"doc_id": doc_root.doc_id, "h1_id": h1_id},
                {"$set": {"doc_id": doc_root.doc_id, "h1_id": h1_id, **entry}},
                upsert=True,
            )

    def load_tree(self, doc_id: str) -> TreeIndex:
        h1_nodes: Dict[str, H1Node] = {}
        h2_nodes: Dict[str, H2Node] = {}
        lookup: Dict[str, Dict[str, List[str]]] = {}

        for record in self.h1_nodes.find({"doc_id": doc_id}):
            record.pop("_id", None)
            record.pop("doc_id", None)
            h1_node = H1Node(**record)
            h1_nodes[h1_node.node_id] = h1_node

        for record in self.h2_nodes.find({"doc_id": doc_id}):
            record.pop("_id", None)
            record.pop("doc_id", None)
            h2_node = H2Node(**record)
            h2_nodes[h2_node.node_id] = h2_node

        for record in self.lookup.find({"doc_id": doc_id}):
            record.pop("_id", None)
            record.pop("doc_id", None)
            h1_id = record.pop("h1_id")
            lookup[h1_id] = record

        return TreeIndex(h1_nodes=h1_nodes, h2_nodes=h2_nodes, lookup=lookup)
