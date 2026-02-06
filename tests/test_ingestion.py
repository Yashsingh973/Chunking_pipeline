import os
import unittest

from rag_pipeline.ingestion import IngestionConfig, parse_markdown_to_tree
from rag_pipeline.llm import load_summarizer_from_env


class IngestionTests(unittest.TestCase):
    def test_h2_fallback_created_when_missing(self) -> None:
        markdown = "# Title\n[[PAGE 1]]\nSome statutory text.\n"
        tree = parse_markdown_to_tree(
            markdown,
            IngestionConfig(doc_id="doc_01", pdf_name="doc.pdf"),
        )

        self.assertEqual(len(tree.h1_nodes), 1)
        self.assertEqual(len(tree.h2_nodes), 1)

        h1 = next(iter(tree.h1_nodes.values()))
        h2 = next(iter(tree.h2_nodes.values()))
        self.assertEqual(h1.children, [h2.node_id])
        self.assertIn("Some statutory text", h2.text)

    def test_h2_chunks_created_for_multiple_sections(self) -> None:
        markdown = (
            "# Title\n"
            "[[PAGE 1]]\n"
            "## Section A\n"
            "A text.\n"
            "## Section B\n"
            "B text.\n"
        )
        tree = parse_markdown_to_tree(
            markdown,
            IngestionConfig(doc_id="doc_02", pdf_name="doc.pdf"),
        )

        self.assertEqual(len(tree.h1_nodes), 1)
        self.assertEqual(len(tree.h2_nodes), 2)

    def test_load_summarizer_from_env_none_without_url(self) -> None:
        original = dict(os.environ)
        try:
            os.environ.pop("LLM_API_URL", None)
            summarizer = load_summarizer_from_env()
            self.assertIsNone(summarizer)
        finally:
            os.environ.clear()
            os.environ.update(original)


if __name__ == "__main__":
    unittest.main()
