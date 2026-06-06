import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch
from retriever.document_retriever import DocumentRetriever


def _make_retriever(tmp_path: str) -> DocumentRetriever:
    mock_embed = MagicMock()
    mock_embed.encode_documents.return_value = [[0.1] * 384]
    mock_embed.encode_query.return_value = [0.1] * 384
    return DocumentRetriever(
        persist_directory=tmp_path,
        collection_name="test_collection",
        embedding_handler=mock_embed,
    )


class TestDocumentRetriever(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        self.retriever = _make_retriever(self.tmp_dir)

    def test_retrieve_returns_list(self):
        self.retriever.add_documents(
            ["The quick brown fox jumps over the lazy dog."],
            [{"source": "test.txt", "chunk_id": 0}],
        )
        results = self.retriever.retrieve("fox", top_k=1)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_retrieve_empty_collection_returns_empty(self):
        results = self.retriever.retrieve("anything", top_k=3)
        self.assertEqual(results, [])

    def test_result_has_required_keys(self):
        self.retriever.add_documents(
            ["TechVision Inc. is an AI company based in San Francisco."],
            [{"source": "info.txt", "chunk_id": 0}],
        )
        results = self.retriever.retrieve("TechVision", top_k=1)
        self.assertTrue(len(results) > 0)
        doc = results[0]
        self.assertIn("document", doc)
        self.assertIn("metadata", doc)
        self.assertIn("retrieval_type", doc)

    def test_clear_collection(self):
        self.retriever.add_documents(
            ["Some document text here."],
            [{"source": "doc.txt", "chunk_id": 0}],
        )
        self.retriever.clear_collection()
        self.assertEqual(self.retriever.collection.count(), 0)
        self.assertEqual(self.retriever.retrieve("something"), [])


if __name__ == "__main__":
    unittest.main()
