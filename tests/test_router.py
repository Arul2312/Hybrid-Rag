import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router.query_router import QueryRouter, ROUTE_CONFIGS


def _mock_router(return_type: str) -> QueryRouter:
    router = QueryRouter.__new__(QueryRouter)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=f'{{"type": "{return_type}", "reason": "test"}}'))]
    )
    router.client = mock_client
    router.model = "gpt-3.5-turbo"
    return router


class TestQueryRouter(unittest.TestCase):

    def test_classify_returns_known_type(self):
        router = _mock_router("factual")
        result = router.classify("What is the price?")
        self.assertIn(result["type"], ROUTE_CONFIGS)

    def test_factual_does_not_need_multi_hop(self):
        router = _mock_router("factual")
        result = router.classify("What is the API rate limit?")
        self.assertFalse(result["needs_multi_hop"])

    def test_relational_needs_multi_hop(self):
        router = _mock_router("relational")
        result = router.classify("How does employee training connect to customer support?")
        self.assertTrue(result["needs_multi_hop"])

    def test_comparative_has_high_top_k_multiplier(self):
        router = _mock_router("comparative")
        result = router.classify("Compare Starter, Professional, and Enterprise plans.")
        self.assertGreaterEqual(result["top_k_multiplier"], 1.5)

    def test_heuristic_fallback_comparative(self):
        router = QueryRouter.__new__(QueryRouter)
        router.client = None
        router.model = "gpt-3.5-turbo"
        result = router._heuristic_classify("What are the differences between plan A and plan B?")
        self.assertEqual(result, "comparative")

    def test_heuristic_fallback_summarisation(self):
        router = QueryRouter.__new__(QueryRouter)
        result = router._heuristic_classify("Summarise all pricing options")
        self.assertEqual(result, "summarisation")

    def test_unknown_type_falls_back_to_factual(self):
        router = _mock_router("unknown_category")
        result = router.classify("Something random")
        self.assertEqual(result["type"], "factual")

    def test_config_keys_present(self):
        router = _mock_router("relational")
        result = router.classify("How does X relate to Y?")
        for key in ("type", "reason", "top_k_multiplier", "graph_alpha", "needs_multi_hop"):
            self.assertIn(key, result)


class TestMultiHopRetriever(unittest.TestCase):

    def _make_chunk(self, id_, text, source):
        return {
            "id": id_,
            "document": text,
            "metadata": {"source": source},
            "retrieval_type": "hybrid",
            "final_score": 0.5,
            "graph_score": 1.0,
        }

    def test_deduplication_across_hops(self):
        from retriever.multi_hop import MultiHopRetriever

        mock_retriever = MagicMock()
        chunk_a = self._make_chunk("a", "Chunk A text", "doc1.txt")
        chunk_b = self._make_chunk("b", "Chunk B text", "doc2.txt")

        # Hop 1 returns A and B; hop 2 returns A again (duplicate) + B
        mock_retriever.retrieve.side_effect = [
            [chunk_a, chunk_b],
            [chunk_a, chunk_b],
        ]

        mhr = MultiHopRetriever.__new__(MultiHopRetriever)
        mhr.retriever = mock_retriever
        mhr.model = "gpt-3.5-turbo"
        mhr.max_hops = 2

        # Make _identify_gap return a follow-up on first call, None on second
        mhr._identify_gap = MagicMock(side_effect=["What else?", None])

        all_chunks, hop_trace = mhr.retrieve("test query", top_k=3)

        self.assertEqual(len(all_chunks), 2)  # A and B, no duplicates
        self.assertEqual(len(hop_trace), 1)

    def test_stops_when_context_sufficient(self):
        from retriever.multi_hop import MultiHopRetriever

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            self._make_chunk("a", "Answer is here", "doc.txt")
        ]

        mhr = MultiHopRetriever.__new__(MultiHopRetriever)
        mhr.retriever = mock_retriever
        mhr.model = "gpt-3.5-turbo"
        mhr.max_hops = 3
        mhr._identify_gap = MagicMock(return_value=None)  # sufficient after hop 1

        all_chunks, hop_trace = mhr.retrieve("test query", top_k=3)

        self.assertEqual(mock_retriever.retrieve.call_count, 1)
        self.assertEqual(hop_trace, [])


if __name__ == "__main__":
    unittest.main()
