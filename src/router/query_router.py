import json
import os
from openai import OpenAI
from typing import Dict

# Retrieval config per query type.
# top_k_multiplier scales the base top_k from config.
# graph_alpha controls how much the graph score shifts final ranking (0–1).
# needs_multi_hop triggers iterative retrieval for queries that require chaining.
ROUTE_CONFIGS = {
    "factual": {
        "top_k_multiplier": 1.0,
        "graph_alpha": 0.2,
        "needs_multi_hop": False,
    },
    "relational": {
        "top_k_multiplier": 1.5,
        "graph_alpha": 0.5,
        "needs_multi_hop": True,
    },
    "comparative": {
        "top_k_multiplier": 2.0,
        "graph_alpha": 0.3,
        "needs_multi_hop": False,
    },
    "summarisation": {
        "top_k_multiplier": 2.0,
        "graph_alpha": 0.2,
        "needs_multi_hop": False,
    },
}

CLASSIFICATION_PROMPT = """Classify this query into exactly one category:

- factual: looking up a specific fact or value (e.g. "What is the API rate limit?")
- relational: requires chaining entities or understanding relationships (e.g. "How does X connect to Y?", "What does the person responsible for Z do?")
- comparative: side-by-side comparison of multiple named options (e.g. "What are the differences between plan A, B, and C?")
- summarisation: broad overview or listing of a topic (e.g. "Summarise all pricing", "List all benefits")

Query: {query}

Respond with JSON only — no markdown, no explanation:
{{"type": "<category>", "reason": "<one sentence why>"}}"""


def _parse_json(text: str) -> dict:
    """Parse JSON that may be wrapped in markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


class QueryRouter:
    """
    Classifies an incoming query and returns a retrieval configuration dict.

    Config keys returned:
      type             — query category
      reason           — one-line classification reason
      top_k_multiplier — scale factor for base top_k
      graph_alpha      — graph score weight in final re-ranking
      needs_multi_hop  — whether iterative retrieval should be used
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def classify(self, query: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(query=query)}],
                temperature=0,
                max_tokens=120,
            )
            result = _parse_json(response.choices[0].message.content)
            query_type = result.get("type", "factual")
            reason = result.get("reason", "")
        except Exception as e:
            query_type = self._heuristic_classify(query)
            reason = f"heuristic fallback ({e})"

        if query_type not in ROUTE_CONFIGS:
            query_type = "factual"

        config = ROUTE_CONFIGS[query_type].copy()
        config["type"] = query_type
        config["reason"] = reason
        return config

    def _heuristic_classify(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["difference", "compare", "vs", "versus", "between", "contrast"]):
            return "comparative"
        if any(w in q for w in ["summarise", "summarize", "overview", "all ", "list all", "everything"]):
            return "summarisation"
        if any(w in q for w in ["relate", "connect", "relationship", "how does", "why does", "leads to"]):
            return "relational"
        return "factual"
