import json
import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple

GAP_PROMPT = """You are helping a retrieval system decide whether to fetch more information.

Original question: {query}

Context retrieved so far ({n_chunks} chunks):
{context}

Does this context contain enough information to fully answer the original question?

If YES:  {{"sufficient": true, "follow_up": null}}
If NO:   {{"sufficient": false, "follow_up": "<the single most important follow-up question>"}}

Respond with JSON only — no markdown, no explanation."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


class MultiHopRetriever:
    """
    Wraps a DocumentRetriever with an iterative retrieval loop.

    Pipeline per hop:
      1. Retrieve chunks for the current sub-query.
      2. Merge new chunks into the accumulated context (deduplicating by ID).
      3. Ask the LLM whether the accumulated context is sufficient.
      4. If not, use the LLM-generated follow-up question as the next sub-query.
      5. Repeat up to max_hops times.
    """

    def __init__(self, retriever, model: str = "gpt-3.5-turbo", max_hops: int = 3):
        self.retriever = retriever
        self.model = model
        self.max_hops = max_hops
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def retrieve(
        self,
        query: str,
        top_k: int,
        graph_alpha: float = 0.3,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Run iterative retrieval.

        Returns:
          all_chunks  — deduplicated list of chunks across all hops
          hop_trace   — sub-questions generated at each hop (empty if one hop sufficed)
        """
        all_chunks: List[Dict] = []
        seen_ids: set = set()
        hop_trace: List[str] = []
        current_query = query

        for hop in range(self.max_hops):
            print(f"    Hop {hop + 1}: \"{current_query}\"")

            chunks = self.retriever.retrieve(
                current_query, top_k=top_k, graph_alpha=graph_alpha
            )

            new_chunks = [c for c in chunks if c["id"] not in seen_ids]
            for c in new_chunks:
                seen_ids.add(c["id"])
            all_chunks.extend(new_chunks)

            print(f"           → {len(new_chunks)} new chunk(s)  (total: {len(all_chunks)})")

            follow_up = self._identify_gap(query, all_chunks)
            if follow_up is None:
                break

            hop_trace.append(follow_up)
            current_query = follow_up

        return all_chunks, hop_trace

    def _identify_gap(self, original_query: str, context: List[Dict]) -> Optional[str]:
        """
        Ask the LLM whether the accumulated context is sufficient.
        Returns a follow-up question string, or None if context is sufficient.
        """
        # Truncate each chunk so the prompt stays within token budget
        context_text = "\n\n".join(
            f"[{i + 1}] (source: {c['metadata'].get('source', '?')}) {c['document'][:350]}"
            for i, c in enumerate(context[:10])
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": GAP_PROMPT.format(
                            query=original_query,
                            n_chunks=len(context),
                            context=context_text,
                        ),
                    }
                ],
                temperature=0,
                max_tokens=150,
            )
            result = _parse_json(response.choices[0].message.content)
            if result.get("sufficient"):
                return None
            follow_up = result.get("follow_up")
            return follow_up if follow_up else None
        except Exception:
            return None  # treat parse/API failures as "sufficient" to avoid loops
