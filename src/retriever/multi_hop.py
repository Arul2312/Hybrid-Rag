import json
import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple

GAP_PROMPT = """You are helping a retrieval system decide whether to fetch more information.

Original question: {query}

Top relevant context retrieved so far ({n_chunks} chunks total, showing most relevant):
{context}

Previously asked follow-up questions (do NOT repeat any of these):
{asked}

Does this context contain enough information to fully answer the original question?

If YES:  {{"sufficient": true, "follow_up": null}}
If NO:   {{"sufficient": false, "follow_up": "<a NEW follow-up question not in the list above>"}}

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
      3. Sort accumulated context by relevance score.
      4. Ask the LLM (showing top-8 by relevance) whether context is sufficient.
      5. If not, use the LLM-generated follow-up question as the next sub-query.
         Previously asked questions are shown to prevent repetition.
      6. Repeat up to max_hops times.

    Returns deduplicated chunks sorted by final_score so callers can safely
    slice the top-N for the generator.
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
          all_chunks  — deduplicated chunks sorted by final_score (best first)
          hop_trace   — sub-questions asked at each hop (empty if one hop sufficed)
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

            # Keep accumulated context sorted by relevance for gap-check and callers
            all_chunks.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)

            print(f"           → {len(new_chunks)} new chunk(s)  (total: {len(all_chunks)})")

            # Stop early if this hop added nothing new
            if not new_chunks and hop > 0:
                print("           → no new chunks; stopping early")
                break

            # Only check for a gap if there are hops remaining to use the result
            if hop < self.max_hops - 1:
                follow_up = self._identify_gap(query, all_chunks, asked=hop_trace)
                if follow_up is None:
                    break
                hop_trace.append(follow_up)
                current_query = follow_up

        # Final re-rank: blend hop-specific scores with vector similarity to
        # the ORIGINAL query so that evidence found by targeted sub-questions
        # is fairly compared against evidence from hop 1.
        all_chunks = self._rerank_by_original_query(query, all_chunks)

        return all_chunks, hop_trace

    def _rerank_by_original_query(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Re-score accumulated chunks using vector similarity to the original query.
        Corrects for hop-specific score bias when chunks were retrieved against
        different sub-queries.
        Final score = 0.5 × normalised_hop_score + 0.5 × normalised_vector_rank_score
        """
        if len(chunks) <= 1:
            return chunks
        try:
            q_emb = self.retriever.embedding_handler.encode_query(query)
            q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb

            # Query the collection for all items so we get original-query ranks
            n = min(self.retriever.collection.count(), len(chunks) + 20)
            results = self.retriever.collection.query(
                query_embeddings=[q_emb_list],
                n_results=n,
            )
            # Lower rank index = more similar to original query
            vector_rank: Dict[str, int] = {
                doc_id: rank for rank, doc_id in enumerate(results["ids"][0])
            }

            # Normalise hop scores to [0, 1]
            scores = [c.get("final_score", 0.0) for c in chunks]
            max_s = max(scores) or 1.0

            for chunk in chunks:
                norm_hop = chunk.get("final_score", 0.0) / max_s
                vrank = vector_rank.get(chunk["id"], n)
                norm_vec = 1.0 / (1.0 + vrank)   # rank 0 → 1.0, rank n → ~0
                chunk["final_score"] = 0.5 * norm_hop + 0.5 * norm_vec

            return sorted(chunks, key=lambda c: c["final_score"], reverse=True)
        except Exception:
            return chunks  # leave unchanged on any failure

    def _identify_gap(
        self,
        original_query: str,
        context: List[Dict],
        asked: List[str] = None,
    ) -> Optional[str]:
        """
        Show the top-8 most relevant chunks (already sorted) to the LLM and ask
        whether they are sufficient to answer the original query.
        Returns a follow-up question, or None if the context is sufficient.
        """
        # context is pre-sorted by final_score; show the most relevant 8
        context_text = "\n\n".join(
            f"[{i + 1}] (source: {c['metadata'].get('source', '?')}) {c['document'][:350]}"
            for i, c in enumerate(context[:8])
        )

        asked_text = "\n".join(f"- {q}" for q in asked) if asked else "None"

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
                            asked=asked_text,
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
            # Guard: if the LLM returns a question identical to a prior one, stop
            if follow_up and asked and follow_up.strip() in [q.strip() for q in asked]:
                return None
            return follow_up if follow_up else None
        except Exception:
            return None
