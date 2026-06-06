import os
import re
import sys
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi

# Ensure src/ is on the path so graph can be imported regardless of how this
# module is loaded (as `retriever.document_retriever` or `src.retriever.…`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.knowledge_graph import KnowledgeGraph


class DocumentRetriever:
    """
    Graph-enhanced hybrid document retriever.

    Retrieval pipeline:
      1. Dense vector search  (ChromaDB / sentence-transformers)
      2. Sparse keyword search (BM25)
      3. Reciprocal Rank Fusion (RRF) to merge 1 & 2
      4. Knowledge-graph expansion of top-ranked chunks
    """

    def __init__(self, persist_directory: str, collection_name: str, embedding_handler):
        self.embedding_handler = embedding_handler

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 state
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []
        self.bm25_ids: List[str] = []

        # Chunk metadata cache (id → {document, metadata})
        self.chunk_store: Dict[str, Dict] = {}

        # Knowledge graph
        self.knowledge_graph = KnowledgeGraph()

        count = self.collection.count()
        print(f"Collection '{collection_name}' ready ({count} documents)")

        if count > 0:
            self._rebuild_indexes()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_indexes(self) -> None:
        """Rebuild BM25 index and knowledge graph from ChromaDB contents."""
        result = self.collection.get(include=["documents", "metadatas"])
        if not result["documents"]:
            return

        ids = result["ids"]
        docs = result["documents"]
        metas = result["metadatas"]

        self.bm25_corpus = docs
        self.bm25_ids = ids
        self.bm25 = BM25Okapi([self._tokenize(d) for d in docs])

        self.chunk_store = {
            id_: {"document": doc, "metadata": meta}
            for id_, doc, meta in zip(ids, docs, metas)
        }

        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.build_from_chunks(docs, metas, ids)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not documents:
            return

        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]

        embeddings = self.embedding_handler.encode_documents(documents)
        ids = [str(uuid.uuid4()) for _ in documents]

        emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        self.collection.add(
            embeddings=emb_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        # Update BM25
        self.bm25_corpus.extend(documents)
        self.bm25_ids.extend(ids)
        self.bm25 = BM25Okapi([self._tokenize(d) for d in self.bm25_corpus])

        # Update chunk store
        for id_, doc, meta in zip(ids, documents, metadatas):
            self.chunk_store[id_] = {"document": doc, "metadata": meta}

        # Rebuild knowledge graph over all chunks
        all_metas = [self.chunk_store[i]["metadata"] for i in self.bm25_ids]
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.build_from_chunks(self.bm25_corpus, all_metas, self.bm25_ids)

        print(f"Added {len(documents)} chunks (total: {len(self.bm25_corpus)})")

    # ------------------------------------------------------------------
    # Individual retrieval strategies
    # ------------------------------------------------------------------

    def _retrieve_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        n = min(top_k, self.collection.count())
        if n == 0:
            return []

        q_emb = self.embedding_handler.encode_query(query)
        q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb
        results = self.collection.query(
            query_embeddings=[q_emb_list],
            n_results=n,
        )

        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "retrieval_type": "vector",
            }
            for i in range(len(results["documents"][0]))
        ]

    def _retrieve_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.bm25_corpus:
            return []

        scores = self.bm25.get_scores(self._tokenize(query))
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "id": self.bm25_ids[i],
                "document": self.bm25_corpus[i],
                "metadata": self.chunk_store[self.bm25_ids[i]]["metadata"],
                "bm25_score": float(scores[i]),
                "retrieval_type": "bm25",
            }
            for i in top_idxs
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        *ranked_lists: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Merge any number of ranked lists with RRF."""
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Dict] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, 1):
                doc_id = doc["id"]
                rrf_scores[doc_id] += 1.0 / (k + rank)
                doc_map.setdefault(doc_id, doc)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {**doc_map[doc_id], "rrf_score": score, "retrieval_type": "hybrid"}
            for doc_id, score in merged
        ]

    # ------------------------------------------------------------------
    # Public retrieval entry point
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, graph_alpha: float = 0.3) -> List[Dict[str, Any]]:
        """
        Graph-enhanced hybrid retrieval.

        Pipeline:
          1. Dense vector search + sparse BM25 search, each fetching 3×top_k candidates.
          2. Reciprocal Rank Fusion (RRF) merges the two ranked lists.
          3. Knowledge-graph expansion: BFS from top hybrid seeds + entity matching
             surface additional candidate chunks and produce a graph relevance score.
          4. All candidates (hybrid + graph-only) are re-ranked by:
               final_score = rrf_score + graph_alpha × normalised_graph_score
          5. Return top_k from the re-ranked pool, tagged by retrieval type.
        """
        if self.collection.count() == 0:
            return []

        fetch_k = max(top_k * 3, 15)

        vector_results = self._retrieve_vector(query, fetch_k)
        bm25_results = self._retrieve_bm25(query, fetch_k)
        hybrid = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # --- Build candidate pool from hybrid results ---
        candidate_pool: Dict[str, Dict] = {d["id"]: d for d in hybrid}

        # --- Graph scores ---
        top_ids = [d["id"] for d in hybrid[:top_k]]
        graph_neighbours = self.knowledge_graph.get_related_chunks(top_ids, hops=1)
        entity_matches = self.knowledge_graph.query_by_entities(query)

        graph_scores: Dict[str, float] = {}
        for chunk_id, score in graph_neighbours + entity_matches:
            graph_scores[chunk_id] = graph_scores.get(chunk_id, 0.0) + score

        # Add graph-only candidates not found by vector/BM25
        for chunk_id in graph_scores:
            if chunk_id not in candidate_pool and chunk_id in self.chunk_store:
                chunk = self.chunk_store[chunk_id]
                candidate_pool[chunk_id] = {
                    "id": chunk_id,
                    "document": chunk["document"],
                    "metadata": chunk["metadata"],
                    "rrf_score": 0.0,
                    "retrieval_type": "graph",
                }

        # --- Re-rank: rrf_score + alpha × normalised_graph_score ---
        max_graph = max(graph_scores.values(), default=1.0)

        def combined_score(doc: Dict) -> float:
            rrf = doc.get("rrf_score", 0.0)
            g = graph_scores.get(doc["id"], 0.0) / max_graph
            return rrf + graph_alpha * g

        ranked = sorted(candidate_pool.values(), key=combined_score, reverse=True)

        # Tag retrieval type and attach scores
        results = []
        for doc in ranked[:top_k]:
            doc = dict(doc)
            g_score = graph_scores.get(doc["id"], 0.0)
            if g_score > 0 and doc.get("retrieval_type") == "hybrid":
                doc["retrieval_type"] = "hybrid+graph"
            doc["graph_score"] = g_score
            doc["final_score"] = combined_score(doc)
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_collection(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name=name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_ids = []
        self.chunk_store = {}
        self.knowledge_graph = KnowledgeGraph()
        print("Collection cleared")
