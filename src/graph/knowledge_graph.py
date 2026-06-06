import re
import networkx as nx
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class KnowledgeGraph:
    """
    Entity-based knowledge graph over document chunks.
    Chunks are nodes; edges connect chunks that share named entities.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_store: Dict[str, Dict] = {}
        self.stop_entities: Set[str] = set()

    # Words that are capitalised for grammatical reasons, not because they are
    # named entities (question starters, pronouns, common sentence openers).
    _GENERIC_WORDS: Set[str] = {
        'what', 'who', 'why', 'how', 'when', 'where', 'which',
        'we', 'our', 'you', 'your', 'i', 'my', 'it', 'its',
        'yes', 'no', 'all', 'for', 'can', 'the', 'this', 'these',
        'those', 'any', 'each', 'both', 'also', 'and', 'but', 'not',
        'will', 'may', 'use', 'get', 'has', 'have', 'are', 'was',
        'were', 'been', 'that', 'with', 'from', 'please', 'contact',
        'note', 'example', 'whether',
    }

    def _extract_entities(self, text: str) -> Set[str]:
        entities: Set[str] = set()

        # CamelCase product/company names: TechVision, AutoML, MacBook, LinkedIn
        for m in re.findall(r'\b[A-Z][a-z]+[A-Z]\w*\b', text):
            entities.add(m.lower())

        # Pure acronyms: GPU, API, TPU, NLP, GDPR, CEO, HSA
        for m in re.findall(r'\b[A-Z]{2,}\b', text):
            entities.add(m.lower())

        # Multi-word proper noun phrases: "San Francisco", "Sarah Chen", "Series C"
        for m in re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b', text):
            entities.add(m.lower())

        # Single capitalised proper nouns (min 4 chars to skip "Inc", "The", etc.)
        for m in re.findall(r'\b[A-Z][a-z]{3,}\b', text):
            lower = m.lower()
            if lower not in self._GENERIC_WORDS:
                entities.add(lower)

        # Dollar amounts: $99/month, $2,000, $200M
        for m in re.findall(r'\$[\d,]+(?:[MBK])?(?:/\w+)?', text):
            entities.add(m)

        # Percentages: 80%, 99.9%
        for m in re.findall(r'\d+(?:\.\d+)?%', text):
            entities.add(m)

        return entities - self._GENERIC_WORDS

    def build_from_chunks(
        self,
        chunks: List[str],
        metadatas: List[Dict],
        ids: List[str],
        max_entity_freq: float = 0.35,
    ) -> None:
        """
        Build the graph.
        max_entity_freq: entities appearing in more than this fraction of chunks
        are treated as stop-words and excluded from edges (they connect everything
        to everything and produce noisy expansion).
        """
        self.graph.clear()
        self.entity_to_chunks.clear()
        self.chunk_store.clear()

        n_chunks = len(chunks)

        # First pass: collect all entities per chunk
        chunk_entities: Dict[str, Set[str]] = {}
        for chunk_id, text, meta in zip(ids, chunks, metadatas):
            entities = self._extract_entities(text)
            self.chunk_store[chunk_id] = {'text': text, 'metadata': meta, 'entities': entities}
            chunk_entities[chunk_id] = entities
            self.graph.add_node(chunk_id, node_type='chunk')
            for entity in entities:
                self.entity_to_chunks[entity].add(chunk_id)

        # Identify stop entities (too common to be discriminative)
        self.stop_entities = {
            e for e, cids in self.entity_to_chunks.items()
            if len(cids) / n_chunks > max_entity_freq
        }
        stop_entities = self.stop_entities

        # Second pass: wire edges only for non-stop entities
        for chunk_id, entities in chunk_entities.items():
            for entity in entities:
                if entity in stop_entities:
                    continue
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, node_type='entity')
                self.graph.add_edge(chunk_id, entity)

        # Chunk-to-chunk edges via shared non-stop entities
        for entity, chunk_ids in self.entity_to_chunks.items():
            if entity in stop_entities:
                continue
            chunk_list = list(chunk_ids)
            for i in range(len(chunk_list)):
                for j in range(i + 1, len(chunk_list)):
                    a, b = chunk_list[i], chunk_list[j]
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]['weight'] += 1
                    else:
                        self.graph.add_edge(a, b, weight=1)

        print(
            f"Knowledge graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, "
            f"{len(self.entity_to_chunks) - len(stop_entities)} discriminative entities "
            f"({len(stop_entities)} stop-entities filtered)"
        )

    def get_related_chunks(
        self, seed_ids: List[str], hops: int = 1
    ) -> List[Tuple[str, float]]:
        """BFS from seed chunk IDs; return related (chunk_id, score) pairs."""
        scores: Dict[str, float] = defaultdict(float)
        seed_set = set(seed_ids)

        for seed_id in seed_ids:
            if seed_id not in self.graph:
                continue
            visited = {seed_id}
            frontier = {seed_id}

            for hop in range(hops):
                decay = 1.0 / (hop + 2)
                next_frontier: Set[str] = set()
                for node in frontier:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        node_type = self.graph.nodes[neighbor].get('node_type')
                        if node_type == 'chunk':
                            w = self.graph[node][neighbor].get('weight', 1)
                            scores[neighbor] += decay * w
                        next_frontier.add(neighbor)
                frontier = next_frontier

        related = [(cid, s) for cid, s in scores.items() if cid not in seed_set]
        return sorted(related, key=lambda x: x[1], reverse=True)

    def query_by_entities(self, text: str) -> List[Tuple[str, float]]:
        """Score chunks by how many non-stop query entities they share."""
        query_entities = self._extract_entities(text) - self.stop_entities
        scores: Dict[str, float] = defaultdict(float)
        for entity in query_entities:
            for chunk_id in self.entity_to_chunks.get(entity, []):
                scores[chunk_id] += 1.0
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
