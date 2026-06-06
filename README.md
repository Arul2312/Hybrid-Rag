# Graph-Enhanced Hybrid RAG System

A RAG (Retrieval-Augmented Generation) system that combines dense vector search, sparse keyword search, and a knowledge graph to retrieve more relevant context than any single method alone.

## How retrieval works

```
Query
  │
  ▼
Query Router  (LLM classifier)
  │  classifies into: factual | comparative | relational | summarisation
  │  sets: top_k multiplier, graph_alpha, needs_multi_hop
  │
  ├─── Single-pass retrieval  (factual / comparative / summarisation)
  │      │
  │      ├── Vector search (ChromaDB + sentence-transformers)  ─┐
  │      │                                                       ├── RRF fusion ──► candidate pool
  │      ├── BM25 keyword search                               ─┘
  │      │
  │      ├── Knowledge graph expansion
  │      │      • BFS from top-ranked seeds (related chunks via shared entities)
  │      │      • Entity match on query terms
  │      │
  │      └── Re-rank: final_score = rrf_score + α × graph_score
  │
  └─── Multi-hop retrieval  (relational queries only)
         │
         ├── Hop 1: retrieve with original query
         │     └── LLM gap check: "is this enough to answer?"
         │           ├── Yes → done
         │           └── No  → generate follow-up question
         ├── Hop 2: retrieve with follow-up question
         │     └── LLM gap check …
         └── … up to max_hops, deduplicating chunks across hops
```

Each retrieved chunk is tagged with its retrieval type in the output:

| Type | Meaning |
|---|---|
| `hybrid` | Found by vector and/or BM25 search |
| `hybrid+graph` | Found by vector/BM25 and boosted by the graph score |
| `graph` | Not found by vector or BM25 — surfaced only via knowledge graph traversal |

Query routing adjusts retrieval settings per query type:

| Query type | Example | top_k multiplier | graph_alpha | Multi-hop |
|---|---|---|---|---|
| `factual` | "What is the API rate limit?" | 1× | 0.2 | No |
| `comparative` | "Compare Starter vs Professional vs Enterprise" | 2× | 0.3 | No |
| `relational` | "How does employee training connect to customer support?" | 1.5× | 0.5 | Yes |
| `summarisation` | "Summarise all pricing options" | 2× | 0.2 | No |

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key** in `.env`:
```
OPENAI_API_KEY=your_key_here
```

3. **Add documents** — place `.txt` or `.pdf` files in `data/documents/`

4. **Run**:
```bash
# First run — ingests documents automatically
python src/main.py

# Force re-ingestion (after adding/changing documents or tuning chunk settings)
python src/main.py --fresh
```

## Project structure

```
rag-app/
├── config/config.yaml          # All tuning parameters
├── data/
│   ├── documents/              # Source documents (.txt, .pdf)
│   └── chroma_db/              # Persisted vector store (auto-created)
├── src/
│   ├── main.py                 # Entry point and RAGSystem class
│   ├── embeddings/
│   │   └── embedding_handler.py    # Sentence-transformer embeddings
│   ├── router/
│   │   └── query_router.py         # LLM query classifier with heuristic fallback
│   ├── retriever/
│   │   ├── document_retriever.py   # Hybrid retriever: vector + BM25 + graph + RRF
│   │   └── multi_hop.py            # Iterative retrieve → gap-check → re-retrieve loop
│   ├── graph/
│   │   └── knowledge_graph.py      # Entity co-occurrence graph and BFS traversal
│   ├── generator/
│   │   └── response_generator.py   # OpenAI GPT response generation
│   └── utils/
│       └── helpers.py              # Document loading and word-boundary chunker
└── tests/
    ├── test_retriever.py
    └── test_router.py
```

## Configuration

All parameters live in `config/config.yaml`:

```yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"

vector_store:
  persist_directory: "data/chroma_db"
  collection_name: "rag_collection"

llm:
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 500

retrieval:
  top_k: 7             # Base value — scaled by router's top_k_multiplier
  rrf_k: 60            # RRF constant — higher = smoother rank blending
  graph_hops: 1        # BFS depth from seed chunks into the knowledge graph

# Query router — classifies each query and adjusts retrieval settings
routing:
  enabled: true
  model: "gpt-3.5-turbo"   # Use gpt-4o-mini for cheaper/faster classification

# Multi-hop retrieval — iterative retrieve → gap-check → re-retrieve loop
# Only fires when the router classifies a query as "relational"
multi_hop:
  enabled: true
  max_hops: 3               # Maximum retrieval iterations
  model: "gpt-3.5-turbo"

chunking:
  chunk_size: 500  # Characters per chunk (word-boundary aligned)
  chunk_overlap: 50
```

### Tuning tips

- **`top_k`** is the most impactful lever. Multi-faceted questions (e.g. "compare cost, limits, and support across three plans") need `top_k` large enough to cover all facets. A good rule of thumb: `top_k ≥ number of distinct sub-topics + 2`.
- **`graph_alpha`** (hardcoded at `0.3` in `document_retriever.py`) controls how much the graph score shifts the final ranking. Raise it if graph-expanded chunks are being crowded out by noisy hybrid results.
- **`chunk_size`** affects how much context the LLM sees per chunk and how finely the graph can discriminate. Smaller chunks = more precise edges; larger chunks = more context per result.
- Re-run with `--fresh` any time you change `chunking` settings, since the vector store and graph are built from the stored chunks.

## The knowledge graph

The graph is an entity co-occurrence graph built at ingestion time:

- **Nodes**: document chunks + extracted named entities (CamelCase names, acronyms, multi-word proper nouns, dollar amounts, percentages)
- **Edges**: chunk↔entity membership; weighted chunk↔chunk edges where weight = number of shared entities
- **Stop-entity filter**: entities appearing in more than 35% of chunks are excluded from edges — they connect everything to everything and add noise rather than signal
- **At query time**: the top hybrid results seed a BFS that surfaces neighbouring chunks. Graph-expanded candidates are added to the pool and re-ranked by `final_score = rrf_score + 0.3 × normalised_graph_score` before the final top-k is returned

This is particularly effective for cross-document synthesis questions where the answer spans chunks that are not semantically or lexically similar to each other but are connected through shared entities.

## Limitations and next steps

The current graph uses regex-based entity extraction. A production implementation would replace this with:

| Current | Production equivalent |
|---|---|
| Regex (CamelCase, acronyms) | spaCy NER or LLM-extracted named entities |
| Co-occurrence edges | Typed triples: `TechVision → offers → EnterprisePlan` |
| BFS expansion | Multi-hop traversal over typed relationship edges |
| No clustering | Community detection for summarising entity clusters |

## Programmatic usage

```python
from src.main import RAGSystem

rag = RAGSystem()

# Ingestion (skips automatically if data is already loaded)
rag.ingest_documents("data/documents")
rag.ingest_documents("data/documents", force=True)  # force re-ingest

# Query
answer = rag.query("What is the vacation policy?")
print(answer)
```

## Running tests

```bash
pytest tests/ -v
```
