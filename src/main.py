import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, load_documents_from_directory, chunk_text
from embeddings.embedding_handler import EmbeddingHandler
from retriever.document_retriever import DocumentRetriever
from generator.response_generator import ResponseGenerator

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


class RAGSystem:
    """Graph-enhanced hybrid RAG system."""

    def __init__(self, config_path: str = None):
        print("Initializing RAG System...")

        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "config.yaml"
        self.config = load_config(str(config_path))

        self.embedding_handler = EmbeddingHandler(
            model_name=self.config["embedding"]["model"]
        )

        # Resolve persist_directory relative to project root
        persist_dir = str(PROJECT_ROOT / self.config["vector_store"]["persist_directory"])

        self.retriever = DocumentRetriever(
            persist_directory=persist_dir,
            collection_name=self.config["vector_store"]["collection_name"],
            embedding_handler=self.embedding_handler,
        )

        self.generator = ResponseGenerator(
            model=self.config["llm"]["model"],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"],
        )

        print("RAG System initialized.\n")

    def ingest_documents(self, documents_directory: str, force: bool = False) -> None:
        """
        Ingest documents into the vector store.
        Skips ingestion if the collection is already populated (pass force=True to re-ingest).
        """
        if not force and self.retriever.collection.count() > 0:
            print(
                f"Collection already has {self.retriever.collection.count()} chunks. "
                "Skipping ingestion (use force=True to re-ingest).\n"
            )
            return

        if force:
            self.retriever.clear_collection()

        print(f"Loading documents from: {documents_directory}")
        documents = load_documents_from_directory(documents_directory)

        if not documents:
            print("No documents found!")
            return

        print(f"Found {len(documents)} documents")

        all_chunks, all_metadata = [], []
        for filename, content in documents:
            chunks = chunk_text(
                content,
                chunk_size=self.config["chunking"]["chunk_size"],
                chunk_overlap=self.config["chunking"]["chunk_overlap"],
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"source": filename, "chunk_id": i})

        print(f"Created {len(all_chunks)} chunks")
        self.retriever.add_documents(all_chunks, all_metadata)
        print("Ingestion complete.\n")

    def query(self, question: str) -> str:
        print(f"\nQuery: {question}")
        print("-" * 80)

        retrieved_docs = self.retriever.retrieve(
            question, top_k=self.config["retrieval"]["top_k"]
        )

        print(f"Retrieved {len(retrieved_docs)} chunks:\n")
        for i, doc in enumerate(retrieved_docs, 1):
            rtype = doc.get("retrieval_type", "?")
            final = doc.get("final_score", doc.get("rrf_score", 0.0))
            graph = doc.get("graph_score", 0.0)
            print(f"  [{i}] type={rtype}  final={final:.4f}  graph={graph:.2f}  source={doc['metadata']['source']}")
            print(f"       {doc['document'][:150].strip()}…\n")

        print("-" * 80)
        return self.generator.generate_response(question, retrieved_docs)


def main():
    parser = argparse.ArgumentParser(description="Graph-enhanced hybrid RAG system")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing vector store and re-ingest all documents",
    )
    args = parser.parse_args()

    rag = RAGSystem()

    documents_dir = PROJECT_ROOT / "data" / "documents"
    if documents_dir.exists():
        rag.ingest_documents(str(documents_dir), force=args.fresh)
    else:
        print(f"Documents directory '{documents_dir}' not found.")
        documents_dir.mkdir(parents=True, exist_ok=True)
        print("Created empty documents directory — add documents and re-run.")
        return

    print("\nRAG System Ready! (type 'quit' to exit)")
    print("=" * 80)

    while True:
        user_query = input("\nYour question: ").strip()

        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_query:
            continue

        try:
            answer = rag.query(user_query)
            print(f"\nAnswer:\n{answer}")
            print("=" * 80)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
