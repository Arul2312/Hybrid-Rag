from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingHandler:
    """
    Handles document and query embeddings using sentence transformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
    
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            Array of embeddings
        """
        print(f"Encoding {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        embedding = self.model.encode([query])[0]
        return embedding