"""
Test script to verify RAG setup and download embedding model
"""
from sentence_transformers import SentenceTransformer

print("Downloading embedding model (this may take a few minutes on first run)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model downloaded and loaded successfully!")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Test encoding
test_text = "This is a test sentence."
embedding = model.encode([test_text])
print(f"Test embedding shape: {embedding.shape}")
print("Setup complete!")
