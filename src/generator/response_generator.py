from openai import OpenAI
from typing import List, Dict, Any
import os

class ResponseGenerator:
    """
    Generates responses using retrieved context and LLM.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 500):
        """
        Initialize the response generator with OpenAI.
        
        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        print(f"Response generator initialized with model: {model}")
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents with metadata
            
        Returns:
            Generated response
        """
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a helpful assistant. Answer the user's question using only "
                        "the context passages provided. Each passage is a chunk from a document "
                        "in the knowledge base — synthesize across all of them. "
                        "If the passages together contain enough information, give a complete answer. "
                        "Only say the context is insufficient if none of the passages address the question."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['document']}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create the final prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Context information is below:
---
{context}
---

Given the context information above, please answer the following question:
{query}

Answer:"""
        return prompt