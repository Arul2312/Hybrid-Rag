import os
import yaml
from pathlib import Path
from typing import Dict, Any
import PyPDF2

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_text_file(file_path: str) -> str:
    """
    Read content from a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path: str) -> str:
    """
    Read content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from PDF
    """
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def load_documents_from_directory(directory_path: str) -> list:
    """
    Load all documents from a directory.
    Supports .txt and .pdf files.
    
    Args:
        directory_path: Path to the documents directory
        
    Returns:
        List of tuples (filename, content)
    """
    documents = []
    directory = Path(directory_path)
    
    for file_path in directory.glob("*"):
        if file_path.is_file():
            if file_path.suffix == '.txt':
                content = read_text_file(str(file_path))
                documents.append((file_path.name, content))
            elif file_path.suffix == '.pdf':
                content = read_pdf_file(str(file_path))
                documents.append((file_path.name, content))
    
    return documents

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split text into overlapping chunks aligned to word boundaries."""
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        # Greedily add words until we'd exceed chunk_size characters
        j = i
        length = 0
        while j < len(words):
            addition = len(words[j]) + (1 if j > i else 0)  # +1 for the space separator
            if length + addition > chunk_size and j > i:
                break
            length += addition
            j += 1

        # Edge case: single word longer than chunk_size
        if j == i:
            j = i + 1

        chunks.append(" ".join(words[i:j]))

        if j >= len(words):
            break

        # Overlap: walk backwards from j until we've covered chunk_overlap chars
        overlap_chars = 0
        overlap_start = j
        while overlap_start > i and overlap_chars + len(words[overlap_start - 1]) + 1 <= chunk_overlap:
            overlap_chars += len(words[overlap_start - 1]) + 1
            overlap_start -= 1

        i = max(i + 1, overlap_start)  # guaranteed forward progress

    return chunks