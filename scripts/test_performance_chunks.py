#!/usr/bin/env python3
"""Test which PERFORMANCE chunks are retrieved."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder

def test_performance_chunks():
    """Test which PERFORMANCE chunks are retrieved."""
    user_id = 1
    query = "Welche Spezifikationen hat das Lenovo Thinkpad E14 Gen 7"
    
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    
    # Test retrieval without reranking first
    retrieved = retriever.retrieve(user_id, query, n_results=50)
    
    print(f"Retrieved {len(retrieved)} documents\n")
    
    # Find chunks with PERFORMANCE
    performance_chunks = []
    for i, doc in enumerate(retrieved, 1):
        text_lower = doc["text"].lower()
        if "performance" in text_lower:
            performance_chunks.append((i, doc))
    
    print(f"Found {len(performance_chunks)} chunks with PERFORMANCE:\n")
    
    for rank, doc in performance_chunks[:10]:
        similarity = doc.get("similarity", 0)
        print(f"Rank {rank}. Similarity: {similarity:.4f}")
        print(f"   {doc['text'][:300]}...")
        print()

if __name__ == "__main__":
    test_performance_chunks()

