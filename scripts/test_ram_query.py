#!/usr/bin/env python3
"""Test RAM query retrieval."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder

def test_ram_query():
    """Test RAM query retrieval."""
    db = SessionLocal()
    
    try:
        user_id = 1
        retriever = Retriever(VectorStore(), Embedder())
        
        queries = [
            "Wieviel RAM hat das Thinkpad E14?",
            "Wieviel memory hat das Lenovo Thinkpad E14?",
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print('='*80)
            
            results = retriever.retrieve(user_id, query, n_results=10)
            print(f"\nRetrieved {len(results)} documents\n")
            
            # Find the chunk with RAM specs
            ram_chunk_id = "73216111-8c0a-4c5b-9e3d-1a2b3c4d5e6f"  # Partial ID from earlier
            found_ram_chunk = False
            
            for i, doc in enumerate(results, 1):
                chunk_id = doc["id"]
                similarity = doc.get("similarity", 0)
                text_preview = doc["text"][:200].replace("\n", " ")
                
                # Check if this is the RAM spec chunk
                if "16GB DDR4" in doc["text"] or "Memory" in doc["text"] and "DDR4" in doc["text"]:
                    found_ram_chunk = True
                    print(f"✓ FOUND RAM SPEC CHUNK:")
                    print(f"  Rank: {i}")
                    print(f"  Chunk ID: {chunk_id[:16]}...")
                    print(f"  Similarity: {similarity:.4f}")
                    print(f"  Text: {doc['text'][:500]}...")
                    print()
                else:
                    print(f"{i}. Similarity: {similarity:.4f} - {text_preview}...")
            
            if not found_ram_chunk:
                print("\n❌ RAM spec chunk NOT found in top results!")
            else:
                print("\n✓ RAM spec chunk found!")
    
    finally:
        db.close()

if __name__ == "__main__":
    test_ram_query()

