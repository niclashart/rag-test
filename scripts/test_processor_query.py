#!/usr/bin/env python3
"""Test if the processor chunk is retrieved for a processor query."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder

query = "Welche Prozessor-Konfigurationen hat das Dell Pro 16 PC16250?"
target_chunk_id = "a62d7f28-4301-47a9-9e42-66868fca5779"

def test_query():
    print(f"Query: {query}\n")
    print(f"Ziel-Chunk ID: {target_chunk_id}\n")
    print("="*80)
    
    # Initialize components
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    
    # Test retrieval without reranking
    print("\n1. Test: Retrieval OHNE Reranking (top_k * 8 = 40 Ergebnisse)")
    print("-"*80)
    retrieved = retriever.retrieve(query, n_results=40)
    
    found = False
    for i, doc in enumerate(retrieved, 1):
        if doc["id"] == target_chunk_id:
            found = True
            print(f"\n✓ Chunk gefunden an Position {i}")
            print(f"  Similarity: {doc.get('similarity', 'N/A')}")
            print(f"  Distance: {doc.get('distance', 'N/A')}")
            print(f"  Text-Ausschnitt (erste 300 Zeichen):")
            print(f"    {doc['text'][:300]}...")
            break
    
    if not found:
        print(f"\n✗ Chunk NICHT in den ersten {len(retrieved)} Ergebnissen gefunden!")
        print(f"\nGefundene Chunks (erste 10):")
        for i, doc in enumerate(retrieved[:10], 1):
            print(f"\n  {i}. Chunk ID: {doc['id']}")
            print(f"     Similarity: {doc.get('similarity', 'N/A')}")
            text_preview = doc['text'][:150].replace('\n', ' ')
            print(f"     Text: {text_preview}...")
    
    # Test with reranking
    print("\n\n2. Test: Retrieval MIT Reranking (rerank_top_k = 15)")
    print("-"*80)
    from src.rerank.reranker import Reranker
    reranker = Reranker()
    
    retrieved_reranked = retriever.retrieve_with_reranking(
        user_id=1,
        query=query,
        reranker=reranker,
        n_results=15
    )
    
    found_reranked = False
    for i, doc in enumerate(retrieved_reranked, 1):
        if doc["id"] == target_chunk_id:
            found_reranked = True
            print(f"\n✓ Chunk gefunden an Position {i} (nach Reranking)")
            print(f"  Similarity: {doc.get('similarity', 'N/A')}")
            break
    
    if not found_reranked:
        print(f"\n✗ Chunk NICHT in den top {len(retrieved_reranked)} Ergebnissen nach Reranking!")
        print(f"\nTop Chunks nach Reranking (erste 10):")
        for i, doc in enumerate(retrieved_reranked[:10], 1):
            print(f"\n  {i}. Chunk ID: {doc['id']}")
            print(f"     Similarity: {doc.get('similarity', 'N/A')}")
            text_preview = doc['text'][:150].replace('\n', ' ')
            print(f"     Text: {text_preview}...")
    
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG:")
    print(f"  Ohne Reranking: {'✓ Gefunden' if found else '✗ Nicht gefunden'}")
    print(f"  Mit Reranking: {'✓ Gefunden' if found_reranked else '✗ Nicht gefunden'}")

if __name__ == "__main__":
    test_query()





