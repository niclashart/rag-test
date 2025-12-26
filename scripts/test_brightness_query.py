#!/usr/bin/env python3
"""Test brightness query retrieval."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.rerank.reranker import Reranker
from src.qa.chain import QAChain

query1 = "Welche technischen Spezifikationen hat das Dell Pro 16 PC16250?"
query2 = "Wie hoch ist die Display-Helligkeit (in nits) des Dell Pro 16 PC16250?"

def test_queries():
    print("="*80)
    print("TEST 1: Allgemeine Spezifikationsfrage")
    print("="*80)
    print(f"Query: {query1}\n")
    
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    
    retrieved1 = retriever.retrieve(query1, n_results=50)
    
    # Suche nach Chunks mit Helligkeit
    brightness_chunks = []
    for doc in retrieved1:
        text_lower = doc.get("text", "").lower()
        if any(kw in text_lower for kw in ["nits", "brightness", "helligkeit", "300"]):
            brightness_chunks.append(doc)
    
    print(f"Gefundene Chunks: {len(retrieved1)}")
    print(f"Chunks mit Helligkeit: {len(brightness_chunks)}")
    
    if brightness_chunks:
        print("\nHelligkeits-Chunks gefunden:")
        for i, doc in enumerate(brightness_chunks[:3], 1):
            print(f"\n{i}. Chunk ID: {doc['id'][:8]}...")
            print(f"   Similarity: {doc.get('similarity', 'N/A')}")
            text_preview = doc['text'][:300].replace('\n', ' ')
            print(f"   Text: {text_preview}...")
    
    print("\n" + "="*80)
    print("TEST 2: Spezifische Helligkeitsfrage")
    print("="*80)
    print(f"Query: {query2}\n")
    
    retrieved2 = retriever.retrieve(query2, n_results=50)
    
    # Suche nach Chunks mit Helligkeit
    brightness_chunks2 = []
    for doc in retrieved2:
        text_lower = doc.get("text", "").lower()
        if any(kw in text_lower for kw in ["nits", "brightness", "helligkeit", "300"]):
            brightness_chunks2.append(doc)
    
    print(f"Gefundene Chunks: {len(retrieved2)}")
    print(f"Chunks mit Helligkeit: {len(brightness_chunks2)}")
    
    if brightness_chunks2:
        print("\nHelligkeits-Chunks gefunden:")
        for i, doc in enumerate(brightness_chunks2[:3], 1):
            print(f"\n{i}. Chunk ID: {doc['id'][:8]}...")
            print(f"   Similarity: {doc.get('similarity', 'N/A')}")
            text_preview = doc['text'][:300].replace('\n', ' ')
            print(f"   Text: {text_preview}...")
    else:
        print("\n⚠️ KEINE Helligkeits-Chunks gefunden!")
        print("\nTop 5 Chunks:")
        for i, doc in enumerate(retrieved2[:5], 1):
            print(f"\n{i}. Chunk ID: {doc['id'][:8]}...")
            print(f"   Similarity: {doc.get('similarity', 'N/A')}")
            text_preview = doc['text'][:200].replace('\n', ' ')
            print(f"   Text: {text_preview}...")
    
    # Test mit Reranking
    print("\n" + "="*80)
    print("TEST 3: Spezifische Helligkeitsfrage MIT Reranking")
    print("="*80)
    
    reranker = Reranker()
    retrieved3 = retriever.retrieve_with_reranking(
        user_id=1,
        query=query2,
        reranker=reranker,
        n_results=15
    )
    
    brightness_chunks3 = []
    for doc in retrieved3:
        text_lower = doc.get("text", "").lower()
        if any(kw in text_lower for kw in ["nits", "brightness", "helligkeit", "300"]):
            brightness_chunks3.append(doc)
    
    print(f"Gefundene Chunks (nach Reranking): {len(retrieved3)}")
    print(f"Chunks mit Helligkeit: {len(brightness_chunks3)}")
    
    if brightness_chunks3:
        print("\nHelligkeits-Chunks gefunden:")
        for i, doc in enumerate(brightness_chunks3[:3], 1):
            print(f"\n{i}. Chunk ID: {doc['id'][:8]}...")
            print(f"   Similarity: {doc.get('similarity', 'N/A')}")
            text_preview = doc['text'][:300].replace('\n', ' ')
            print(f"   Text: {text_preview}...")
    else:
        print("\n⚠️ KEINE Helligkeits-Chunks gefunden (nach Reranking)!")

if __name__ == "__main__":
    test_queries()





