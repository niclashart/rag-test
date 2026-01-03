#!/usr/bin/env python3
"""Debug script to test Screen-to-Body Ratio retrieval."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder

query = "Wie ist die Screen-to-Body Ratio des Lenovo Thinkpad L14 Gen 5 (Intel)?"

def test_screen_to_body_retrieval():
    print(f"Query: {query}\n")
    print("="*80)
    
    # Initialize components
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    
    # Test retrieval (as used in query.py for screen-to-body queries)
    print("\n1. Testing retrieval (n_results=80 as configured for screen-to-body queries)...")
    print("-"*80)
    retrieved_docs = retriever.retrieve(query=query, n_results=80)
    print(f"Retrieved {len(retrieved_docs)} documents\n")
    
    # Search for chunks with Screen-to-Body Ratio information
    screen_to_body_chunks = []
    display_chunks_with_percent = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        text_lower = doc.get("text", "").lower()
        text = doc.get("text", "")
        
        # Check for explicit Screen-to-Body Ratio mention
        has_screen_to_body = "screen-to-body" in text_lower or "screen to body" in text_lower
        has_percent = "%" in text
        has_display = "display" in text_lower or "screen" in text_lower
        
        # Check for percentage values
        has_85_5 = "85.5%" in text or "85,5%" in text
        has_85 = "85%" in text
        
        if has_screen_to_body:
            screen_to_body_chunks.append((i, doc, "explicit"))
        elif has_display and has_percent:
            if has_85_5 or has_85:
                display_chunks_with_percent.append((i, doc, "display_with_percent"))
    
    print(f"Found {len(screen_to_body_chunks)} chunks with explicit 'Screen-to-Body Ratio'")
    print(f"Found {len(display_chunks_with_percent)} display chunks with percentage\n")
    
    # Show explicit Screen-to-Body Ratio chunks
    if screen_to_body_chunks:
        print("Chunks with explicit 'Screen-to-Body Ratio':")
        print("-"*80)
        for rank, doc, reason in screen_to_body_chunks[:5]:
            similarity = doc.get("similarity", 0)
            chunk_id = doc.get("id", "")[:16]
            text = doc.get("text", "")
            
            # Find the line with Screen-to-Body Ratio
            lines = text.split('\n')
            relevant_lines = []
            for line in lines:
                if "screen-to-body" in line.lower() or "screen to body" in line.lower():
                    relevant_lines.append(line.strip())
            
            print(f"\nRank {rank} (ID: {chunk_id}..., Similarity: {similarity:.4f})")
            if relevant_lines:
                print(f"  Relevant line: {relevant_lines[0]}")
            else:
                print(f"  Text preview: {text[:300]}...")
    else:
        print("⚠️ NO chunks with explicit 'Screen-to-Body Ratio' found!\n")
    
    # Show display chunks with percentage
    if display_chunks_with_percent:
        print("\nDisplay chunks with percentage:")
        print("-"*80)
        for rank, doc, reason in display_chunks_with_percent[:5]:
            similarity = doc.get("similarity", 0)
            chunk_id = doc.get("id", "")[:16]
            text = doc.get("text", "")
            
            # Find lines with percentage
            lines = text.split('\n')
            relevant_lines = []
            for line in lines:
                if "%" in line and ("85" in line or "86" in line or "87" in line or "88" in line or "89" in line or "90" in line):
                    relevant_lines.append(line.strip())
            
            print(f"\nRank {rank} (ID: {chunk_id}..., Similarity: {similarity:.4f})")
            if relevant_lines:
                for line in relevant_lines[:3]:
                    print(f"  {line}")
            else:
                print(f"  Text preview: {text[:300]}...")
    
    # Show top 10 chunks overall
    print("\n\nTop 10 retrieved chunks:")
    print("-"*80)
    for i, doc in enumerate(retrieved_docs[:10], 1):
        similarity = doc.get("similarity", 0)
        chunk_id = doc.get("id", "")[:16]
        text = doc.get("text", "")
        text_lower = text.lower()
        
        has_screen_to_body = "screen-to-body" in text_lower or "screen to body" in text_lower
        has_display = "display" in text_lower or "screen" in text_lower
        has_percent = "%" in text
        has_l14 = "l14" in text_lower
        has_gen5 = "gen 5" in text_lower or "generation 5" in text_lower
        
        print(f"\n{i}. Rank {i} (ID: {chunk_id}..., Similarity: {similarity:.4f})")
        print(f"   L14: {has_l14}, Gen5: {has_gen5}, Display: {has_display}, Percent: {has_percent}, Screen-to-Body: {has_screen_to_body}")
        print(f"   Preview: {text[:200].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    test_screen_to_body_retrieval()

