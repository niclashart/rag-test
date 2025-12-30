#!/usr/bin/env python3
"""Test what the QA chain actually sees and responds."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.rerank.reranker import Reranker
from src.qa.chain import QAChain

query = "Welche Prozessor-Konfigurationen hat das Dell Pro 16 PC16250?"
target_chunk_id = "a62d7f28-4301-47a9-9e42-66868fca5779"

def test_qa():
    print(f"Query: {query}\n")
    print("="*80)
    
    # Initialize components
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    reranker = Reranker()
    qa_chain = QAChain()
    
    # Retrieve documents
    retrieved_docs = retriever.retrieve_with_reranking(
        user_id=1,
        query=query,
        reranker=reranker,
        n_results=15
    )
    
    # Check if target chunk is in retrieved docs
    target_found = False
    target_position = None
    for i, doc in enumerate(retrieved_docs, 1):
        if doc["id"] == target_chunk_id:
            target_found = True
            target_position = i
            break
    
    print(f"\n1. Retrieval-Status:")
    print(f"   Ziel-Chunk gefunden: {'✓ Ja' if target_found else '✗ Nein'}")
    if target_found:
        print(f"   Position: {target_position}")
    
    # Show context that will be sent to LLM
    print(f"\n2. Kontext für LLM (erste 5 Chunks):")
    print("-"*80)
    for i, doc in enumerate(retrieved_docs[:5], 1):
        print(f"\n   Chunk {i} (ID: {doc['id'][:8]}...):")
        print(f"   {'>>> ZIEL-CHUNK <<<' if doc['id'] == target_chunk_id else ''}")
        text_preview = doc['text'][:400].replace('\n', ' ')
        print(f"   {text_preview}...")
    
    # Format context
    context = qa_chain.format_context(retrieved_docs)
    
    # Check if target chunk is in formatted context
    if target_chunk_id in context:
        print(f"\n3. Kontext-Formatierung:")
        print(f"   ✓ Ziel-Chunk ist im formatierten Kontext enthalten")
        
        # Find the chunk in context
        context_parts = context.split("\n---\n")
        for i, part in enumerate(context_parts, 1):
            if target_chunk_id in part:
                print(f"\n   Ziel-Chunk ist Teil {i} von {len(context_parts)}")
                print(f"   Vollständiger Text dieses Chunks:")
                print("-"*80)
                print(part[:2000])  # Show first 2000 chars
                print("-"*80)
                break
    else:
        print(f"\n3. Kontext-Formatierung:")
        print(f"   ✗ Ziel-Chunk ist NICHT im formatierten Kontext!")
    
    # Get answer from LLM
    print(f"\n4. LLM-Antwort:")
    print("-"*80)
    result = qa_chain.answer_with_retrieved_docs(
        question=query,
        retrieved_docs=retrieved_docs,
        chat_history=None
    )
    
    answer = result.get("answer", "")
    print(answer)
    
    # Check if missing processors are mentioned
    print(f"\n5. Prüfung auf fehlende Prozessoren:")
    print("-"*80)
    missing_processors = ["U300E", "i3-1315U", "Core 3 100U", "Core 5 120U"]
    for proc in missing_processors:
        if proc in answer:
            print(f"   ✓ {proc} erwähnt")
        else:
            print(f"   ✗ {proc} NICHT erwähnt")

if __name__ == "__main__":
    test_qa()











