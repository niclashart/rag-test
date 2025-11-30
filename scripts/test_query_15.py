#!/usr/bin/env python3
"""Test query with 15 results."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.qa.chain import QAChain
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder

def test_query():
    """Test query with 15 results."""
    user_id = 1
    query = "Welche Spezifikationen hat das Lenovo Thinkpad E14 Gen 7"
    
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    reranker = Reranker()
    qa_chain = QAChain()
    
    # Test with 15 results
    retrieved = retriever.retrieve_with_reranking(
        user_id=user_id,
        query=query,
        reranker=reranker,
        n_results=15
    )
    
    print(f"Retrieved {len(retrieved)} documents\n")
    
    # Check which chunks have actual specs
    spec_keywords = ["processor", "cpu", "memory", "ram", "storage", "graphics", "gpu", "display", "battery"]
    
    for i, doc in enumerate(retrieved, 1):
        text_lower = doc["text"].lower()
        has_specs = any(keyword in text_lower for keyword in spec_keywords)
        has_gen7 = "gen 7" in text_lower
        similarity = doc.get("similarity", 0)
        
        print(f"{i}. Similarity: {similarity:.4f} | Gen7: {has_gen7} | Has Specs: {has_specs}")
        print(f"   {doc['text'][:200]}...")
        print()
    
    # Test QA chain
    print("\n" + "="*80)
    print("Testing QA Chain with 15 chunks:")
    print("="*80 + "\n")
    
    result = qa_chain.answer_with_retrieved_docs(
        question=query,
        retrieved_docs=retrieved
    )
    
    print(f"Answer: {result.get('answer', 'No answer')}")

if __name__ == "__main__":
    test_query()

