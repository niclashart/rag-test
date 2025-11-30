#!/usr/bin/env python3
"""Test a specific query to see what's retrieved."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from src.retrieval.retriever import Retriever
from src.rerank.reranker import Reranker
from src.qa.chain import QAChain
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from database.models import Document, Chunk
from database.crud import get_user_documents, get_document_chunks

def test_query():
    """Test a specific query."""
    db = SessionLocal()
    
    try:
        user_id = 1
        
        # Check documents
        documents = get_user_documents(db, user_id)
        print(f"=== Documents for user {user_id} ===")
        for doc in documents:
            chunks = get_document_chunks(db, doc.id)
            print(f"Document {doc.id}: {doc.filename}")
            print(f"  Status: {doc.status}")
            print(f"  Chunks: {len(chunks)}")
            
            # Check for E14 Gen 7 mentions
            e14_gen7_chunks = [c for c in chunks if "gen 7" in c.text.lower() or "generation 7" in c.text.lower() or ("e14" in c.text.lower() and "gen" in c.text.lower())]
            print(f"  Chunks with E14 Gen 7: {len(e14_gen7_chunks)}")
            if e14_gen7_chunks:
                print(f"  Sample: {e14_gen7_chunks[0].text[:200]}...")
        
        print("\n" + "="*80)
        print("Testing Query: 'Welche Spezifikationen hat das Lenovo Thinkpad E14 Gen 7'")
        print("="*80 + "\n")
        
        # Initialize components
        vector_store = VectorStore()
        embedder = Embedder()
        retriever = Retriever(vector_store, embedder)
        reranker = Reranker()
        qa_chain = QAChain()
        
        query = "Welche Spezifikationen hat das Lenovo Thinkpad E14 Gen 7"
        
        # Test retrieval
        print("1. Testing retrieval...")
        retrieved_docs = retriever.retrieve(user_id, query, n_results=20)
        print(f"   Retrieved {len(retrieved_docs)} documents\n")
        
        if retrieved_docs:
            print("   Top 10 retrieved chunks:")
            for i, doc in enumerate(retrieved_docs[:10], 1):
                similarity = doc.get("similarity", 0)
                text_preview = doc["text"][:150].replace("\n", " ")
                has_e14 = "e14" in doc["text"].lower()
                has_gen7 = "gen 7" in doc["text"].lower() or "generation 7" in doc["text"].lower()
                print(f"   {i}. Similarity: {similarity:.4f} | E14: {has_e14} | Gen7: {has_gen7}")
                print(f"      {text_preview}...")
                print()
        
        # Test retrieval with reranking
        print("\n2. Testing retrieval with reranking...")
        retrieved_with_rerank = retriever.retrieve_with_reranking(
            user_id=user_id,
            query=query,
            reranker=reranker,
            n_results=8
        )
        print(f"   Retrieved {len(retrieved_with_rerank)} documents after reranking\n")
        
        if retrieved_with_rerank:
            print("   Top 8 reranked chunks:")
            for i, doc in enumerate(retrieved_with_rerank, 1):
                similarity = doc.get("similarity", 0)
                text_preview = doc["text"][:150].replace("\n", " ")
                has_e14 = "e14" in doc["text"].lower()
                has_gen7 = "gen 7" in doc["text"].lower() or "generation 7" in doc["text"].lower()
                print(f"   {i}. Similarity: {similarity:.4f} | E14: {has_e14} | Gen7: {has_gen7}")
                print(f"      {text_preview}...")
                print()
        
        # Test QA chain
        print("\n3. Testing QA chain...")
        if retrieved_with_rerank:
            result = qa_chain.answer_with_retrieved_docs(
                question=query,
                retrieved_docs=retrieved_with_rerank
            )
            print(f"   Answer: {result.get('answer', 'No answer')[:500]}...")
        else:
            print("   No documents to answer from!")
    
    finally:
        db.close()

if __name__ == "__main__":
    test_query()

