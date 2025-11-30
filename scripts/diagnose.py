#!/usr/bin/env python3
"""Diagnose script to check document indexing and search issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.models import Document, Chunk, User
from database.crud import get_user_documents, get_document_chunks
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from sqlalchemy import func

def diagnose():
    """Diagnose document indexing and search issues."""
    db = SessionLocal()
    
    try:
        # Get all users
        users = db.query(User).all()
        print(f"Found {len(users)} users\n")
        
        for user in users:
            print(f"=== User {user.id} ({user.email}) ===")
            
            # Get all documents for this user
            documents = get_user_documents(db, user.id)
            print(f"Documents: {len(documents)}")
            
            for doc in documents:
                print(f"\n  Document ID: {doc.id}")
                print(f"  Filename: {doc.filename}")
                print(f"  Status: {doc.status}")
                print(f"  File path: {doc.file_path}")
                print(f"  File type: {doc.file_type}")
                
                # Count chunks
                chunks = get_document_chunks(db, doc.id)
                print(f"  Chunks in database: {len(chunks)}")
                
                # Check for RAM/memory related chunks
                ram_chunks = [c for c in chunks if "ram" in c.text.lower() or "memory" in c.text.lower() or "speicher" in c.text.lower()]
                print(f"  Chunks containing RAM/memory/speicher: {len(ram_chunks)}")
                
                if ram_chunks:
                    print(f"\n  Sample RAM-related chunks:")
                    for i, chunk in enumerate(ram_chunks[:3], 1):
                        preview = chunk.text[:200].replace("\n", " ")
                        print(f"    {i}. Chunk {chunk.id[:8]}... (page {chunk.page_number}): {preview}...")
                
                # Check ThinkPad E14 related chunks
                e14_chunks = [c for c in chunks if "e14" in c.text.lower() or "thinkpad e14" in c.text.lower()]
                print(f"  Chunks containing 'e14' or 'thinkpad e14': {len(e14_chunks)}")
                
                if e14_chunks:
                    print(f"\n  Sample E14-related chunks:")
                    for i, chunk in enumerate(e14_chunks[:3], 1):
                        preview = chunk.text[:200].replace("\n", " ")
                        print(f"    {i}. Chunk {chunk.id[:8]}... (page {chunk.page_number}): {preview}...")
                
                # Check vector store
                print(f"\n  Checking vector store...")
                vector_store = VectorStore()
                collection = vector_store.get_collection(user.id)
                
                # Count documents in collection
                collection_count = collection.count()
                print(f"  Documents in ChromaDB collection: {collection_count}")
                
                # Try to query for RAM/memory
                if collection_count > 0:
                    embedder = Embedder()
                    test_query = "RAM memory Thinkpad E14"
                    query_embedding = embedder.embed_text(test_query)
                    
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5
                    )
                    
                    if results.get("ids") and len(results["ids"][0]) > 0:
                        print(f"  Test query '{test_query}' returned {len(results['ids'][0])} results")
                        print(f"  Top result IDs: {[id[:8] + '...' for id in results['ids'][0][:3]]}")
                        
                        # Check if any results contain RAM/memory
                        if results.get("documents"):
                            ram_results = [doc for doc in results["documents"][0] if "ram" in doc.lower() or "memory" in doc.lower()]
                            print(f"  Results containing RAM/memory: {len(ram_results)}")
                    else:
                        print(f"  Test query returned no results!")
            
            print("\n" + "="*60 + "\n")
        
        # Test retrieval
        print("=== Testing Retrieval ===")
        if users:
            user = users[0]
            retriever = Retriever(VectorStore(), Embedder())
            
            test_queries = [
                "Wieviel RAM hat das Thinkpad E14?",
                "Wieviel memory hat das Lenovo Thinkpad E14?",
                "RAM Thinkpad E14"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                try:
                    results = retriever.retrieve(user.id, query, n_results=5)
                    print(f"  Retrieved {len(results)} documents")
                    
                    if results:
                        for i, doc in enumerate(results[:3], 1):
                            preview = doc["text"][:150].replace("\n", " ")
                            similarity = doc.get("similarity", "N/A")
                            print(f"    {i}. Similarity: {similarity:.3f} - {preview}...")
                    else:
                        print("  No results found!")
                except Exception as e:
                    print(f"  Error: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    diagnose()

