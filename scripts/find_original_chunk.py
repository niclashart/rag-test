#!/usr/bin/env python3
"""Find the original RAM chunk."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from database.crud import get_document_chunks

def find_original_chunk():
    """Find the original RAM chunk."""
    db = SessionLocal()
    
    try:
        user_id = 1
        
        # Find chunk with "16GB DDR4 (2666MHz)"
        from database.models import Document
        docs = db.query(Document).filter(Document.user_id == user_id).all()
        
        for doc in docs:
            chunks = get_document_chunks(db, doc.id)
            for chunk in chunks:
                if "16GB DDR4" in chunk.text and "2666MHz" in chunk.text:
                    print(f"Found original chunk:")
                    print(f"  Chunk ID: {chunk.id}")
                    print(f"  Page: {chunk.page_number}")
                    print(f"  Text:\n{chunk.text}\n")
                    
                    # Test if this chunk is retrieved
                    retriever = Retriever(VectorStore(), Embedder())
                    results = retriever.retrieve(user_id, "Wieviel RAM hat das Thinkpad E14?", n_results=20)
                    
                    found = False
                    for i, doc_result in enumerate(results, 1):
                        if doc_result["id"] == chunk.id:
                            print(f"✓ Chunk found in retrieval results at rank {i}")
                            print(f"  Similarity: {doc_result.get('similarity', 0):.4f}")
                            found = True
                            break
                    
                    if not found:
                        print("❌ Chunk NOT found in retrieval results!")
    
    finally:
        db.close()

if __name__ == "__main__":
    find_original_chunk()

