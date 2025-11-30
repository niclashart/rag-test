#!/usr/bin/env python3
"""Find Gen 7 PERFORMANCE chunks from the right document."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks

def find_gen7_performance():
    """Find Gen 7 PERFORMANCE chunks."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)
        
        for doc in documents:
            if "Gen_7" not in doc.filename and "Gen 7" not in doc.filename:
                continue
                
            print(f"\n=== Document: {doc.filename} ===")
            chunks = get_document_chunks(db, doc.id)
            
            # Find chunks with PERFORMANCE
            performance_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                if "performance" in text_lower:
                    has_specs = any(kw in text_lower for kw in ["processor", "cpu", "memory", "ram", "storage"])
                    performance_chunks.append((chunk, has_specs))
            
            print(f"Found {len(performance_chunks)} PERFORMANCE chunks\n")
            
            for i, (chunk, has_specs) in enumerate(performance_chunks[:5], 1):
                print(f"{i}. Chunk {chunk.id[:8]}... (Page {chunk.page_number}, Has specs: {has_specs})")
                print(f"   Length: {len(chunk.text)}")
                print(f"   Text: {chunk.text[:400]}...")
                print()
    
    finally:
        db.close()

if __name__ == "__main__":
    find_gen7_performance()

