#!/usr/bin/env python3
"""Find E14 Gen 7 PERFORMANCE chunks."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks

def find_e14_gen7_performance():
    """Find E14 Gen 7 PERFORMANCE chunks."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)
        
        for doc in documents:
            chunks = get_document_chunks(db, doc.id)
            
            # Find chunks with both E14 Gen 7 and PERFORMANCE
            for chunk in chunks:
                text_lower = chunk.text.lower()
                has_e14_gen7 = ("e14" in text_lower and ("gen 7" in text_lower or "generation 7" in text_lower))
                has_performance = "performance" in text_lower
                has_specs = any(kw in text_lower for kw in ["processor", "cpu", "memory", "ram"])
                
                if has_e14_gen7 and has_performance:
                    print(f"\n=== Found E14 Gen 7 PERFORMANCE chunk ===")
                    print(f"Document: {doc.filename}")
                    print(f"Chunk ID: {chunk.id}")
                    print(f"Page: {chunk.page_number}")
                    print(f"Has specs: {has_specs}")
                    print(f"Text length: {len(chunk.text)}")
                    print(f"\nText:\n{chunk.text[:800]}...")
                    print()
    
    finally:
        db.close()

if __name__ == "__main__":
    find_e14_gen7_performance()

