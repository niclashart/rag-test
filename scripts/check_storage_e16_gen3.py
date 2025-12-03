#!/usr/bin/env python3
"""Check storage info for E16 Gen 3."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks

def check_storage():
    """Check storage info for E16 Gen 3."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)
        
        for doc in documents:
            if "E16" not in doc.filename or "Gen_3" not in doc.filename:
                continue
                
            print(f"\n=== Document: {doc.filename} ===\n")
            chunks = get_document_chunks(db, doc.id)
            
            # Find chunks with storage info
            storage_keywords = ["storage", "speicher", "ssd", "hdd", "tb", "gb", "capacity", "kapazität"]
            storage_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                if any(kw in text_lower for kw in storage_keywords):
                    # Check if it mentions E16 Gen 3
                    has_e16_gen3 = ("e16" in text_lower and ("gen 3" in text_lower or "gen3" in text_lower))
                    storage_chunks.append((chunk, has_e16_gen3))
            
            print(f"Storage chunks: {len(storage_chunks)}")
            for i, (chunk, has_e16) in enumerate(storage_chunks[:10], 1):
                print(f"\n{i}. Page {chunk.page_number} | E16 Gen 3: {has_e16}")
                print(f"   {chunk.text[:600]}...")
    
    finally:
        db.close()

if __name__ == "__main__":
    check_storage()














