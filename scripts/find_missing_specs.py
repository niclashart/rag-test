#!/usr/bin/env python3
"""Find chunks with battery and dimensions info."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks

def find_missing_specs():
    """Find chunks with battery and dimensions info."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)
        
        for doc in documents:
            if "Gen_7" not in doc.filename and "Gen 7" not in doc.filename:
                continue
                
            print(f"\n=== Document: {doc.filename} ===\n")
            chunks = get_document_chunks(db, doc.id)
            
            # Find chunks with battery info
            battery_keywords = ["battery", "akku", "batterie", "wh", "watt", "capacity", "life"]
            battery_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                if any(kw in text_lower for kw in battery_keywords):
                    battery_chunks.append(chunk)
            
            print(f"Battery chunks: {len(battery_chunks)}")
            for i, chunk in enumerate(battery_chunks[:3], 1):
                print(f"\n{i}. Page {chunk.page_number}:")
                print(f"   {chunk.text[:400]}...")
            
            # Find chunks with dimensions/weight info
            dim_keywords = ["dimensions", "abmessungen", "weight", "gewicht", "mm", "kg", "lbs", "width", "height", "depth"]
            dim_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                if any(kw in text_lower for kw in dim_keywords):
                    dim_chunks.append(chunk)
            
            print(f"\n\nDimensions/Weight chunks: {len(dim_chunks)}")
            for i, chunk in enumerate(dim_chunks[:3], 1):
                print(f"\n{i}. Page {chunk.page_number}:")
                print(f"   {chunk.text[:400]}...")
    
    finally:
        db.close()

if __name__ == "__main__":
    find_missing_specs()

