#!/usr/bin/env python3
"""Find chunks with display brightness info."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks

def find_display_brightness():
    """Find chunks with display brightness info."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)
        
        for doc in documents:
            if "E16" not in doc.filename and "Gen_3" not in doc.filename:
                continue
                
            print(f"\n=== Document: {doc.filename} ===\n")
            chunks = get_document_chunks(db, doc.id)
            
            # Find chunks with display/brightness info
            brightness_keywords = ["brightness", "helligkeit", "nits", "cd/m2", "luminance", "display", "bildschirm"]
            brightness_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                if any(kw in text_lower for kw in brightness_keywords):
                    brightness_chunks.append(chunk)
            
            print(f"Display/Brightness chunks: {len(brightness_chunks)}")
            for i, chunk in enumerate(brightness_chunks[:5], 1):
                print(f"\n{i}. Page {chunk.page_number}:")
                print(f"   {chunk.text[:500]}...")
    
    finally:
        db.close()

if __name__ == "__main__":
    find_display_brightness()












