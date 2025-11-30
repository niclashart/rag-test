#!/usr/bin/env python3
"""Check specific chunks for RAM information."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.models import Document, Chunk
from database.crud import get_user_documents, get_document_chunks

def check_chunks():
    """Check chunks for RAM information."""
    db = SessionLocal()
    
    try:
        documents = db.query(Document).all()
        
        for doc in documents:
            print(f"=== Document: {doc.filename} ===\n")
            chunks = get_document_chunks(db, doc.id)
            
            # Find chunks with RAM/memory information
            ram_keywords = ["ram", "memory", "speicher", "ddr", "gb", "16gb", "32gb", "8gb"]
            
            for chunk in chunks:
                text_lower = chunk.text.lower()
                if any(keyword in text_lower for keyword in ram_keywords):
                    # Check if it also mentions E14
                    if "e14" in text_lower or "thinkpad e14" in text_lower:
                        print(f"Chunk ID: {chunk.id}")
                        print(f"Page: {chunk.page_number}")
                        print(f"Chunk Index: {chunk.chunk_index}")
                        print(f"Text:\n{chunk.text}\n")
                        print("-" * 80 + "\n")
    
    finally:
        db.close()

if __name__ == "__main__":
    check_chunks()

