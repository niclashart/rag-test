#!/usr/bin/env python3
"""Find chunks with actual specifications."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks

def find_spec_chunks():
    """Find chunks with actual specifications."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)
        
        for doc in documents:
            print(f"\n=== {doc.filename} ===")
            chunks = get_document_chunks(db, doc.id)
            print(f"Total chunks: {len(chunks)}")
            
            # Check if this is Gen 7 document by checking chunks
            has_gen7 = any("gen 7" in chunk.text.lower() or "generation 7" in chunk.text.lower() for chunk in chunks[:10])
            if not has_gen7:
                print("Skipping (not Gen 7)")
                continue
            
            print("Found Gen 7 document!")
            
            # Find chunks with actual specs (not just titles)
            spec_keywords = ["processor", "cpu", "memory", "ram", "storage", "graphics", "gpu", "display", "battery", "prozessor", "speicher", "grafik", "bildschirm", "akku"]
            
            spec_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                # Skip chunks that are just titles/headers
                if len(chunk.text) < 200:
                    continue
                # Check if chunk contains spec keywords
                if any(keyword in text_lower for keyword in spec_keywords):
                    spec_chunks.append(chunk)
            
            print(f"Chunks with specifications: {len(spec_chunks)}")
            
            # Show first few spec chunks
            for i, chunk in enumerate(spec_chunks[:5], 1):
                print(f"\n{i}. Chunk {chunk.id[:8]}... (Page {chunk.page_number}):")
                print(f"   {chunk.text[:500]}...")
    
    finally:
        db.close()

if __name__ == "__main__":
    find_spec_chunks()

