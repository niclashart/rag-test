#!/usr/bin/env python3
"""Check all processor chunks for L16 Gen 2."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.models import Chunk, Document

def check_processor_chunks():
    """Check all processor chunks for L16 Gen 2."""
    db = SessionLocal()
    
    try:
        # Find L16 Gen 2 document (document_id = 3 based on logs)
        doc = db.query(Document).filter(Document.id == 3).first()
        if not doc:
            print("Document ID 3 not found!")
            return
        
        print(f"Document: {doc.filename}")
        print(f"Document ID: {doc.id}\n")
        
        # Find all chunks for this document, ordered by chunk_index
        chunks = db.query(Chunk).filter(Chunk.document_id == 3).order_by(Chunk.chunk_index).all()
        print(f"Total chunks: {len(chunks)}\n")
        
        # Find processor table chunks (chunk_index 1 is the processor table based on logs)
        processor_chunks = []
        for chunk in chunks:
            text_lower = chunk.text.lower()
            # Check if this is a processor-related chunk
            if ("processor" in text_lower and "|" in chunk.text) or \
               ("core ultra" in text_lower and ("225" in chunk.text or "235" in chunk.text or "255" in chunk.text or "265" in chunk.text)):
                processor_chunks.append(chunk)
        
        print(f"Found {len(processor_chunks)} processor table chunks:\n")
        
        for i, chunk in enumerate(processor_chunks, 1):
            print(f"{'='*80}")
            print(f"Chunk {i}: chunk_index={chunk.chunk_index}, chunk_id={chunk.id[:8]}...")
            print(f"Length: {len(chunk.text)} chars")
            print(f"Page: {chunk.page_number}")
            print(f"\nPreview (first 500 chars):")
            print(chunk.text[:500])
            print(f"\n... (last 200 chars):")
            print(chunk.text[-200:])
            
            # Check which processors are in this chunk
            processors = ["225H", "225U", "235H", "235U", "255H", "255U", "265U"]
            found_processors = [p for p in processors if p in chunk.text]
            print(f"\n✓ Found processors: {', '.join(found_processors) if found_processors else 'None'}")
            print()
        
        # Now check if chunk_index 1 (the processor table) contains all processors
        chunk_1 = db.query(Chunk).filter(Chunk.document_id == 3, Chunk.chunk_index == 1).first()
        if chunk_1:
            print(f"\n{'='*80}")
            print(f"Chunk Index 1 (Processor Table):")
            print(f"Length: {len(chunk_1.text)} chars")
            print(f"Chunk ID: {chunk_1.id}")
            
            processors = ["225H", "225U", "235H", "235U", "255H", "255U", "265U"]
            found_processors = [p for p in processors if p in chunk_1.text]
            print(f"\n✓ Found processors in chunk_index 1: {', '.join(found_processors) if found_processors else 'None'}")
            
            if len(found_processors) < 7:
                print(f"\n⚠ WARNING: Only {len(found_processors)} out of 7 processors found in chunk_index 1!")
                print("The processor table might be split across multiple chunks.")
                
                # Check adjacent chunks
                for idx in [0, 2, 3]:
                    adj_chunk = db.query(Chunk).filter(Chunk.document_id == 3, Chunk.chunk_index == idx).first()
                    if adj_chunk:
                        adj_found = [p for p in processors if p in adj_chunk.text]
                        if adj_found:
                            print(f"  → Found processors in chunk_index {idx}: {', '.join(adj_found)}")
    
    finally:
        db.close()

if __name__ == "__main__":
    check_processor_chunks()






