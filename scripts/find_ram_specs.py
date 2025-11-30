#!/usr/bin/env python3
"""Find chunks with specific RAM specifications."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.crud import get_user_documents, get_document_chunks
import re

def find_ram_specs():
    """Find chunks with RAM specifications."""
    db = SessionLocal()
    
    try:
        documents = get_user_documents(db, 1)  # User 1
        
        for doc in documents:
            print(f"=== Document: {doc.filename} ===\n")
            chunks = get_document_chunks(db, doc.id)
            
            # Patterns for RAM specifications
            ram_patterns = [
                r'\d+\s*gb\s*(?:ram|ddr|memory|speicher)',
                r'(?:ram|ddr|memory|speicher)\s*:\s*\d+\s*gb',
                r'(?:ram|ddr|memory|speicher)\s*up\s*to\s*\d+\s*gb',
                r'\d+\s*gb\s*(?:ddr\d+)',
            ]
            
            print(f"Total chunks: {len(chunks)}\n")
            
            found_specs = False
            for chunk in chunks:
                text = chunk.text
                text_lower = text.lower()
                
                # Check for RAM patterns
                for pattern in ram_patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        print(f"✓ Found RAM spec in Chunk {chunk.id[:8]}... (Page {chunk.page_number}):")
                        print(f"  Text preview: {text[:300]}...")
                        print()
                        found_specs = True
                        break
                
                # Also check for any mention of numbers with GB near memory-related terms
                if re.search(r'\d+\s*gb', text_lower) and any(term in text_lower for term in ['memory', 'ram', 'speicher', 'ddr']):
                    if not found_specs or chunk.id not in [c.id for c in chunks if any(re.search(p, c.text.lower()) for p in ram_patterns)]:
                        print(f"  Possible RAM spec in Chunk {chunk.id[:8]}... (Page {chunk.page_number}):")
                        print(f"  Text: {text[:400]}...")
                        print()
            
            if not found_specs:
                print("❌ No specific RAM specifications found in chunks!")
                print("\nSearching all chunks for any GB mentions...\n")
                
                for chunk in chunks:
                    if re.search(r'\d+\s*gb', chunk.text.lower()):
                        print(f"Chunk {chunk.id[:8]}... (Page {chunk.page_number}):")
                        print(f"{chunk.text[:500]}...")
                        print("-" * 80)
                        print()
    
    finally:
        db.close()

if __name__ == "__main__":
    find_ram_specs()

