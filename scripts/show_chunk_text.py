#!/usr/bin/env python3
"""Show full text of a specific chunk."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import SessionLocal
from database.models import Chunk, Document

chunk_id = "a62d7f28-4301-47a9-9e42-66868fca5779"

def show_chunk():
    db = SessionLocal()
    try:
        chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
        if chunk:
            doc = db.query(Document).filter(Document.id == chunk.document_id).first()
            print(f"Dokument: {doc.filename if doc else 'Unknown'}")
            print(f"Seite: {chunk.page_number}")
            print(f"Chunk Index: {chunk.chunk_index}")
            print(f"\n{'='*80}")
            print("VOLLSTÄNDIGER CHUNK-TEXT:")
            print('='*80)
            print(chunk.text)
            print('='*80)
            
            # Check for processors
            processors = ["U300E", "i3-1315U", "Core 3 100U", "Core 5 120U"]
            print("\n\nProzessor-Checks:")
            for proc in processors:
                if proc in chunk.text:
                    print(f"✓ {proc} gefunden")
                else:
                    print(f"✗ {proc} NICHT gefunden")
        else:
            print(f"Chunk {chunk_id} nicht gefunden!")
    finally:
        db.close()

if __name__ == "__main__":
    show_chunk()








