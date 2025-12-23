#!/usr/bin/env python3
"""Check for missing processors in the database."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import SessionLocal
from database.models import Chunk, Document
from sqlalchemy import or_

# Missing processors
missing_processors = [
    "U300E",
    "i3-1315U",
    "Core 3 100U",
    "Core 5 120U"
]

def search_processors():
    db = SessionLocal()
    try:
        print("Suche nach fehlenden Prozessoren in der Datenbank...\n")
        
        for processor in missing_processors:
            print(f"\n{'='*60}")
            print(f"Suche nach: {processor}")
            print('='*60)
            
            # Search in chunks
            chunks = db.query(Chunk).filter(
                Chunk.text.ilike(f'%{processor}%')
            ).all()
            
            if chunks:
                print(f"✓ Gefunden in {len(chunks)} Chunk(s):")
                for chunk in chunks[:5]:  # Show first 5
                    doc = db.query(Document).filter(Document.id == chunk.document_id).first()
                print(f"\n  Chunk ID: {chunk.id}")
                print(f"  Dokument: {doc.filename if doc else 'Unknown'}")
                print(f"  Seite: {chunk.page_number}")
                print(f"  Text-Ausschnitt (erste 200 Zeichen):")
                text_preview = chunk.text[:200].replace('\n', ' ')
                print(f"    {text_preview}...")
            else:
                print(f"✗ NICHT gefunden!")
        
        # Also search for PC16250 or Dell Pro 16
        print(f"\n{'='*60}")
        print("Suche nach PC16250 oder Dell Pro 16...")
        print('='*60)
        
        chunks = db.query(Chunk).filter(
            or_(
                Chunk.text.ilike('%PC16250%'),
                Chunk.text.ilike('%Dell Pro 16%')
            )
        ).limit(10).all()
        
        if chunks:
            print(f"✓ Gefunden in {len(chunks)} Chunk(s):")
            for chunk in chunks:
                doc = db.query(Document).filter(Document.id == chunk.document_id).first()
                print(f"\n  Chunk ID: {chunk.id}")
                print(f"  Dokument: {doc.filename if doc else 'Unknown'}")
                print(f"  Seite: {chunk.page_number}")
                # Check if any missing processor is in this chunk
                found_processors = []
                for proc in missing_processors:
                    if proc.lower() in chunk.text.lower():
                        found_processors.append(proc)
                if found_processors:
                    print(f"  ✓ Enthält fehlende Prozessoren: {', '.join(found_processors)}")
                else:
                    print(f"  ✗ Enthält KEINE fehlenden Prozessoren")
                print(f"  Text-Ausschnitt (erste 300 Zeichen):")
                text_preview = chunk.text[:300].replace('\n', ' ')
                print(f"    {text_preview}...")
        else:
            print("✗ Keine Chunks mit PC16250 oder Dell Pro 16 gefunden!")
            
    finally:
        db.close()

if __name__ == "__main__":
    search_processors()

