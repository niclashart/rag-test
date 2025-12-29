#!/usr/bin/env python3
"""Find all processor chunks for L16 Gen 2."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from database.models import Chunk, Document
from sqlalchemy import or_

def find_processor_chunks():
    """Find all processor chunks for L16 Gen 2."""
    db = SessionLocal()
    
    try:
        # Find L16 Gen 2 document
        documents = db.query(Document).filter(
            or_(
                Document.filename.ilike('%l16%gen%2%'),
                Document.filename.ilike('%l16%gen2%'),
                Document.filename.ilike('%thinkpad%l16%')
            )
        ).all()
        
        print(f"Found {len(documents)} potential L16 Gen 2 document(s)\n")
        
        for doc in documents:
            print(f"="*80)
            print(f"Document: {doc.filename}")
            print(f"Document ID: {doc.id}")
            print("="*80)
            
            # Find all chunks for this document
            chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).order_by(Chunk.chunk_index).all()
            print(f"Total chunks: {len(chunks)}\n")
            
            # Find processor-related chunks
            processor_keywords = [
                "processor", "cpu", "core ultra", "intel core", 
                "225h", "225u", "235h", "235u", "245h", "245u",
                "p-core", "e-core", "lp e-core"
            ]
            
            processor_chunks = []
            for chunk in chunks:
                text_lower = chunk.text.lower()
                # Check if chunk contains processor information
                if any(keyword in text_lower for keyword in processor_keywords):
                    # Also check if it mentions L16 Gen 2
                    has_l16_gen2 = (
                        "l16" in text_lower and 
                        ("gen 2" in text_lower or "gen2" in text_lower or "generation 2" in text_lower)
                    )
                    
                    processor_chunks.append({
                        "chunk": chunk,
                        "has_l16_gen2": has_l16_gen2
                    })
            
            print(f"Found {len(processor_chunks)} processor-related chunks:\n")
            
            for i, pc in enumerate(processor_chunks, 1):
                chunk = pc["chunk"]
                print(f"{i}. Chunk ID: {chunk.id}")
                print(f"   Page: {chunk.page_number}, Chunk Index: {chunk.chunk_index}")
                print(f"   Length: {len(chunk.text)} chars")
                print(f"   Has L16 Gen 2: {pc['has_l16_gen2']}")
                print(f"   Preview: {chunk.text[:200].replace(chr(10), ' ')}...")
                
                # Check for specific processors
                processors_found = []
                for proc in ["225H", "225U", "235H", "235U", "245H", "245U", "Core Ultra 5", "Core Ultra 7"]:
                    if proc in chunk.text:
                        processors_found.append(proc)
                
                if processors_found:
                    print(f"   ✓ Contains processors: {', '.join(processors_found)}")
                print()
            
            # Now test retrieval
            print("\n" + "="*80)
            print("Testing retrieval with query:")
            print("="*80)
            
            from src.retrieval.retriever import Retriever
            from src.index.vector_store import VectorStore
            from src.embeddings.embedder import Embedder
            
            retriever = Retriever(VectorStore(), Embedder())
            query = "Welche Prozessoren unterstützt das Lenovo Thinkpad L16 Gen 2?"
            
            results = retriever.retrieve(
                query=query,
                n_results=400,
                target_product="l16",
                target_gen=2,
                is_spec_query=True
            )
            
            print(f"\nRetrieved {len(results)} documents after filtering\n")
            
            # Check which processor chunks were found
            found_chunk_ids = {r["id"] for r in results}
            
            print("Processor chunks found in retrieval:")
            for i, pc in enumerate(processor_chunks, 1):
                chunk_id = pc["chunk"].id
                if chunk_id in found_chunk_ids:
                    rank = next((j for j, r in enumerate(results, 1) if r["id"] == chunk_id), None)
                    print(f"  ✓ Chunk {i} (ID: {chunk_id[:8]}...) found at rank {rank}")
                else:
                    print(f"  ✗ Chunk {i} (ID: {chunk_id[:8]}...) NOT found")
    
    finally:
        db.close()

if __name__ == "__main__":
    find_processor_chunks()





