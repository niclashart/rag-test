#!/usr/bin/env python3
"""Debug script to check why the graphics table chunk is not being retrieved."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from database.crud import get_chunk_by_id
from database.models import Document

TARGET_CHUNK_ID = "bd9d0fc1-98f4-4ebe-a47c-eed250205951"

def debug_graphics_chunk():
    """Debug why the graphics table chunk is not being retrieved."""
    db = SessionLocal()
    
    try:
        # First, check if chunk exists in database
        chunk = get_chunk_by_id(db, TARGET_CHUNK_ID)
        if not chunk:
            print(f"❌ Chunk {TARGET_CHUNK_ID} NOT FOUND in database!")
            return
        
        doc = db.query(Document).filter(Document.id == chunk.document_id).first()
        print(f"✓ Chunk found in database:")
        print(f"  Chunk ID: {chunk.id}")
        print(f"  Document: {doc.filename if doc else 'Unknown'}")
        print(f"  Page: {chunk.page_number}")
        print(f"  Chunk Index: {chunk.chunk_index}")
        print(f"  Text length: {len(chunk.text)}")
        print(f"\nText preview (first 500 chars):")
        print(chunk.text[:500])
        print("\n" + "="*80)
        
        # Check if chunk is in ChromaDB
        vector_store = VectorStore()
        collection = vector_store.collection
        
        # Try to get the chunk from ChromaDB
        try:
            result = collection.get(ids=[TARGET_CHUNK_ID])
            if result and result.get("ids"):
                print(f"✓ Chunk found in ChromaDB")
                print(f"  Metadata: {result.get('metadatas', [{}])[0] if result.get('metadatas') else 'None'}")
            else:
                print(f"❌ Chunk NOT found in ChromaDB!")
        except Exception as e:
            print(f"❌ Error querying ChromaDB: {e}")
        
        # Now test retrieval with the actual query
        print("\n" + "="*80)
        print("Testing retrieval with query:")
        query = "Welche genauen Modelle gibt es als Optionen für Grafikkarten für das Lenovo Thinkpad L16 Gen 2?"
        print(f"Query: {query}")
        print("="*80)
        
        retriever = Retriever(vector_store, Embedder())
        
        # Extract product and generation from query
        import re
        product_match = re.search(r'thinkpad\s+([a-z]\d{1,2})', query.lower())
        gen_match = re.search(r'gen\s+(\d+)', query.lower())
        
        target_product = product_match.group(1) if product_match else None
        target_gen = int(gen_match.group(1)) if gen_match else None
        
        print(f"Extracted: product={target_product}, gen={target_gen}")
        
        # Retrieve with same parameters as the actual query
        results = retriever.retrieve(
            query=query,
            n_results=400,
            target_product=target_product,
            target_gen=target_gen,
            is_spec_query=True
        )
        
        print(f"\nRetrieved {len(results)} documents after filtering")
        
        # Check if target chunk is in results
        found = False
        for i, doc_result in enumerate(results, 1):
            if doc_result["id"] == TARGET_CHUNK_ID:
                found = True
                print(f"\n✓ TARGET CHUNK FOUND at rank {i}!")
                print(f"  Similarity: {doc_result.get('similarity', 0):.4f}")
                print(f"  Text preview: {doc_result.get('text', '')[:200]}...")
                break
        
        if not found:
            print(f"\n❌ TARGET CHUNK NOT FOUND in retrieval results!")
            print(f"\nFirst 20 retrieved chunks:")
            for i, doc_result in enumerate(results[:20], 1):
                chunk_id_short = doc_result["id"][:8]
                similarity = doc_result.get("similarity", 0)
                text_preview = doc_result.get("text", "")[:100].replace("\n", " ")
                print(f"  {i}. {chunk_id_short}... (sim: {similarity:.4f}) - {text_preview}...")
    
    finally:
        db.close()

if __name__ == "__main__":
    debug_graphics_chunk()



