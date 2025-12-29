#!/usr/bin/env python3
"""Debug script to check how many chunks are retrieved for processor queries."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import Retriever
from src.index.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from logging_config.logger import get_logger

logger = get_logger(__name__)

def main():
    query = "Welche Prozessoren unterstützt das Lenovo ThinkPad E14 Gen 6 (Intel)?"
    
    print(f"Query: {query}")
    print("=" * 80)
    
    # Initialize components
    vector_store = VectorStore()
    embedder = Embedder()
    retriever = Retriever(vector_store, embedder)
    
    # Retrieve documents
    print("\nRetrieving documents...")
    retrieved_docs = retriever.retrieve(query=query, n_results=100)
    
    print(f"\nTotal retrieved chunks: {len(retrieved_docs)}")
    print("=" * 80)
    
    # Get document filenames
    from database.database import SessionLocal
    from database.models import Document
    db = SessionLocal()
    doc_id_to_filename = {}
    try:
        documents = db.query(Document).all()
        for doc in documents:
            doc_id_to_filename[doc.id] = doc.filename
    finally:
        db.close()
    
    # Check which chunks contain processor information
    processor_chunks = []
    for i, doc in enumerate(retrieved_docs, 1):
        text = doc.get("text", "").lower()
        chunk_id = doc.get("id", "")
        page = doc.get("metadata", {}).get("page_number", "?")
        document_id = doc.get("metadata", {}).get("document_id")
        filename = doc_id_to_filename.get(document_id, "?") if document_id else "?"
        
        # Check for processor keywords
        has_processor = any(term in text for term in [
            "core ultra 7 165u", "core ultra 5 125u", "core ultra 7 155u",
            "core ultra 5 135u", "core ultra 5 125h", "core ultra 7 155h",
            "processor", "prozessor", "core ultra"
        ])
        
        if has_processor:
            processor_chunks.append((i, chunk_id, page, doc, filename))
            print(f"\nChunk {i} (page {page}, filename: {filename}, chunk_id: {chunk_id[:16]}...):")
            print(f"  Text preview: {doc.get('text', '')[:200]}...")
            
            # Check which processors are mentioned
            processors_found = []
            text_lower = doc.get("text", "").lower()
            if "core ultra 7 165u" in text_lower:
                processors_found.append("Core Ultra 7 165U")
            if "core ultra 5 125u" in text_lower:
                processors_found.append("Core Ultra 5 125U")
            if "core ultra 7 155u" in text_lower:
                processors_found.append("Core Ultra 7 155U")
            if "core ultra 5 135u" in text_lower:
                processors_found.append("Core Ultra 5 135U")
            if "core ultra 5 125h" in text_lower:
                processors_found.append("Core Ultra 5 125H")
            if "core ultra 7 155h" in text_lower:
                processors_found.append("Core Ultra 7 155H")
            
            if processors_found:
                print(f"  Processors found: {', '.join(processors_found)}")
    
    print("\n" + "=" * 80)
    print(f"Total processor-related chunks: {len(processor_chunks)}")
    
    # Show full text of first few processor chunks
    print("\n" + "=" * 80)
    print("Full text of first processor chunk:")
    if processor_chunks:
        first_chunk = processor_chunks[0][3]
        print(first_chunk.get("text", "")[:1000])
    
    if len(processor_chunks) > 1:
        print("\n" + "=" * 80)
        print("Full text of second processor chunk:")
        second_chunk = processor_chunks[1][3]
        print(second_chunk.get("text", "")[:1000])

if __name__ == "__main__":
    main()

