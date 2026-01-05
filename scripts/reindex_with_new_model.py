"""Re-index all documents with a new embedding model."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.database import SessionLocal
from database.crud import get_all_documents, get_document_chunks
from src.embeddings.embedder import Embedder
from src.index.vector_store import VectorStore
from logging_config.logger import get_logger

logger = get_logger(__name__)


def reindex_all_documents(clear_existing: bool = False):
    """Re-index all documents with the current embedding model.
    
    Args:
        clear_existing: If True, delete existing collection before re-indexing
    """
    db = SessionLocal()
    
    try:
        # Initialize with current model from environment
        embedder = Embedder()
        vector_store = VectorStore()
        
        model_name = embedder.model_name
        logger.info(f"Re-indexing all documents with model: {model_name}")
        logger.info(f"Using collection: {vector_store.collection_name}")
        
        # Clear existing collection if requested
        if clear_existing:
            try:
                vector_store.delete_collection()
                logger.info("Deleted existing collection")
            except Exception as e:
                logger.warning(f"Could not delete collection (might not exist): {e}")
        
        # Get all indexed documents
        all_documents = get_all_documents(db)
        indexed_docs = [doc for doc in all_documents if doc.status == "indexed"]
        
        if not indexed_docs:
            logger.warning("No indexed documents found. Nothing to re-index.")
            return
        
        logger.info(f"Found {len(indexed_docs)} documents to re-index")
        
        total_chunks = 0
        successful_docs = 0
        failed_docs = 0
        
        for doc in indexed_docs:
            try:
                logger.info(f"Re-indexing document {doc.id}: {doc.filename}")
                
                # Get all chunks for this document
                chunks = get_document_chunks(db, doc.id)
                
                if not chunks:
                    logger.warning(f"No chunks found for document {doc.id}")
                    continue
                
                # Prepare texts and metadata
                texts = [chunk.text for chunk in chunks]
                metadatas = [{
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                } for chunk in chunks]
                ids = [chunk.id for chunk in chunks]
                
                # Generate new embeddings with current model
                logger.info(f"Generating embeddings for {len(texts)} chunks...")
                embeddings = embedder.embed_texts(texts)
                
                # Add to vector store (will use new collection based on model)
                vector_store.add_documents(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_chunks += len(chunks)
                successful_docs += 1
                logger.info(f"Successfully re-indexed {len(chunks)} chunks for document {doc.id}")
                
            except Exception as e:
                failed_docs += 1
                logger.error(f"Error re-indexing document {doc.id}: {e}")
        
        logger.info("=" * 60)
        logger.info(f"Re-indexing completed!")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Collection: {vector_store.collection_name}")
        logger.info(f"  Successful documents: {successful_docs}")
        logger.info(f"  Failed documents: {failed_docs}")
        logger.info(f"  Total chunks re-indexed: {total_chunks}")
        logger.info("=" * 60)
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-index all documents with current embedding model")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing collection before re-indexing"
    )
    
    args = parser.parse_args()
    
    reindex_all_documents(clear_existing=args.clear_existing)

