"""CLI script for document ingestion."""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import DocumentLoader
from src.ingestion.pdf_processor import PDFProcessor
from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.index.vector_store import VectorStore
from database.database import SessionLocal, init_db
from database.crud import create_document, create_chunk, get_user_by_id
from logging_config.logger import get_logger

logger = get_logger(__name__)


def ingest_file(file_path: str, user_id: int, db):
    """Ingest a single file."""
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    # Create document record
    document = create_document(
        db=db,
        user_id=user_id,
        filename=file_path_obj.name,
        file_path=str(file_path_obj.absolute()),
        file_type=file_path_obj.suffix[1:].lower(),
        file_size=file_path_obj.stat().st_size
    )
    
    logger.info(f"Processing document {document.id}: {file_path}")
    
    # Initialize components
    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()
    
    try:
        # Load document
        if document.file_type == "pdf":
            processor = PDFProcessor()
            doc_data = processor.process_pdf(str(file_path_obj))
            pages_data = doc_data.get("pages", [])
            chunks = chunker.chunk_pages(pages_data, document.id)
        else:
            loader = DocumentLoader()
            doc_data = loader.load_document(str(file_path_obj))
            chunks = chunker.chunk_text(
                text=doc_data["text"],
                document_id=document.id,
                page_number=1
            )
        
        # Create chunks and prepare for embedding
        texts = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            create_chunk(
                db=db,
                chunk_id=chunk["id"],
                document_id=document.id,
                page_number=chunk["page_number"],
                text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                bbox=chunk.get("bbox")
            )
            
            texts.append(chunk["text"])
            metadatas.append({
                "document_id": document.id,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            })
            ids.append(chunk["id"])
        
        # Generate embeddings and add to vector store
        embeddings = embedder.embed_texts(texts)
        vector_store.add_documents(
            user_id=user_id,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error ingesting {file_path}: {e}")
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG pipeline")
    parser.add_argument("file", help="Path to file to ingest")
    parser.add_argument("--user-id", type=int, required=True, help="User ID")
    parser.add_argument("--init-db", action="store_true", help="Initialize database")
    
    args = parser.parse_args()
    
    # Initialize database if requested
    if args.init_db:
        init_db()
        logger.info("Database initialized")
    
    # Get database session
    db = SessionLocal()
    
    try:
        # Verify user exists
        user = get_user_by_id(db, args.user_id)
        if not user:
            logger.error(f"User {args.user_id} not found")
            sys.exit(1)
        
        # Ingest file
        success = ingest_file(args.file, args.user_id, db)
        
        if success:
            logger.info("Ingestion completed successfully")
            sys.exit(0)
        else:
            logger.error("Ingestion failed")
            sys.exit(1)
    
    finally:
        db.close()


if __name__ == "__main__":
    main()


