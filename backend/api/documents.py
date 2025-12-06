"""Document management endpoints."""
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import time

from database.database import get_db
from database.models import Document
from database.crud import (
    create_document,
    get_all_documents,
    get_document_by_id,
    update_document_status,
    delete_document,
    create_chunk,
    get_document_chunks
)
from src.ingestion.loader import DocumentLoader
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.pdf_processor_advanced import PDFProcessorAdvanced
from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.index.vector_store import VectorStore
from logging_config.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Initialize components
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "104857600"))  # 100MB

embedder = Embedder()
vector_store = VectorStore()
chunker = Chunker()


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: str
    metadata: dict

    class Config:
        from_attributes = True
        populate_by_name = True


class ChunkResponse(BaseModel):
    id: str
    page_number: int
    text: str
    chunk_index: int
    bbox: Optional[dict]


def document_to_response(document: Document) -> DocumentResponse:
    """Convert Document ORM object to DocumentResponse, mapping doc_metadata to metadata."""
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        created_at=document.created_at.isoformat() if hasattr(document.created_at, 'isoformat') else str(document.created_at),
        metadata=document.doc_metadata if hasattr(document, 'doc_metadata') else {}
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a document."""
    # Check file size
    file_content = await file.read()
    if len(file_content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )
    
    # Create upload directory (no user-specific subdirectories)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Determine file type
    file_type = file_extension[1:].lower() if file_extension else "unknown"
    
    # Create document record
    document = create_document(
        db=db,
        filename=file.filename,
        file_path=str(file_path),
        file_type=file_type,
        file_size=len(file_content)
    )
    
    logger.info(f"Uploaded document {document.id}")
    
    return document_to_response(document)


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    db: Session = Depends(get_db)
):
    """List all documents."""
    documents = get_all_documents(db)
    return [document_to_response(doc) for doc in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific document."""
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document_to_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document_endpoint(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document."""
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete file
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete from database
    delete_document(db, document_id)
    
    logger.info(f"Deleted document {document_id}")
    return None


@router.post("/ingest-all", status_code=status.HTTP_200_OK)
def ingest_all_documents(
    db: Session = Depends(get_db)
):
    """Ingest all non-indexed documents."""
    # Get all non-indexed documents
    all_documents = get_all_documents(db)
    non_indexed_docs = [doc for doc in all_documents if doc.status != "indexed"]
    
    if not non_indexed_docs:
        return {
            "message": "All documents are already indexed",
            "total_documents": len(all_documents),
            "indexed_count": 0
        }
    
    results = {
        "success": [],
        "errors": [],
        "total": len(non_indexed_docs)
    }
    
    for document in non_indexed_docs:
        try:
            if document.status == "indexed":
                continue
            
            update_document_status(db, document_id=document.id, status="processing")
            
            # Load document based on type
            if document.file_type == "pdf":
                processor = PDFProcessorAdvanced(
                    remove_headers_footers=True,
                    output_format="text"
                )
                doc_data = processor.process_pdf(document.file_path)
                pages_data = doc_data.get("pages", [])
                
                # Chunk pages
                chunks = chunker.chunk_pages(pages_data, document.id)
            else:
                # Load other file types
                loader = DocumentLoader()
                doc_data = loader.load_document(document.file_path)
                
                # Chunk text
                chunks = chunker.chunk_text(
                    text=doc_data["text"],
                    document_id=document.id,
                    page_number=1
                )
            
            # Create chunks in database and prepare for embedding
            texts = []
            embeddings_list = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Save chunk to database
                create_chunk(
                    db=db,
                    chunk_id=chunk["id"],
                    document_id=document.id,
                    page_number=chunk["page_number"],
                    text=chunk["text"],
                    chunk_index=chunk["chunk_index"],
                    bbox=chunk.get("bbox")
                )
                
                # Prepare for vector store
                texts.append(chunk["text"])
                metadatas.append({
                    "document_id": document.id,
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"]
                })
                ids.append(chunk["id"])
            
            # Generate embeddings
            embeddings_list = embedder.embed_texts(texts)
            
            # Add to vector store
            vector_store.add_documents(
                texts=texts,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update document status
            update_document_status(db, document.id, "indexed")
            
            results["success"].append({
                "document_id": document.id,
                "filename": document.filename,
                "chunks_count": len(chunks)
            })
            
            logger.info(f"Ingested document {document.id} ({document.filename}) with {len(chunks)} chunks")
        
        except Exception as e:
            update_document_status(db, document.id, "error")
            results["errors"].append({
                "document_id": document.id,
                "filename": document.filename,
                "error": str(e)
            })
            logger.error(f"Error ingesting document {document.id} ({document.filename}): {e}")
    
    return {
        "message": f"Processed {len(non_indexed_docs)} document(s)",
        "success_count": len(results["success"]),
        "error_count": len(results["errors"]),
        "success": results["success"],
        "errors": results["errors"]
    }


@router.post("/{document_id}/ingest", status_code=status.HTTP_200_OK)
def ingest_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Ingest a document into the vector store."""
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if document.status == "indexed":
        return {"message": "Document already indexed", "document_id": document_id}
    
    try:
        update_document_status(db, document_id, "processing")
        
        # Load document based on type
        if document.file_type == "pdf":
            processor = PDFProcessorAdvanced(
                remove_headers_footers=True,
                output_format="text"
            )
            doc_data = processor.process_pdf(document.file_path)
            pages_data = doc_data.get("pages", [])
            
            # Chunk pages
            chunks = chunker.chunk_pages(pages_data, document_id)
        else:
            # Load other file types
            loader = DocumentLoader()
            doc_data = loader.load_document(document.file_path)
            
            # Chunk text
            chunks = chunker.chunk_text(
                text=doc_data["text"],
                document_id=document_id,
                page_number=1
            )
        
        # Create chunks in database and prepare for embedding
        texts = []
        embeddings_list = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Save chunk to database
            create_chunk(
                db=db,
                chunk_id=chunk["id"],
                document_id=document_id,
                page_number=chunk["page_number"],
                text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                bbox=chunk.get("bbox")
            )
            
            # Prepare for vector store
            texts.append(chunk["text"])
            metadatas.append({
                "document_id": document_id,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            })
            ids.append(chunk["id"])
        
        # Generate embeddings
        embeddings_list = embedder.embed_texts(texts)
        
        # Add to vector store
        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update document status
        update_document_status(db, document_id, "indexed")
        
        logger.info(f"Ingested document {document_id} with {len(chunks)} chunks")
        
        return {
            "message": "Document ingested successfully",
            "document_id": document_id,
            "chunks_count": len(chunks)
        }
    
    except Exception as e:
        update_document_status(db, document_id, "error")
        logger.error(f"Error ingesting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document: {str(e)}"
        )


@router.get("/{document_id}/preview")
def preview_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Preview a document (for PDFs, return the PDF file)."""
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if not os.path.exists(document.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    if document.file_type == "pdf":
        return FileResponse(
            document.file_path,
            media_type="application/pdf",
            filename=document.filename
        )
    else:
        # For other file types, return as text
        with open(document.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"content": content, "filename": document.filename}


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
def get_document_chunks_endpoint(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get all chunks for a document."""
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    chunks = get_document_chunks(db, document_id)
    return chunks


@router.get("/{document_id}/chunks/{chunk_id}", response_model=ChunkResponse)
def get_chunk(
    document_id: int,
    chunk_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific chunk."""
    from database.crud import get_chunk_by_id
    
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    chunk = get_chunk_by_id(db, chunk_id)
    if not chunk or chunk.document_id != document_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found"
        )
    
    return chunk


