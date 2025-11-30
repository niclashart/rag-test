"""CRUD operations for database models."""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
from .models import User, Document, Chunk, QueryHistory


# User CRUD
def create_user(db: Session, email: str, hashed_password: str) -> User:
    """Create a new user."""
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


# Document CRUD
def create_document(
    db: Session,
    user_id: int,
    filename: str,
    file_path: str,
    file_type: str,
    file_size: int,
    metadata: Optional[dict] = None
) -> Document:
    """Create a new document record."""
    document = Document(
        user_id=user_id,
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        doc_metadata=metadata or {}
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def get_document_by_id(db: Session, document_id: int) -> Optional[Document]:
    """Get document by ID."""
    return db.query(Document).filter(Document.id == document_id).first()


def get_user_documents(db: Session, user_id: int) -> List[Document]:
    """Get all documents for a user."""
    return db.query(Document).filter(Document.user_id == user_id).all()


def update_document_status(db: Session, document_id: int, status: str) -> Optional[Document]:
    """Update document status."""
    document = get_document_by_id(db, document_id)
    if document:
        document.status = status
        db.commit()
        db.refresh(document)
    return document


def delete_document(db: Session, document_id: int) -> bool:
    """Delete a document."""
    document = get_document_by_id(db, document_id)
    if document:
        db.delete(document)
        db.commit()
        return True
    return False


# Chunk CRUD
def create_chunk(
    db: Session,
    chunk_id: str,
    document_id: int,
    page_number: int,
    text: str,
    chunk_index: int,
    bbox: Optional[dict] = None,
    embedding_id: Optional[str] = None
) -> Chunk:
    """Create a new chunk."""
    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        text=text,
        chunk_index=chunk_index,
        bbox=bbox,
        embedding_id=embedding_id
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def get_chunk_by_id(db: Session, chunk_id: str) -> Optional[Chunk]:
    """Get chunk by ID."""
    return db.query(Chunk).filter(Chunk.id == chunk_id).first()


def get_document_chunks(db: Session, document_id: int) -> List[Chunk]:
    """Get all chunks for a document."""
    return db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()


def get_chunks_by_ids(db: Session, chunk_ids: List[str]) -> List[Chunk]:
    """Get chunks by their IDs."""
    return db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()


# Query History CRUD
def create_query_history(
    db: Session,
    user_id: int,
    query: str,
    answer: Optional[str] = None,
    sources: Optional[List[str]] = None,
    metadata: Optional[dict] = None
) -> QueryHistory:
    """Create a query history entry."""
    history = QueryHistory(
        user_id=user_id,
        query=query,
        answer=answer,
        sources=sources or [],
        query_metadata=metadata or {}
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history


def get_user_query_history(db: Session, user_id: int, limit: int = 50) -> List[QueryHistory]:
    """Get query history for a user."""
    return (
        db.query(QueryHistory)
        .filter(QueryHistory.user_id == user_id)
        .order_by(QueryHistory.created_at.desc())
        .limit(limit)
        .all()
    )


