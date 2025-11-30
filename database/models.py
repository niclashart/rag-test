"""Database models for User, Document, and Chunk."""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    """Document model for uploaded files."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, docx, txt, etc.
    file_size = Column(Integer, nullable=False)  # in bytes
    status = Column(String, default="uploaded")  # uploaded, processing, indexed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    doc_metadata = Column(JSON, default={})  # Additional metadata (page count, etc.)

    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Chunk model for document chunks with position tracking."""
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)  # UUID string
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    bbox = Column(JSON, nullable=True)  # {"x": 0, "y": 0, "width": 100, "height": 50}
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within document
    embedding_id = Column(String, nullable=True)  # Reference to ChromaDB
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class QueryHistory(Base):
    """Query history for user queries."""
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    sources = Column(JSON, default=[])  # List of chunk IDs used
    created_at = Column(DateTime, default=datetime.utcnow)
    query_metadata = Column(JSON, default={})  # Additional metrics, timing, etc.


