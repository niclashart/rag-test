"""Database models for User, Document, and Chunk."""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Boolean, Float
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
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, default=1)  # Optional for compatibility
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
    product_specs = relationship("ProductSpecification", back_populates="document", cascade="all, delete-orphan")
    requirement_specs = relationship("RequirementSpecification", back_populates="document", cascade="all, delete-orphan")


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
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, default=1)  # Optional for compatibility
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    sources = Column(JSON, default=[])  # List of chunk IDs used
    created_at = Column(DateTime, default=datetime.utcnow)
    query_metadata = Column(JSON, default={})  # Additional metrics, timing, etc.


class ProductSpecification(Base):
    """Strukturierte Produktspezifikationen aus Produktdatenblättern."""
    __tablename__ = "product_specifications"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Produkt-Identifikation
    product_name = Column(String, nullable=False, index=True)  # z.B. "ThinkPad E14 Gen 6 AMD"
    brand = Column(String, nullable=True)  # z.B. "Lenovo", "HP"
    model_series = Column(String, nullable=True)  # z.B. "ThinkPad E14"
    generation = Column(String, nullable=True)  # z.B. "Gen 6"
    
    # Display Specs
    display_size_inch = Column(Float, nullable=True)  # z.B. 14.0
    display_brightness_nits = Column(Integer, nullable=True, index=True)  # z.B. 300
    display_resolution = Column(String, nullable=True)  # z.B. "1920x1200"
    screen_to_body_ratio = Column(Float, nullable=True, index=True)  # z.B. 87.5
    
    # Gewicht & Abmessungen
    weight_kg = Column(Float, nullable=True, index=True)  # z.B. 1.43
    width_mm = Column(Float, nullable=True)
    depth_mm = Column(Float, nullable=True)
    height_mm = Column(Float, nullable=True)
    
    # Performance Specs
    ram_max_gb = Column(Integer, nullable=True)  # z.B. 64
    storage_max_tb = Column(Float, nullable=True)  # z.B. 2.0
    battery_wh = Column(Float, nullable=True)  # z.B. 57.0
    
    # Metadaten
    extracted_at = Column(DateTime, default=datetime.utcnow)
    extraction_confidence = Column(Float, nullable=True)  # 0.0-1.0
    raw_specs = Column(JSON, nullable=True)  # Alle extrahierten Rohdaten
    
    # Relationships
    document = relationship("Document", back_populates="product_specs")


class RequirementSpecification(Base):
    """Ausschreibungsanforderungen aus Anforderungsdokumenten."""
    __tablename__ = "requirement_specifications"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Dokumenttyp
    document_type = Column(String, nullable=False, default="requirement")  # "requirement", "product_spec"
    
    # Display Anforderungen
    min_display_brightness_nits = Column(Integer, nullable=True, index=True)
    min_screen_to_body_ratio = Column(Float, nullable=True, index=True)
    display_size_inch_min = Column(Float, nullable=True)
    display_size_inch_max = Column(Float, nullable=True)
    
    # Gewicht & Abmessungen
    max_weight_kg = Column(Float, nullable=True, index=True)
    max_width_mm = Column(Float, nullable=True)
    max_depth_mm = Column(Float, nullable=True)
    max_height_mm = Column(Float, nullable=True)
    
    # Performance Anforderungen
    min_ram_gb = Column(Integer, nullable=True)
    min_storage_tb = Column(Float, nullable=True)
    min_battery_wh = Column(Float, nullable=True)
    
    # Weitere Anforderungen (flexibel als JSON)
    additional_requirements = Column(JSON, nullable=True)  # z.B. {"processor": "Intel Core Ultra5", "wifi": "Wi-Fi 7"}
    
    # Metadaten
    extracted_at = Column(DateTime, default=datetime.utcnow)
    extraction_confidence = Column(Float, nullable=True)  # 0.0-1.0
    raw_requirements = Column(JSON, nullable=True)  # Alle extrahierten Rohdaten
    
    # Relationships
    document = relationship("Document", back_populates="requirement_specs")


class Tender(Base):
    """Uploaded tender document with raw extraction metadata."""
    __tablename__ = "tenders"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")
    raw_text = Column(Text, nullable=True)
    tender_metadata = Column(JSON, default={})

    requirements = relationship("CanonicalRequirement", back_populates="tender", cascade="all, delete-orphan")
    match_results = relationship("TenderMatchResult", back_populates="tender", cascade="all, delete-orphan")


class CanonicalRequirement(Base):
    """Flexible canonical requirement extracted from a tender."""
    __tablename__ = "canonical_requirements"

    id = Column(String, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id"), nullable=False, index=True)
    product_group = Column(String, nullable=True)
    requirement_type = Column(String, nullable=False, default="unknown")
    attribute = Column(String, nullable=False, index=True)
    operator = Column(String, nullable=False)
    value = Column(JSON, nullable=True)
    unit = Column(String, nullable=True)
    original_text = Column(Text, nullable=False)
    source_page = Column(Integer, nullable=True)
    source_chunk_id = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    needs_review = Column(Boolean, default=True)
    status = Column(String, default="pending", index=True)
    points = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tender = relationship("Tender", back_populates="requirements")


class ProductFact(Base):
    """Canonical product fact used for deterministic matching."""
    __tablename__ = "product_facts"

    id = Column(String, primary_key=True, index=True)
    product_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    attribute = Column(String, nullable=False, index=True)
    value = Column(JSON, nullable=True)
    unit = Column(String, nullable=True)
    source_text = Column(Text, nullable=False)
    source_document = Column(String, nullable=False)
    source_page = Column(Integer, nullable=True)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class TenderMatchResult(Base):
    """Persisted match output for a tender/product pair."""
    __tablename__ = "tender_match_results"

    id = Column(String, primary_key=True, index=True)
    tender_id = Column(String, ForeignKey("tenders.id"), nullable=False, index=True)
    product_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    eligibility = Column(String, nullable=False, index=True)
    score = Column(Float, default=0.0)
    max_score = Column(Float, default=0.0)
    must_passed = Column(Integer, default=0)
    must_failed = Column(Integer, default=0)
    unknown_count = Column(Integer, default=0)
    requirement_results = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)

    tender = relationship("Tender", back_populates="match_results")
