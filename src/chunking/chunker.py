"""Text chunking strategies with position tracking."""
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import uuid
import yaml
import os
from logging_config.logger import get_logger

logger = get_logger(__name__)


class Chunker:
    """Chunk text with position tracking for PDF preview."""
    
    def __init__(self, config_path: str = None):
        """Initialize chunker with configuration."""
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "./config/settings.yaml")
        
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                chunking_config = config.get("chunking", {})
        else:
            chunking_config = {}
        
        self.chunk_size = chunking_config.get("chunk_size", 1000)
        self.chunk_overlap = chunking_config.get("chunk_overlap", 200)
        self.separators = chunking_config.get("separators", ["\n\n", "\n", ". ", " ", ""])
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
    def chunk_text(
        self,
        text: str,
        document_id: int,
        page_number: int = 1,
        bboxes: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Chunk text and return list of chunks with metadata.
        For PDFs, bboxes should be provided to track positions.
        """
        # Split text into chunks
        chunks = self.splitter.split_text(text)
        
        chunk_objects = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            
            # Try to find bbox for this chunk (simplified - in production, map more accurately)
            bbox = None
            if bboxes and idx < len(bboxes):
                bbox = bboxes[idx]
            
            chunk_obj = {
                "id": chunk_id,
                "text": chunk_text,
                "chunk_index": idx,
                "page_number": page_number,
                "bbox": bbox,
                "document_id": document_id
            }
            chunk_objects.append(chunk_obj)
        
        logger.info(f"Created {len(chunk_objects)} chunks from text")
        return chunk_objects
    
    def chunk_pages(
        self,
        pages_data: List[Dict],
        document_id: int
    ) -> List[Dict]:
        """
        Chunk multiple pages (e.g., from PDF).
        pages_data should contain: page_number, text, bboxes
        """
        all_chunks = []
        
        for page_data in pages_data:
            page_number = page_data.get("page_number", 1)
            page_text = page_data.get("text", "")
            bboxes = page_data.get("bboxes", [])
            
            page_chunks = self.chunk_text(
                text=page_text,
                document_id=document_id,
                page_number=page_number,
                bboxes=bboxes
            )
            all_chunks.extend(page_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages")
        return all_chunks


