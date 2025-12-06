"""Advanced PDF processor using pymupdf.layout and pymupdf4llm for better table extraction and layout analysis."""
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Literal
from pathlib import Path
import json
from logging_config.logger import get_logger

logger = get_logger(__name__)

try:
    import pymupdf.layout
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError as e:
    PYMUPDF4LLM_AVAILABLE = False
    logger.warning(f"pymupdf.layout or pymupdf4llm not installed: {e}. Install with: pip install pymupdf-layout pymupdf4llm")


class PDFProcessorAdvanced:
    """Advanced PDF processor with table extraction and header/footer removal."""
    
    def __init__(
        self,
        remove_headers_footers: bool = True,
        output_format: Literal["text", "markdown"] = "markdown"
    ):
        """
        Initialize advanced PDF processor.
        
        Args:
            remove_headers_footers: Whether to automatically remove headers and footers
            output_format: Output format: "text" or "markdown" (markdown recommended for better table preservation)
        """
        if not PYMUPDF4LLM_AVAILABLE:
            raise ImportError(
                "pymupdf.layout or pymupdf4llm is not installed. Install with: pip install pymupdf-layout pymupdf4llm"
            )
        
        self.remove_headers_footers = remove_headers_footers
        self.output_format = output_format
    
    def process_pdf(
        self,
        pdf_path: str,
        output_format: Optional[Literal["text", "markdown"]] = None
    ) -> Dict[str, any]:
        """
        Process PDF with advanced features.
        
        Args:
            pdf_path: Path to PDF file
            output_format: Override default output format ("text" or "markdown")
        
        Returns:
            Dictionary with processed content and metadata
        """
        output_format = output_format or self.output_format
        
        logger.info(f"Processing PDF with advanced processor: {pdf_path}")
        logger.info(f"Output format: {output_format}, Remove headers/footers: {self.remove_headers_footers}")
        
        # Open PDF with pymupdf (pymupdf4llm expects pymupdf.open, not fitz.open)
        import pymupdf
        doc = pymupdf.open(pdf_path)
        
        try:
            # pymupdf4llm processes the entire document at once, not page by page
            # pymupdf.layout is used internally for better table recognition
            if output_format == "markdown":
                full_text = pymupdf4llm.to_markdown(
                    doc,
                    remove_header_footer=self.remove_headers_footers
                )
            else:  # text
                full_text = pymupdf4llm.to_text(
                    doc,
                    remove_header_footer=self.remove_headers_footers
                )
            
            # Split by pages if possible
            pages_data = []
            # Try to split by page markers if they exist
            page_markers = full_text.split("--- Page")
            if len(page_markers) > 1:
                for i, page_content in enumerate(page_markers[1:], 1):
                    pages_data.append({
                        "page_number": i,
                        "text": page_content.strip(),
                        "format": output_format
                    })
            else:
                # If no page markers, create single page entry
                pages_data.append({
                    "page_number": 1,
                    "text": full_text,
                    "format": output_format
                })
            
            logger.info(f"Processed {len(doc)} pages")
        
        finally:
            doc.close()
        
        return {
            "text": full_text,
            "pages": pages_data,
            "metadata": {
                "file_type": "pdf",
                "page_count": len(pages_data),
                "output_format": output_format,
                "headers_footers_removed": self.remove_headers_footers,
                "processor": "advanced_pymupdf4llm"
            }
        }
    
    def extract_text_with_structure(
        self,
        pdf_path: str,
        remove_headers_footers: Optional[bool] = None
    ) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page structure.
        Uses markdown format for better structure preservation (tables, lists, etc.).
        
        Args:
            pdf_path: Path to PDF file
            remove_headers_footers: Override default setting
        
        Returns:
            List of pages with text and metadata
        """
        remove_hf = remove_headers_footers if remove_headers_footers is not None else self.remove_headers_footers
        
        logger.info(f"Extracting text with structure from: {pdf_path}")
        
        import pymupdf
        doc = pymupdf.open(pdf_path)
        pages_data = []
        
        try:
            # Get markdown for the full document (best format for preserving tables)
            full_markdown = pymupdf4llm.to_markdown(
                doc,
                remove_header_footer=remove_hf
            )
            
            # Count tables in markdown
            markdown_lines = full_markdown.split('\n')
            table_count = 0
            in_table = False
            for line in markdown_lines:
                if '|' in line and '---' in line:
                    if not in_table:
                        table_count += 1
                        in_table = True
                elif in_table and '|' not in line and line.strip():
                    in_table = False
            
            # Create single page entry with full markdown
            # (pymupdf4llm processes entire document, page splitting would need additional logic)
            pages_data.append({
                "page_number": 1,
                "text": full_markdown,
                "markdown": full_markdown,
                "num_tables": table_count,
                "metadata": {
                    "total_pages": len(doc),
                    "has_tables": table_count > 0
                }
            })
            
            if table_count > 0:
                logger.info(f"Detected {table_count} table(s) in markdown")
        
        finally:
            doc.close()
        
        return pages_data
    
    def extract_tables(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[Dict[str, any]]:
        """
        Extract tables from PDF using markdown detection.
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Optional list of page numbers (1-indexed). Currently not used as pymupdf4llm processes entire document.
        
        Returns:
            List of tables with their markdown representation
        """
        logger.info(f"Extracting tables from: {pdf_path}")
        
        import pymupdf
        doc = pymupdf.open(pdf_path)
        all_tables = []
        
        try:
            # Get markdown (best format for table extraction)
            full_markdown = pymupdf4llm.to_markdown(
                doc,
                remove_header_footer=self.remove_headers_footers
            )
            
            # Extract tables from markdown
            markdown_lines = full_markdown.split('\n')
            table_count = 0
            current_table_lines = []
            in_table = False
            
            for i, line in enumerate(markdown_lines):
                stripped = line.strip()
                # Simple detection: line with | and --- indicates table separator
                if '|' in stripped and '---' in stripped:
                    if not in_table:
                        # Start new table
                        table_count += 1
                        in_table = True
                        # Include previous line as header if it has |
                        if i > 0 and '|' in markdown_lines[i-1].strip():
                            current_table_lines = [markdown_lines[i-1], line]
                        else:
                            current_table_lines = [line]
                    else:
                        current_table_lines.append(line)
                elif in_table:
                    if '|' in stripped:
                        current_table_lines.append(line)
                    elif stripped:
                        # End of table
                        if len(current_table_lines) >= 2:
                            table_markdown = '\n'.join(current_table_lines)
                            num_rows = len([l for l in current_table_lines if '|' in l.strip() and '---' not in l.strip()])
                            if num_rows > 0:
                                table_data = {
                                    "page_number": 1,  # pymupdf4llm processes entire document
                                    "table_index": table_count - 1,
                                    "markdown": table_markdown,
                                    "detected_from_markdown": True,
                                    "num_rows": num_rows
                                }
                                all_tables.append(table_data)
                                logger.info(f"Detected table {table_count} from markdown ({num_rows} rows)")
                        in_table = False
                        current_table_lines = []
            
            # Handle table at end of document
            if in_table and current_table_lines and len(current_table_lines) >= 2:
                table_markdown = '\n'.join(current_table_lines)
                num_rows = len([l for l in current_table_lines if '|' in l.strip() and '---' not in l.strip()])
                if num_rows > 0:
                    table_data = {
                        "page_number": 1,
                        "table_index": table_count - 1,
                        "markdown": table_markdown,
                        "detected_from_markdown": True,
                        "num_rows": num_rows
                    }
                    all_tables.append(table_data)
                    logger.info(f"Detected table {table_count} from markdown (end, {num_rows} rows)")
        
        finally:
            doc.close()
        
        logger.info(f"Extracted {len(all_tables)} table(s) total")
        return all_tables
    
    def get_page_text(
        self,
        pdf_path: str,
        page_number: int,
        output_format: Optional[Literal["text", "markdown"]] = None
    ) -> Optional[str]:
        """
        Get text from a specific page.
        Note: pymupdf4llm processes entire document, so this returns full document content.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed) - currently returns full document
            output_format: Output format: "text" or "markdown"
        
        Returns:
            Document content in requested format, or None if page doesn't exist
        """
        output_format = output_format or self.output_format
        
        import pymupdf
        doc = pymupdf.open(pdf_path)
        try:
            if page_number < 1 or page_number > len(doc):
                return None
            
            # pymupdf4llm works with entire document
            if output_format == "markdown":
                return pymupdf4llm.to_markdown(
                    doc,
                    remove_header_footer=self.remove_headers_footers
                )
            else:  # text
                return pymupdf4llm.to_text(
                    doc,
                    remove_header_footer=self.remove_headers_footers
                )
        finally:
            doc.close()
    
    @staticmethod
    def is_available() -> bool:
        """Check if pymupdf4llm is available."""
        return PYMUPDF4LLM_AVAILABLE

