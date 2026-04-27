"""PDF processor with OCR support for images."""
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from PIL import Image
import pytesseract
import io
import os
import sys
import shutil
from logging_config.logger import get_logger

logger = get_logger(__name__)

def _find_tesseract_cmd() -> Optional[str]:
    configured_cmd = os.getenv("TESSERACT_CMD")
    if configured_cmd and os.path.exists(configured_cmd):
        return configured_cmd

    tesseract_in_path = shutil.which("tesseract")
    if tesseract_in_path:
        return tesseract_in_path

    possible_paths = []
    if sys.platform.startswith('win'):
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
        ]
    else:
        possible_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
        ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


TESSERACT_CMD = _find_tesseract_cmd()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    logger.info(f"Tesseract found at: {TESSERACT_CMD}")
else:
    logger.warning("Tesseract not found. OCR functionality will be disabled. Install Tesseract-OCR to enable image text extraction.")


class PDFProcessor:
    """Process PDF files with text extraction and OCR for images."""
    
    @staticmethod
    def extract_text_with_structure(pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page structure and bounding boxes.
        Returns list of pages with text and metadata.
        """
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with positions
            blocks = page.get_text("dict")
            text_parts = []
            bboxes = []
            
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    block_bbox = block["bbox"]
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + "\n"
                    
                    if block_text.strip():
                        text_parts.append(block_text.strip())
                        bboxes.append({
                            "x": block_bbox[0],
                            "y": block_bbox[1],
                            "width": block_bbox[2] - block_bbox[0],
                            "height": block_bbox[3] - block_bbox[1]
                        })
            
            # Extract images and perform OCR
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Perform OCR on image
                    ocr_text = pytesseract.image_to_string(image, lang='deu+eng')
                    if ocr_text.strip():
                        # Get image position on page
                        image_rects = page.get_image_rects(xref)
                        if image_rects:
                            rect = image_rects[0]
                            text_parts.append(f"\n[Image {img_index + 1}]:\n{ocr_text}")
                            bboxes.append({
                                "x": rect.x0,
                                "y": rect.y0,
                                "width": rect.width,
                                "height": rect.height
                            })
                        logger.info(f"OCR extracted text from image on page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Failed to process image {img_index} on page {page_num + 1}: {e}")
            
            page_text = "\n\n".join(text_parts)
            
            pages_data.append({
                "page_number": page_num + 1,
                "text": page_text,
                "bboxes": bboxes,
                "metadata": {
                    "width": page.rect.width,
                    "height": page.rect.height
                }
            })
        
        doc.close()
        return pages_data
    
    @staticmethod
    def process_pdf(pdf_path: str) -> Dict[str, any]:
        """
        Process PDF and return structured data.
        Returns dict with full_text, pages, and metadata.
        """
        logger.info(f"Processing PDF: {pdf_path}")
        pages_data = PDFProcessor.extract_text_with_structure(pdf_path)
        
        # Combine all pages into full text
        full_text_parts = []
        for page_data in pages_data:
            full_text_parts.append(f"--- Page {page_data['page_number']} ---\n{page_data['text']}")
        
        full_text = "\n\n".join(full_text_parts)
        
        return {
            "text": full_text,
            "pages": pages_data,
            "metadata": {
                "file_type": "pdf",
                "page_count": len(pages_data),
                "has_images": any(len(page.get("bboxes", [])) > 0 for page in pages_data)
            }
        }
    
    @staticmethod
    def get_page_text(pdf_path: str, page_number: int) -> Optional[str]:
        """Get text from a specific page."""
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            doc.close()
            return None

        page = doc[page_number - 1]
        text = page.get_text()
        doc.close()
        return text

    @staticmethod
    def get_pages_text(pdf_path: str, start_page: int, end_page: int) -> str:
        """Extract text from a range of pages (1-indexed, inclusive).

        Uses OCR on images within the page range for text embedded in images.
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        start = max(1, start_page)
        end = min(total_pages, end_page)

        if start > end:
            doc.close()
            return ""

        pages_text = []
        for page_num in range(start, end + 1):
            page = doc[page_num - 1]

            text_parts = []
            blocks = page.get_text("dict")
            for block in blocks["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                        block_text += "\n"
                    if block_text.strip():
                        text_parts.append(block_text.strip())

            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image, lang='deu+eng')
                    if ocr_text.strip():
                        text_parts.append(f"\n[Image {img_index + 1}]:\n{ocr_text}")
                except Exception as e:
                    logger.warning(f"Failed to OCR image {img_index} on page {page_num}: {e}")

            page_text = "\n\n".join(text_parts)
            if page_text.strip():
                pages_text.append(f"--- Seite {page_num} ---\n{page_text}")

        doc.close()
        return "\n\n".join(pages_text)

