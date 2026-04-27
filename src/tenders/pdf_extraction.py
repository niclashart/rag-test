"""PDF chunk extraction for tender documents."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional
from uuid import uuid4

from src.ingestion.pdf_processor import PDFProcessor

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfplumber = None


@dataclass
class TenderChunk:
    chunk_id: str
    text: str
    page_number: int
    section_heading: Optional[str] = None
    table_context: Optional[str] = None
    surrounding_text: Optional[str] = None


class TenderPDFExtractor:
    """Extract page/section chunks without fixed page assumptions."""

    def extract(self, pdf_path: str) -> Dict[str, object]:
        processed = PDFProcessor.process_pdf(pdf_path)
        tables_by_page = self._extract_tables(pdf_path)
        chunks: List[TenderChunk] = []
        active_heading: Optional[str] = None

        for page in processed["pages"]:
            page_text = page.get("text") or ""
            page_number = int(page["page_number"])
            for table_index, table_text in enumerate(tables_by_page.get(page_number, []), start=1):
                chunks.append(
                    TenderChunk(
                        chunk_id=str(uuid4()),
                        text=table_text,
                        page_number=page_number,
                        section_heading=active_heading,
                        table_context=f"pdfplumber_table_{table_index}",
                        surrounding_text=page_text[:1200],
                    )
                )
            sections = self._split_sections(page_text)
            for section in sections:
                heading = self._detect_heading(section) or active_heading
                if heading:
                    active_heading = heading
                chunks.append(
                    TenderChunk(
                        chunk_id=str(uuid4()),
                        text=section,
                        page_number=page_number,
                        section_heading=heading,
                        table_context="table-like" if self._looks_like_table(section) else None,
                        surrounding_text=page_text[:1200],
                    )
                )

        return {
            "raw_text": processed["text"],
            "chunks": chunks,
            "metadata": processed["metadata"],
        }

    def _extract_tables(self, pdf_path: str) -> Dict[int, List[str]]:
        if pdfplumber is None:
            return {}
        tables_by_page: Dict[int, List[str]] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_tables = []
                for table in page.extract_tables() or []:
                    rows = []
                    for row in table:
                        cells = [(cell or "").strip() for cell in row]
                        if any(cells):
                            rows.append(" | ".join(cells))
                    if rows:
                        page_tables.append("\n".join(rows))
                if page_tables:
                    tables_by_page[page_index] = page_tables
        return tables_by_page

    def _split_sections(self, text: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if current and self._is_heading(line):
                parts.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            parts.append("\n".join(current).strip())
        return [part for part in parts if part]

    def _is_heading(self, line: str) -> bool:
        stripped = line.strip()
        if len(stripped) < 4 or len(stripped) > 140:
            return False
        if re.match(r"^\d+(?:\.\d+)*\s+\S+", stripped):
            return True
        return stripped.isupper() or stripped.endswith(":")

    def _detect_heading(self, text: str) -> Optional[str]:
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return first if self._is_heading(first) else None

    def _looks_like_table(self, text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        return sum(1 for line in lines if "\t" in line or "  " in line or "|" in line) >= 2
