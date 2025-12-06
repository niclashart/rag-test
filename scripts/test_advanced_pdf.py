"""Test script for advanced PDF processor."""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_processor_advanced import PDFProcessorAdvanced
from logging_config.logger import get_logger

logger = get_logger(__name__)


def test_advanced_pdf(pdf_path: str, save_outputs: bool = True):
    """Test advanced PDF processor."""
    # Check if available
    if not PDFProcessorAdvanced.is_available():
        print("ERROR: pymupdf4llm is not installed!")
        print("Install with: pip install pymupdf4llm")
        return
    
    print(f"Testing advanced PDF processor on: {pdf_path}")
    print("-" * 60)
    
    # Create output directory
    output_dir = None
    if save_outputs:
        pdf_file = Path(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/pdf_outputs") / f"{pdf_file.stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
    
    # Test with text output
    print("\n1. Processing with TEXT output (headers/footers removed):")
    processor_text = PDFProcessorAdvanced(
        remove_headers_footers=True,
        output_format="text"
    )
    result_text = processor_text.process_pdf(pdf_path)
    print(f"   Pages: {result_text['metadata']['page_count']}")
    print(f"   Text length: {len(result_text['text'])} characters")
    print(f"   First 500 chars:\n{result_text['text'][:500]}...")
    
    # Save text file
    if save_outputs and output_dir:
        text_file = output_dir / f"{Path(pdf_path).stem}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(result_text['text'])
        print(f"   ✓ Saved to: {text_file}")
    
    # Test with markdown output
    print("\n2. Processing with MARKDOWN output:")
    processor_md = PDFProcessorAdvanced(
        remove_headers_footers=True,
        output_format="markdown"
    )
    result_md = processor_md.process_pdf(pdf_path)
    print(f"   Pages: {result_md['metadata']['page_count']}")
    print(f"   Markdown length: {len(result_md['text'])} characters")
    print(f"   First 500 chars:\n{result_md['text'][:500]}...")
    
    # Save markdown file
    if save_outputs and output_dir:
        md_file = output_dir / f"{Path(pdf_path).stem}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(result_md['text'])
        print(f"   ✓ Saved to: {md_file}")
    
    # Test table extraction
    print("\n3. Extracting tables:")
    processor_tables = PDFProcessorAdvanced(remove_headers_footers=True)
    tables = processor_tables.extract_tables(pdf_path)
    print(f"   Found {len(tables)} table(s)")
    for i, table in enumerate(tables):
        print(f"   Table {i+1} on page {table['page_number']}: {table.get('num_rows', 0)} rows")
    
    # Save tables if found
    if save_outputs and output_dir and tables:
        tables_file = output_dir / f"{Path(pdf_path).stem}_tables.md"
        with open(tables_file, 'w', encoding='utf-8') as f:
            f.write(f"# Extracted Tables from {Path(pdf_path).name}\n\n")
            f.write(f"Total tables found: {len(tables)}\n\n")
            for i, table in enumerate(tables, 1):
                f.write(f"## Table {i} (Page {table['page_number']})\n\n")
                f.write(table.get('markdown', 'No markdown available'))
                f.write("\n\n---\n\n")
        print(f"   ✓ Tables saved to: {tables_file}")
    
    # Test structured extraction
    print("\n4. Structured extraction (with tables):")
    pages_data = processor_md.extract_text_with_structure(pdf_path)
    total_tables = sum(page.get('num_tables', 0) for page in pages_data)
    print(f"   Pages: {len(pages_data)}")
    print(f"   Total tables found: {total_tables}")
    for page in pages_data[:3]:  # Show first 3 pages
        print(f"   Page {page['page_number']}: {page['num_tables']} table(s), {len(page['text'])} chars")
    
    if save_outputs and output_dir:
        print(f"\n✓ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test advanced PDF processor")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output files"
    )
    
    args = parser.parse_args()
    
    test_advanced_pdf(args.pdf_path, save_outputs=not args.no_save)

