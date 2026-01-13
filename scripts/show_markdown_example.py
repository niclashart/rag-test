#!/usr/bin/env python3
"""Skript zum Anzeigen der Markdown-Konvertierung einer PDF."""
import sys
from pathlib import Path

# Pfad zum Projekt-Root hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.pdf_processor_advanced import PDFProcessorAdvanced

def show_markdown_example(pdf_path: str, max_chars: int = 5000):
    """
    Zeigt die Markdown-Konvertierung einer PDF.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        max_chars: Maximale Anzahl Zeichen zum Anzeigen (Standard: 5000)
    """
    print(f"Verarbeite PDF: {pdf_path}\n")
    print("=" * 80)
    
    try:
        # PDF verarbeiten (wie beim Upload)
        processor = PDFProcessorAdvanced(
            remove_headers_footers=True,
            output_format="markdown"
        )
        
        doc_data = processor.process_pdf(pdf_path)
        
        # Metadaten anzeigen
        print("\n📄 Metadaten:")
        print(f"  - Seitenanzahl: {doc_data['metadata']['page_count']}")
        print(f"  - Format: {doc_data['metadata']['output_format']}")
        print(f"  - Header/Footer entfernt: {doc_data['metadata']['headers_footers_removed']}")
        print(f"  - Prozessor: {doc_data['metadata']['processor']}")
        
        # Vollständiger Text (erste max_chars Zeichen)
        print("\n" + "=" * 80)
        print("📝 MARKDOWN-AUSGABE (erste Zeichen):")
        print("=" * 80 + "\n")
        
        full_text = doc_data['text']
        if len(full_text) > max_chars:
            print(full_text[:max_chars])
            print(f"\n... ({len(full_text) - max_chars} weitere Zeichen)")
        else:
            print(full_text)
        
        # Beispiel-Seiten zeigen
        print("\n" + "=" * 80)
        print("📄 BEISPIEL-SEITEN:")
        print("=" * 80 + "\n")
        
        pages_data = doc_data.get('pages', [])
        for i, page in enumerate(pages_data[:3], 1):  # Erste 3 Seiten zeigen
            print(f"\n--- Seite {page['page_number']} ---")
            page_text = page['text']
            if len(page_text) > 1000:
                print(page_text[:1000])
                print(f"\n... ({len(page_text) - 1000} weitere Zeichen auf dieser Seite)")
            else:
                print(page_text)
            print("\n" + "-" * 80)
        
        if len(pages_data) > 3:
            print(f"\n... ({len(pages_data) - 3} weitere Seiten)")
        
        # Option: Vollständige Ausgabe in Datei speichern
        output_file = Path(pdf_path).with_suffix('.md')
        print(f"\n💾 Vollständige Markdown-Ausgabe wird gespeichert in: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"✅ Gespeichert!")
        
    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten der PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Standard: Verwende die aktuell geöffnete PDF
        pdf_path = project_root / "data" / "uploads" / "7afe7621-63e9-4e79-9775-af1eb3f7c1e1.pdf"
        print(f"Kein Pfad angegeben, verwende Standard-PDF: {pdf_path}")
    else:
        pdf_path = sys.argv[1]
    
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    exit(show_markdown_example(str(pdf_path), max_chars))







