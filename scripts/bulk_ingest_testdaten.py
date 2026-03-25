"""Bulk-Ingest-Script für Laptop-Datenblätter und Leistungsverzeichnis.

Verarbeitet alle PDFs aus ./testdaten (Produktdatenblätter) und
./checkliste (Leistungsverzeichnis/Anforderungen) und überführt sie
in die Vektordatenbank + extrahiert strukturierte Spezifikationen.

Verwendung:
    python scripts/bulk_ingest_testdaten.py
    python scripts/bulk_ingest_testdaten.py --skip-existing
    python scripts/bulk_ingest_testdaten.py --only-requirements
"""
import sys
import argparse
from pathlib import Path

# Projektroot zum Python-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_processor_advanced import PDFProcessorAdvanced
from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.index.vector_store import VectorStore
from src.extraction.spec_extractor import SpecExtractor
from database.database import SessionLocal, init_db
from database.crud import create_document, create_chunk, get_all_documents, update_document_status
from database.models import ProductSpecification, RequirementSpecification
from logging_config.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).parent.parent
TESTDATEN_DIR = ROOT_DIR / "testdaten"
CHECKLISTE_DIR = ROOT_DIR / "checkliste"


def get_existing_filenames(db) -> set:
    """Gibt alle bereits indizierten Dateinamen zurück."""
    all_docs = get_all_documents(db)
    return {doc.filename for doc in all_docs if doc.status == "indexed"}


def ingest_pdf(file_path: Path, db, chunker, embedder, vector_store, spec_extractor) -> bool:
    """Verarbeitet eine einzelne PDF-Datei vollständig.

    Returns:
        True bei Erfolg, False bei Fehler.
    """
    print(f"\n  📄 Verarbeite: {file_path.name}")

    # 1. Dokument in DB anlegen
    document = create_document(
        db=db,
        filename=file_path.name,
        file_path=str(file_path.absolute()),
        file_type="pdf",
        file_size=file_path.stat().st_size
    )
    update_document_status(db, document.id, "processing")

    try:
        # 2. PDF verarbeiten (Markdown-Format für bessere Tabellenstruktur)
        processor = PDFProcessorAdvanced(
            remove_headers_footers=True,
            output_format="markdown"
        )
        doc_data = processor.process_pdf(str(file_path))
        pages_data = doc_data.get("pages", [])

        if not pages_data:
            print(f"  ⚠️  Keine Seiten extrahiert aus {file_path.name}")
            update_document_status(db, document.id, "error")
            return False

        # 3. Text chunken
        chunks = chunker.chunk_pages(pages_data, document.id)
        print(f"  ✂️  {len(chunks)} Chunks erstellt")

        if not chunks:
            print(f"  ⚠️  Keine Chunks erstellt aus {file_path.name}")
            update_document_status(db, document.id, "error")
            return False

        # 4. Chunks in DB speichern + für Embedding vorbereiten
        texts, metadatas, ids = [], [], []
        for chunk in chunks:
            create_chunk(
                db=db,
                chunk_id=chunk["id"],
                document_id=document.id,
                page_number=chunk["page_number"],
                text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                bbox=chunk.get("bbox")
            )
            texts.append(chunk["text"])
            metadatas.append({
                "document_id": document.id,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"]
            })
            ids.append(chunk["id"])

        # 5. Embeddings generieren und in ChromaDB speichern
        embeddings = embedder.embed_texts(texts)
        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  🔢  Embeddings gespeichert in ChromaDB")

        # 6. Dokument als indiziert markieren
        update_document_status(db, document.id, "indexed")

        # 7. Strukturierte Spezifikationen mit GPT extrahieren
        full_text = "\n\n".join([page.get("text", "") for page in pages_data])
        extracted_specs = spec_extractor.extract_from_document(full_text, file_path.name)

        if extracted_specs.get("document_type") == "requirement":
            req_spec = RequirementSpecification(
                document_id=document.id,
                min_display_brightness_nits=extracted_specs.get("min_display_brightness_nits"),
                min_screen_to_body_ratio=extracted_specs.get("min_screen_to_body_ratio"),
                max_weight_kg=extracted_specs.get("max_weight_kg"),
                min_ram_gb=extracted_specs.get("min_ram_gb"),
                min_storage_tb=extracted_specs.get("min_storage_tb"),
                min_battery_wh=extracted_specs.get("min_battery_wh"),
                display_size_inch_min=extracted_specs.get("display_size_inch_min"),
                display_size_inch_max=extracted_specs.get("display_size_inch_max"),
                additional_requirements=extracted_specs.get("additional_requirements"),
                raw_requirements=extracted_specs
            )
            db.add(req_spec)
            db.commit()
            print(f"  📋  Anforderungen extrahiert und gespeichert (Requirement-Dokument)")
        else:
            prod_spec = ProductSpecification(
                document_id=document.id,
                product_name=extracted_specs.get("product_name", file_path.stem),
                brand=extracted_specs.get("brand"),
                display_brightness_nits=extracted_specs.get("display_brightness_nits"),
                screen_to_body_ratio=extracted_specs.get("screen_to_body_ratio"),
                weight_kg=extracted_specs.get("weight_kg"),
                ram_max_gb=extracted_specs.get("ram_max_gb"),
                storage_max_tb=extracted_specs.get("storage_max_tb"),
                battery_wh=extracted_specs.get("battery_wh"),
                display_size_inch=extracted_specs.get("display_size_inch"),
                display_resolution=extracted_specs.get("display_resolution"),
                raw_specs=extracted_specs
            )
            db.add(prod_spec)
            db.commit()
            print(f"  💻  Produktspezifikationen extrahiert: {extracted_specs.get('product_name', 'Unbekannt')}")

        return True

    except Exception as e:
        logger.error(f"Fehler bei {file_path.name}: {e}", exc_info=True)
        print(f"  ❌  Fehler: {e}")
        update_document_status(db, document.id, "error")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-Ingest von Laptop-Datenblättern und Leistungsverzeichnis"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Bereits indizierte Dateien überspringen (Standard: aktiviert)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Alle Dateien neu indizieren (auch bereits vorhandene)"
    )
    parser.add_argument(
        "--only-requirements",
        action="store_true",
        help="Nur das Leistungsverzeichnis aus ./checkliste verarbeiten"
    )
    parser.add_argument(
        "--only-products",
        action="store_true",
        help="Nur Produktdatenblätter aus ./testdaten verarbeiten"
    )
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing

    # Datenbank initialisieren
    print("🔧 Initialisiere Datenbank...")
    init_db()
    db = SessionLocal()

    # Shared-Komponenten einmalig initialisieren (spart Ladezeit)
    print("🔧 Lade Modelle (Embedder, VectorStore)...")
    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()
    spec_extractor = SpecExtractor()
    print("✅ Modelle geladen\n")

    # Bereits vorhandene Dateien ermitteln
    existing_filenames = get_existing_filenames(db) if skip_existing else set()
    if skip_existing and existing_filenames:
        print(f"ℹ️  {len(existing_filenames)} bereits indizierte Dokumente werden übersprungen\n")

    # Dateien sammeln
    pdf_files = []

    if not args.only_requirements:
        product_pdfs = sorted(TESTDATEN_DIR.glob("*.pdf"))
        if not product_pdfs:
            print(f"⚠️  Keine PDFs in {TESTDATEN_DIR} gefunden")
        else:
            print(f"📁 {len(product_pdfs)} Produktdatenblätter in ./testdaten gefunden")
            pdf_files.extend(("product", p) for p in product_pdfs)

    if not args.only_products:
        requirement_pdfs = sorted(CHECKLISTE_DIR.glob("*.pdf"))
        if not requirement_pdfs:
            print(f"⚠️  Keine PDFs in {CHECKLISTE_DIR} gefunden")
        else:
            print(f"📁 {len(requirement_pdfs)} Anforderungsdokument(e) in ./checkliste gefunden")
            pdf_files.extend(("requirement", p) for p in requirement_pdfs)

    if not pdf_files:
        print("❌ Keine Dateien zum Verarbeiten gefunden.")
        db.close()
        return

    # Bereits indizierte herausfiltern
    to_process = []
    skipped = []
    for doc_type, path in pdf_files:
        if skip_existing and path.name in existing_filenames:
            skipped.append(path.name)
        else:
            to_process.append((doc_type, path))

    if skipped:
        print(f"\n⏭️  Überspringe {len(skipped)} bereits indizierte Dateien")

    if not to_process:
        print("\n✅ Alle Dateien sind bereits indiziert.")
        db.close()
        return

    print(f"\n🚀 Starte Verarbeitung von {len(to_process)} Dateien...\n")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for idx, (doc_type, file_path) in enumerate(to_process, 1):
        label = "📋 Anforderung" if doc_type == "requirement" else "💻 Produkt"
        print(f"[{idx}/{len(to_process)}] {label}: {file_path.name}")

        success = ingest_pdf(file_path, db, chunker, embedder, vector_store, spec_extractor)

        if success:
            success_count += 1
            print(f"  ✅ Erfolgreich")
        else:
            error_count += 1

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"  Verarbeitet:  {len(to_process)} Dateien")
    print(f"  ✅ Erfolgreich: {success_count}")
    print(f"  ❌ Fehler:      {error_count}")
    if skipped:
        print(f"  ⏭️  Übersprungen: {len(skipped)}")

    # Produkte und Anforderungen zählen
    from database.models import ProductSpecification, RequirementSpecification
    prod_count = db.query(ProductSpecification).count()
    req_count = db.query(RequirementSpecification).count()
    print(f"\n📦 Gesamt in Datenbank:")
    print(f"  Produktspezifikationen: {prod_count}")
    print(f"  Anforderungen:          {req_count}")

    db.close()
    print("\n✅ Fertig!")


if __name__ == "__main__":
    main()
