# RAG Pipeline 

Eine vollständige RAG (Retrieval Augmented Generation) Pipeline mit Dokument-Management, PDF-Preview und LangChain-basierter RAG-Implementierung.

## Features

- Unterstützung für PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON
- PDF-Preview mit Chunk-Highlighting
- LangChain-basierte RAG-Pipeline
- Vector-basierte Suche mit ChromaDB
- Reranking für verbesserte Retrieval-Qualität
- Strukturiertes Logging
- Streamlit Frontend

## Installation

1. Repository klonen und in das Verzeichnis wechseln:
```bash
cd rag-test
```

2. Python-Umgebung erstellen und aktivieren:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

3. Dependencies installieren:
```bash
pip install -r requirements.txt
```

4. Environment-Variablen konfigurieren:
```bash
cp .env.example .env
# .env Datei bearbeiten und OPENAI_API_KEY eintragen
# Empfohlenes Modell: gpt-4o-mini (Reasoning-Modelle sind aufgrund zu hohem Tokenkonsum ungeeignet)
```

## Verwendung

### Backend starten:
```bash
# Vom Root-Verzeichnis aus:
uvicorn backend.main:app --reload --port 8000
# oder
./run_backend.sh
```

### Streamlit Frontend starten:
```bash
streamlit run streamlit_app/main.py
# oder
./run_streamlit.sh
```

Die Anwendung ist dann unter `http://localhost:8501` erreichbar.

**Wichtig:** Starten Sie zuerst das Backend, bevor Sie das Streamlit-Frontend starten.

## Dokumente hochladen

### Über das Streamlit Frontend

1. Öffnen Sie das Dashboard im Streamlit Frontend
2. Nutzen Sie die Upload-Funktion, um Dokumente hochzuladen
3. Die Dokumente werden automatisch verarbeitet, in Chunks aufgeteilt und in die Vector-Datenbank indexiert

### Über die Kommandozeile

Verwenden Sie das `ingest.py` Skript, um Dokumente direkt zu verarbeiten:

```bash
python scripts/ingest.py <pfad-zum-dokument> --user-id 1 --init-db
```

Beispiel:
```bash
python scripts/ingest.py testdaten/dokument.pdf --user-id 1
```

### Testdaten

Das Projekt enthält Testdaten im Verzeichnis `testdaten/` mit 77 PDF-Dateien. Diese können für Tests und Demonstrationen verwendet werden:

```bash
# Einzelnes Dokument ingestieren
python scripts/ingest.py testdaten/dokument.pdf --user-id 1

# Alle Testdaten verarbeiten (Beispiel mit Schleife)
for file in testdaten/*.pdf; do
    python scripts/ingest.py "$file" --user-id 1
done
```

## Abfragen stellen

### Über das Streamlit Frontend

1. Navigieren Sie zur Chat-Seite im Streamlit Frontend
2. Geben Sie Ihre Frage in das Eingabefeld ein
3. Die RAG-Pipeline sucht relevante Dokumenten-Chunks und generiert eine Antwort

### Über die API

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Ihre Frage hier", "user_id": 1}'
```

### Über die Kommandozeile

```bash
python scripts/query.py "Ihre Frage hier"
```

## Projektstruktur

- `backend/` - FastAPI Backend mit REST API
- `streamlit_app/` - Streamlit Frontend für interaktive Nutzung
- `src/` - RAG Pipeline Module
  - `ingestion/` - Dokumentenlader und PDF-Prozessoren
  - `chunking/` - Text-Chunking
  - `embeddings/` - Embedding-Generierung
  - `index/` - Vector Store (ChromaDB)
  - `retrieval/` - Dokumenten-Retrieval
  - `rerank/` - Reranking für bessere Ergebnisse
  - `qa/` - Question-Answering Chain
- `database/` - Datenbank-Models und CRUD-Operationen
- `scripts/` - CLI Tools für Dokumenten-Ingestion und Abfragen
- `config/` - Konfigurationsdateien
- `testdaten/` - Test-PDF-Dateien (77 Dokumente)
- `data/` - Datenbanken und gespeicherte Daten
  - `chroma_db/` - ChromaDB Vector Store
  - `rag_pipeline.db` - SQLite Datenbank
  - `uploads/` - Hochgeladene Dokumente

## API Dokumentation

Nach dem Start des Backends ist die interaktive API-Dokumentation unter `http://localhost:8000/docs` verfügbar.

## Lizenz

MIT


