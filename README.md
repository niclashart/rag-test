# RAG Pipeline

Eine vollständige RAG (Retrieval-Augmented Generation) Pipeline zur kontextbasierten Beantwortung von Fragen auf Grundlage von Dokumentbeständen.

## Features

- **Multi-Format Support**: PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON
- **Erweiterte PDF-Verarbeitung**: OCR, Tabellenerkennung, automatische Entfernung von Kopf- und Fußzeilen
- **Semantische Suche**: Vector-basierte Suche mit ChromaDB
- **Reranking**: Cross-Encoder-basiertes Reranking für verbesserte Retrieval-Qualität
- **LLM-Integration**: GPT-4o-mini für Antwortgenerierung
- **Chat-Historie**: Unterstützung für kontextuelle Mehrfachrunden-Dialoge
- **RAGAS-Evaluation**: Automatisierte Evaluation mit RAGAS-Metriken
- **Streamlit & React Frontend**: Interaktive Benutzeroberflächen
- **RESTful API**: Vollständige OpenAPI-konforme REST-API

## Installation

1. **Repository klonen:**
```bash
git clone <repository-url>
cd rag-test
```

2. **Python-Umgebung erstellen:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

4. **Tesseract OCR installieren:**
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng`
   - macOS: `brew install tesseract`
   - Windows: Download von [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Environment-Variablen konfigurieren:**
```bash
cp .env.example .env
# .env Datei bearbeiten und OPENAI_API_KEY eintragen
```

## Verwendung

### Backend starten
```bash
uvicorn backend.main:app --reload --port 8000
# oder
./run_backend.sh
```

### Streamlit Frontend starten
```bash
streamlit run streamlit_app/main.py
# oder
./run_streamlit.sh
```

**Wichtig:** Starten Sie zuerst das Backend, bevor Sie das Streamlit-Frontend starten.

Die Anwendung ist dann unter `http://localhost:8501` erreichbar.

## Dokumente hochladen

### Über Streamlit Frontend
1. Öffnen Sie das Dashboard
2. Nutzen Sie die Upload-Funktion
3. Dokumente indizieren (einzeln oder alle auf einmal)

### Über Kommandozeile
```bash
python scripts/ingest.py testdaten/dokument.pdf --user-id 1
```

## Abfragen stellen

### Über Streamlit Frontend
1. Navigieren Sie zur Chat-Seite
2. Geben Sie Ihre Frage ein
3. Antworten mit Quellenverweisen werden angezeigt

### Über API
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Ihre Frage hier", "use_reranking": true}'
```

## Konfiguration

Die Systemkonfiguration erfolgt über `config/settings.yaml`:

```yaml
chunking:
  chunk_size: 1200
  chunk_overlap: 200

retrieval:
  top_k: 5
  use_reranking: true
  rerank_top_k: 15

qa:
  temperature: 0.3
  max_tokens: 2000
```

## Projektstruktur

```
rag-test/
├── backend/          # FastAPI Backend
├── streamlit_app/    # Streamlit Frontend
├── frontend/         # React Frontend (optional)
├── src/              # RAG Pipeline Module
│   ├── ingestion/   # Dokumentenlader
│   ├── chunking/    # Text-Chunking
│   ├── embeddings/  # Embedding-Generierung
│   ├── index/        # Vector Store (ChromaDB)
│   ├── retrieval/   # Dokumenten-Retrieval
│   ├── rerank/       # Reranking
│   └── qa/           # Question-Answering Chain
├── database/         # Datenbank-Models und CRUD
├── benchmarking/     # Evaluation & Benchmarking
├── scripts/          # CLI-Tools
├── config/           # Konfigurationsdateien
└── data/             # Datenverzeichnis
```

## API Dokumentation

Nach dem Start des Backends ist die interaktive API-Dokumentation unter `http://localhost:8000/docs` verfügbar.

### Wichtige Endpunkte

- `POST /api/documents/upload` - Dokument hochladen
- `GET /api/documents` - Alle Dokumente auflisten
- `POST /api/documents/{id}/ingest` - Dokument indizieren
- `POST /api/query` - RAG-Anfrage stellen
- `GET /api/query/history` - Query-Historie abrufen
- `POST /api/benchmark/run-from-file` - Benchmark ausführen

## Technologie-Stack

- **Backend**: FastAPI, SQLAlchemy, ChromaDB, LangChain
- **Frontend**: Streamlit, React (optional)
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: Sentence-Transformers
- **Evaluation**: RAGAS

## Testdaten

Das Projekt enthält Testdaten im Verzeichnis `testdaten/` mit über 77 PDF-Dateien (Laptop-Spezifikationen).

```bash
python scripts/ingest.py testdaten/ThinkPad_E14_Gen_6_Intel_Spec.pdf --user-id 1
```

## Lizenz

MIT
