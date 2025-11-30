# RAG Pipeline mit Authentifizierung und Benchmarking

Eine vollständige RAG (Retrieval Augmented Generation) Pipeline mit User-Authentifizierung, Dokument-Management, PDF-Preview und RAGAS Benchmarking.

## Features

- JWT-basierte User-Authentifizierung
- Unterstützung für PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON
- PDF-Preview mit Chunk-Highlighting
- LangChain-basierte RAG-Pipeline
- RAGAS Benchmarking und Visualisierung
- Strukturiertes Logging
- Streamlit Frontend

## Installation

1. Repository klonen und in das Verzeichnis wechseln:
```bash
cd RAG_test
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

## Projektstruktur

- `backend/` - FastAPI Backend
- `streamlit_app/` - Streamlit Frontend
- `src/` - RAG Pipeline Module
- `database/` - Datenbank-Models und CRUD
- `benchmarking/` - RAGAS Evaluation
- `scripts/` - CLI Tools
- `config/` - Konfigurationsdateien

## API Dokumentation

Nach dem Start des Backends ist die API-Dokumentation unter `http://localhost:8000/docs` verfügbar.

## Lizenz

MIT


