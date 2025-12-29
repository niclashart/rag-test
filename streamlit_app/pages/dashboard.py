"""Dashboard page for document management."""
import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_BASE_URL = "http://localhost:8000"

def get_headers():
    """Get headers (no authentication needed)."""
    return {}

def show_dashboard():
    """Show dashboard page."""
    # Add navigation hint if sidebar might be collapsed
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("📁 Dokumente")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("☰ Menü", help="Öffnet die Sidebar-Navigation"):
            st.info("Bitte verwenden Sie das ☰ Symbol oben links, um die Sidebar zu öffnen")
    
    st.markdown("Laden Sie Dokumente hoch und indizieren Sie sie für die RAG-Pipeline")
    
    # File upload - multiple files
    st.subheader("Datei(en) hochladen")
    uploaded_files = st.file_uploader(
        "Wählen Sie eine oder mehrere Dateien aus",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "html", "json"],
        help="Unterstützte Formate: PDF, DOCX, TXT, MD, CSV, XLSX, HTML, JSON",
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        # Show selected files
        st.write(f"**{len(uploaded_files)} Datei(en) ausgewählt:**")
        for file in uploaded_files:
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            st.write(f"- {file.name} ({file_size_mb:.2f} MB)")
        
        if st.button("Alle hochladen", use_container_width=True, type="primary"):
            upload_progress = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            error_count = 0
            errors = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Hochladen: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})...")
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_BASE_URL}/api/documents/upload",
                        files=files,
                        headers=get_headers(),
                        timeout=60
                    )
                    
                    if response.status_code == 201:
                        success_count += 1
                    else:
                        error_count += 1
                        try:
                            error_detail = response.json().get('detail', f'HTTP {response.status_code}')
                        except (ValueError, requests.exceptions.JSONDecodeError):
                            error_detail = f"HTTP {response.status_code}: {response.text[:100]}"
                        errors.append(f"{uploaded_file.name}: {error_detail}")
                    
                    upload_progress.progress((idx + 1) / len(uploaded_files))
                    
                except requests.exceptions.ConnectionError:
                    error_count += 1
                    errors.append(f"{uploaded_file.name}: Backend nicht erreichbar")
                except requests.exceptions.Timeout:
                    error_count += 1
                    errors.append(f"{uploaded_file.name}: Zeitüberschreitung")
                except Exception as e:
                    error_count += 1
                    errors.append(f"{uploaded_file.name}: {str(e)}")
            
            # Show results
            upload_progress.empty()
            status_text.empty()
            
            if success_count > 0:
                st.success(f"✅ {success_count} Datei(en) erfolgreich hochgeladen!")
            
            if error_count > 0:
                st.error(f"❌ {error_count} Datei(en) konnten nicht hochgeladen werden:")
                for error in errors:
                    st.error(f"  - {error}")
            
            if success_count > 0:
                st.rerun()
    
    st.divider()
    
    # Document list
    st.subheader("Meine Dokumente")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/documents",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            try:
                documents = response.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                st.error("Ungültige Antwort vom Backend. Bitte überprüfen Sie die Backend-Logs.")
                return
            
            if len(documents) == 0:
                st.info("Noch keine Dokumente hochgeladen")
            else:
                # Count non-indexed documents
                non_indexed = [doc for doc in documents if doc['status'] != 'indexed']
                indexed = [doc for doc in documents if doc['status'] == 'indexed']
                
                # Show summary and bulk ingest button
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Gesamt:** {len(documents)} Dokument(e) | "
                            f"✅ Indiziert: {len(indexed)} | "
                            f"⏳ Nicht indiziert: {len(non_indexed)}")
                with col2:
                    if len(non_indexed) > 0:
                        if st.button("🔄 Alle indizieren", use_container_width=True, type="primary"):
                            ingest_all_documents()
                
                if len(non_indexed) > 0:
                    st.divider()
                
                for doc in documents:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{doc['filename']}**")
                            st.caption(f"{doc['file_type'].upper()} • {format_file_size(doc['file_size'])} • {doc['status']}")
                        
                        with col2:
                            if doc['status'] != 'indexed':
                                if st.button("Indizieren", key=f"ingest_{doc['id']}", use_container_width=True):
                                    ingest_document(doc['id'])
                        with col3:
                            if st.button("Vorschau", key=f"preview_{doc['id']}", use_container_width=True):
                                st.session_state.preview_doc_id = doc['id']
                                st.rerun()
                        with col4:
                            if st.button("Löschen", key=f"delete_{doc['id']}", use_container_width=True):
                                delete_document(doc['id'])
                        
                        st.divider()
        else:
            try:
                error_detail = response.json().get('detail', f'HTTP {response.status_code}')
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_detail = f"HTTP {response.status_code}: {response.text[:200]}"
            st.error(f"Fehler beim Laden der Dokumente: {error_detail}")
    except requests.exceptions.ConnectionError:
        st.error("Verbindungsfehler: Backend nicht erreichbar. Bitte starten Sie das Backend mit './run_backend.sh'")
    except requests.exceptions.Timeout:
        st.error("Zeitüberschreitung: Das Backend antwortet nicht.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")
    
    # PDF Preview
    if 'preview_doc_id' in st.session_state:
        st.divider()
        st.subheader("PDF-Vorschau")
        show_pdf_preview(st.session_state.preview_doc_id)

def format_file_size(bytes_size):
    """Format file size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def ingest_document(doc_id):
    """Ingest a document."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/documents/{doc_id}/ingest",
            headers=get_headers(),
            timeout=300  # 5 minutes for large documents
        )
        
        if response.status_code == 200:
            st.success("Dokument wird indiziert...")
            st.rerun()
        else:
            try:
                error_detail = response.json().get('detail', f'HTTP {response.status_code}')
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_detail = f"HTTP {response.status_code}: {response.text[:200]}"
            st.error(f"Fehler: {error_detail}")
    except requests.exceptions.Timeout:
        st.error("Zeitüberschreitung: Die Indizierung dauert zu lange. Bitte versuchen Sie es später erneut.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

def delete_document(doc_id):
    """Delete a document."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/documents/{doc_id}",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 204:
            st.success("Dokument gelöscht")
            st.rerun()
        else:
            try:
                error_detail = response.json().get('detail', f'HTTP {response.status_code}')
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_detail = f"HTTP {response.status_code}: {response.text[:200]}"
            st.error(f"Fehler: {error_detail}")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

def ingest_all_documents():
    """Ingest all non-indexed documents."""
    try:
        with st.spinner("Indiziere alle Dokumente..."):
            response = requests.post(
                f"{API_BASE_URL}/api/documents/ingest-all",
                headers=get_headers(),
                timeout=600  # 10 minutes for multiple documents
            )
            
            if response.status_code == 200:
                data = response.json()
                success_count = data.get("success_count", 0)
                error_count = data.get("error_count", 0)
                
                if success_count > 0:
                    st.success(f"✅ {success_count} Dokument(e) erfolgreich indiziert!")
                    if data.get("success"):
                        with st.expander("Indizierte Dokumente anzeigen"):
                            for item in data["success"]:
                                st.write(f"- {item['filename']} ({item['chunks_count']} Chunks)")
                
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} Dokument(e) konnten nicht indiziert werden:")
                    if data.get("errors"):
                        for error in data["errors"]:
                            st.error(f"  - {error['filename']}: {error['error']}")
                
                st.rerun()
            else:
                try:
                    error_detail = response.json().get('detail', f'HTTP {response.status_code}')
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_detail = f"HTTP {response.status_code}: {response.text[:200]}"
                st.error(f"Fehler: {error_detail}")
    except requests.exceptions.Timeout:
        st.error("Zeitüberschreitung: Die Indizierung dauert zu lange. Bitte versuchen Sie es später erneut.")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

def show_pdf_preview(doc_id):
    """Show PDF preview."""
    try:
        preview_url = f"{API_BASE_URL}/api/documents/{doc_id}/preview"
        headers = get_headers()
        
        # Get PDF content
        response = requests.get(preview_url, headers=headers, stream=True)
        
        if response.status_code == 200:
            # Display PDF download button
            st.download_button(
                "📥 PDF herunterladen",
                response.content,
                file_name=f"document_{doc_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            # Try to show PDF preview using PyMuPDF
            try:
                import fitz  # PyMuPDF
                import io
                from PIL import Image
                
                pdf_bytes = io.BytesIO(response.content)
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                page_num = st.number_input(
                    "Seite",
                    min_value=1,
                    max_value=len(pdf_document),
                    value=1,
                    key=f"page_{doc_id}"
                )
                
                page = pdf_document[page_num - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                st.image(img, caption=f"Seite {page_num} von {len(pdf_document)}")
                
                pdf_document.close()
            except ImportError:
                st.info("PyMuPDF nicht verfügbar. PDF-Vorschau nicht möglich.")
            except Exception as e:
                st.warning(f"PDF-Vorschau fehlgeschlagen: {str(e)}")
        else:
            st.error("Fehler beim Laden der PDF-Vorschau")
    except Exception as e:
        st.error(f"Fehler: {str(e)}")

