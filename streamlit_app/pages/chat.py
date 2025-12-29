"""Chat page for RAG queries."""
import streamlit as st
import requests
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_BASE_URL = "http://localhost:8000"

def get_headers():
    """Get headers (no authentication needed)."""
    return {}

def get_query_statistics():
    """Get query statistics from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/query/history",
            headers=get_headers(),
            params={"limit": 1000},  # Get more history for statistics
            timeout=10
        )
        if response.status_code == 200:
            history = response.json()
            return history
        return []
    except Exception as e:
        # Don't show error in sidebar, just return empty list
        return []

def show_chat():
    """Show chat page."""
    st.title("💬 RAG Chat")
    
    # Add important notice about Lenovo laptop format
    st.info("""
    **📝 Wichtiger Hinweis:** 
    Bei Fragen zu Lenovo Laptops müssen diese im folgenden Format angegeben werden:
    - **"Lenovo ThinkPad E14 Gen 6 (Intel)"** oder
    - **"Lenovo ThinkPad E14 Gen 6 (AMD)"**
    
    Beispiel: "Welche Prozessoren unterstützt das Lenovo ThinkPad E14 Gen 6 (Intel)?"
    """)
    
    st.markdown("Stellen Sie Fragen zu den indizierten Dokumenten")
    
    # Initialize chat history and statistics
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_times" not in st.session_state:
        st.session_state.query_times = []
    if "query_timestamps" not in st.session_state:
        st.session_state.query_timestamps = []
    
    # Sidebar with statistics and controls
    with st.sidebar:
        st.subheader("Informationen")
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/documents",
                headers=get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                documents = response.json()
                indexed_docs = [doc for doc in documents if doc['status'] == 'indexed']
                
                if indexed_docs:
                    st.success(f"✅ {len(indexed_docs)} indizierte Dokument(e) verfügbar")
                    st.info("Alle Dokumente werden bei der Suche berücksichtigt.")
                else:
                    st.warning("Keine indizierten Dokumente verfügbar")
                    st.info("Bitte indizieren Sie zuerst Dokumente im Dashboard")
        except Exception as e:
            st.error(f"Fehler beim Laden der Dokumente: {str(e)}")
        
        st.divider()
        
        # Statistics section
        st.subheader("📊 Statistiken")
        
        # Get query history for total count and response times
        history = get_query_statistics()
        total_queries = len(history)
        
        # Extract response times from history metadata
        response_times_from_db = []
        for item in history:
            metadata = item.get("query_metadata", {})
            if metadata:
                total_time = metadata.get("total_time", 0.0)
                if total_time > 0:
                    response_times_from_db.append(total_time)
        
        # Combine with session state times (for current session)
        query_times = st.session_state.get("query_times", [])
        all_times = response_times_from_db + query_times
        
        # Display total queries
        st.metric("Gesamt gestellte Anfragen", total_queries)
        
        # Calculate and display average response time
        if all_times:
            avg_time = sum(all_times) / len(all_times)
            st.metric("Durchschnittliche Antwortzeit", f"{avg_time:.2f}s")
            
            # Show response time graph (use last 50 for better visualization)
            if len(all_times) > 1:
                st.subheader("📈 Antwortzeit-Entwicklung")
                # Create DataFrame for the graph
                times_to_show = all_times[-50:] if len(all_times) > 50 else all_times
                df = pd.DataFrame({
                    'Anfrage': range(1, len(times_to_show) + 1),
                    'Antwortzeit (s)': times_to_show
                })
                st.line_chart(df.set_index('Anfrage'))
        else:
            st.info("Noch keine Statistiken verfügbar")
        
        st.divider()
        
        # Clear chat history button (permanently visible)
        st.subheader("⚙️ Einstellungen")
        if st.button("🗑️ Chat-History löschen", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.query_times = []
            st.session_state.query_timestamps = []
            st.success("Chat-History gelöscht!")
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Quellen anzeigen", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        with st.container():
                            # Use original source number if available, otherwise use index
                            source_num = source.get('source_number', i)
                            st.markdown(f"### Quelle {source_num}")
                            
                            # Source metadata
                            if source.get('page_number'):
                                st.caption(f"📄 Seite {source.get('page_number')}")
                            if source.get('document_id'):
                                st.caption(f"📑 Dokument ID: {source.get('document_id')}")
                            
                            # Display chunk text if available
                            if source.get('text'):
                                st.markdown("**Text-Ausschnitt:**")
                                # Create unique key using chunk_id, message index, source index, and source number
                                # Use hash of message content + chunk_id + indices to ensure uniqueness
                                message_idx = st.session_state.messages.index(message)
                                source_num = source.get('source_number', i)
                                chunk_id = source.get('chunk_id', f'idx_{i}')
                                # Create a hash-based unique identifier
                                import hashlib
                                key_string = f"{message_idx}_{chunk_id}_{i}_{source_num}_{message.get('content', '')[:20]}"
                                key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
                                unique_key = f"hist_source_{key_hash}_{message_idx}_{i}"
                                st.text_area(
                                    label=f"Text aus Quelle {source_num}",
                                    value=source.get('text'),
                                    height=300,  # Increased height for better table visibility
                                    disabled=True,
                                    label_visibility="collapsed",
                                    key=unique_key
                                )
                            else:
                                st.caption(f"Chunk ID: {source.get('chunk_id', 'N/A')}")
                            
                            if i < len(message["sources"]):
                                st.divider()
    
    # Chat input
    if prompt := st.chat_input("Stellen Sie eine Frage..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                try:
                    # Prepare chat history (last 10 messages for context)
                    chat_history = []
                    for msg in st.session_state.messages[-10:]:
                        chat_history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    response = requests.post(
                        f"{API_BASE_URL}/api/query",
                        json={
                            "query": prompt,
                            "use_reranking": True,
                            "chat_history": chat_history
                        },
                        headers=get_headers()
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                        retrieval_time = data.get("retrieval_time", 0.0)
                        generation_time = data.get("generation_time", 0.0)
                        total_time = retrieval_time + generation_time
                        
                        # Track response time for statistics
                        st.session_state.query_times.append(total_time)
                        st.session_state.query_timestamps.append(time.time())
                        
                        if answer and answer.strip():
                            st.markdown(answer)
                        else:
                            st.warning("Die Antwort ist leer. Bitte versuchen Sie es mit einer anderen Formulierung.")
                            answer = "Keine Antwort generiert."
                        
                        if sources:
                            with st.expander("📚 Quellen anzeigen", expanded=False):
                                for i, source in enumerate(sources, 1):
                                    with st.container():
                                        # Use original source number if available, otherwise use index
                                        source_num = source.get('source_number', i)
                                        st.markdown(f"### Quelle {source_num}")
                                        
                                        # Source metadata
                                        if source.get('page_number'):
                                            st.caption(f"📄 Seite {source.get('page_number')}")
                                        if source.get('document_id'):
                                            st.caption(f"📑 Dokument ID: {source.get('document_id')}")
                                        
                                        # Display chunk text if available
                                        if source.get('text'):
                                            st.markdown("**Text-Ausschnitt:**")
                                            # Create unique key using chunk_id, timestamp, and source index
                                            import hashlib
                                            chunk_id = source.get('chunk_id', f'idx_{i}')
                                            source_num = source.get('source_number', i)
                                            timestamp = int(time.time() * 1000)
                                            # Create a hash-based unique identifier
                                            key_string = f"{chunk_id}_{i}_{source_num}_{timestamp}_{answer[:20]}"
                                            key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
                                            unique_key = f"source_{key_hash}_{i}_{timestamp}"
                                            st.text_area(
                                                label=f"Text aus Quelle {source_num}",
                                                value=source.get('text'),
                                                height=300,  # Increased height for better table visibility
                                                disabled=True,
                                                label_visibility="collapsed",
                                                key=unique_key
                                            )
                                        else:
                                            st.caption(f"Chunk ID: {source.get('chunk_id', 'N/A')}")
                                        
                                        if i < len(sources):
                                            st.divider()
                        else:
                            st.info("Keine Quellen verfügbar.")
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = response.json().get('detail', 'Unbekannter Fehler')
                        st.error(f"Fehler: {error_msg}")
                except Exception as e:
                    st.error(f"Fehler: {str(e)}")

