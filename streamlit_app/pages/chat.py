"""Chat page for RAG queries."""
import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_BASE_URL = "http://localhost:8000"

def get_headers():
    """Get authorization headers."""
    return {"Authorization": f"Bearer {st.session_state.access_token}"}

def show_chat():
    """Show chat page."""
    st.title("💬 RAG Chat")
    st.markdown("Stellen Sie Fragen zu Ihren indizierten Dokumenten")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Check if user has indexed documents
    with st.sidebar:
        st.subheader("Informationen")
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/documents",
                headers=get_headers(),
                timeout=30  # Increased timeout for document loading
            )
            
            if response.status_code == 200:
                documents = response.json()
                indexed_docs = [doc for doc in documents if doc['status'] == 'indexed']
                
                if indexed_docs:
                    st.success(f"✅ {len(indexed_docs)} indizierte Dokument(e) verfügbar")
                    st.info("Alle Ihre Dokumente werden bei der Suche berücksichtigt.")
                else:
                    st.warning("Keine indizierten Dokumente verfügbar")
                    st.info("Bitte indizieren Sie zuerst Dokumente im Dashboard")
        except Exception as e:
            st.error(f"Fehler beim Laden der Dokumente: {str(e)}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Quellen anzeigen"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Quelle {i}:**")
                        st.write(f"- Dokument ID: {source.get('document_id', 'N/A')}")
                        st.write(f"- Seite: {source.get('page_number', 'N/A')}")
                        if source.get('similarity'):
                            st.write(f"- Ähnlichkeit: {source['similarity']*100:.1f}%")
    
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
                        
                        if answer and answer.strip():
                            st.markdown(answer)
                        else:
                            st.warning("Die Antwort ist leer. Bitte versuchen Sie es mit einer anderen Formulierung.")
                            answer = "Keine Antwort generiert."
                        
                        if sources:
                            with st.expander("Quellen anzeigen"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"**Quelle {i}:**")
                                    st.write(f"- Chunk ID: {source.get('chunk_id', 'N/A')}")
                                    st.write(f"- Dokument ID: {source.get('document_id', 'N/A')}")
                                    st.write(f"- Seite: {source.get('page_number', 'N/A')}")
                                    if source.get('similarity'):
                                        st.write(f"- Ähnlichkeit: {source['similarity']*100:.1f}%")
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
    
    # Clear chat button
    if st.button("Chat leeren", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

