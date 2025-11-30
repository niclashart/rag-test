"""Streamlit main application."""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.pages import login, dashboard, chat, benchmark

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

# Main app logic
def main():
    if not st.session_state.authenticated:
        login.show_login_page()
    else:
        # Sidebar navigation
        with st.sidebar:
            st.title("Navigation")
            page = st.radio(
                "Seite auswählen",
                ["Dashboard", "Chat", "Benchmark"],
                label_visibility="collapsed"
            )
            
            st.divider()
            if st.button("Abmelden", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.access_token = None
                st.rerun()
        
        # Show selected page
        if page == "Dashboard":
            dashboard.show_dashboard()
        elif page == "Chat":
            chat.show_chat()
        elif page == "Benchmark":
            benchmark.show_benchmark()

if __name__ == "__main__":
    main()

