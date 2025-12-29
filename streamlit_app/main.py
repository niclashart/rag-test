"""Streamlit main application."""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.pages import dashboard, chat

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
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
    /* Hide Streamlit page navigation tabs */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    /* Hide Streamlit header menu but keep sidebar toggle button */
    #MainMenu {
        visibility: hidden;
    }
    /* Keep header visible - needed for sidebar toggle button */
    header[data-testid="stHeader"] {
        visibility: visible !important;
        height: 3rem;
    }
    /* Ensure sidebar toggle button is always visible */
    button[title="View sidebar"],
    button[title="Close sidebar"],
    [data-testid="stSidebar"] button[aria-label*="sidebar"] {
        visibility: visible !important;
        display: block !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main app logic
def main():
    # Initialize page in session state if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Sidebar navigation - always visible
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Seite auswählen",
            ["Dashboard", "Chat"],
            index=0 if st.session_state.current_page == "Dashboard" else 1,
            label_visibility="collapsed",
            key="page_selector"
        )
        st.session_state.current_page = page
    
    # Show selected page
    if page == "Dashboard":
        dashboard.show_dashboard()
    elif page == "Chat":
        chat.show_chat()

if __name__ == "__main__":
    main()

