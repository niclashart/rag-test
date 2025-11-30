#!/bin/bash
# Script to run Streamlit app
cd "$(dirname "$0")"
streamlit run streamlit_app/main.py --server.port 8501 --server.address 0.0.0.0

