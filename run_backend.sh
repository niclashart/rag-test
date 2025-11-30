#!/bin/bash
# Script to run FastAPI backend
cd "$(dirname "$0")"
uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0

