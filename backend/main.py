"""FastAPI main application."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database import init_db
from backend.api import auth, documents, query, benchmark
from logging_config.logger import get_logger

logger = get_logger(__name__)

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="RAG Pipeline with Authentication and Benchmarking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:5173", "http://localhost:3000"],  # Streamlit and React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(benchmark.router)


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "RAG Pipeline API", "version": "1.0.0"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


